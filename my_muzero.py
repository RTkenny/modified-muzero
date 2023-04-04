import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from games import tictactoe
import diagnose_model
import models
import replay_buffer
import self_play
import shared_storage
import trainer

@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary

def logging_loop(checkpoint, Game, config, shared_storage_worker, buffer):
    test_worker = self_play.SelfPlay.options(
        num_cpus=0,
        num_gpus=0,
    ).remote(
        checkpoint,
        Game,
        config,
        config.seed,
    )
    test_worker.continuous_self_play.remote(
        shared_storage_worker, None, True
    )

    # Write everything in TensorBoard
    # writer = SummaryWriter(config.results_path)
    #
    # print(
    #     "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
    # )
    #
    # # Save hyperparameters to TensorBoard
    # hp_table = [
    #     f"| {key} | {value} |" for key, value in config.__dict__.items()
    # ]
    # writer.add_text(
    #     "Hyperparameters",
    #     "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
    # )
    # Save model representation
    # Loop for updating the training performance
    counter = 0
    keys = [
        "total_reward",
        "muzero_reward",
        "opponent_reward",
        "episode_length",
        "mean_value",
        "training_step",
        "lr",
        "total_loss",
        "value_loss",
        "reward_loss",
        "policy_loss",
        "num_played_games",
        "num_played_steps",
        "num_reanalysed_games",
    ]
    info = ray.get(shared_storage_worker.get_info.remote(keys))
    try:
        while info["training_step"] < config.training_steps:
            info = ray.get(shared_storage_worker.get_info.remote(keys))
            # writer.add_scalar(
            #     "1.Total_reward/1.Total_reward",
            #     info["total_reward"],
            #     counter,
            # )
            # writer.add_scalar(
            #     "1.Total_reward/2.Mean_value",
            #     info["mean_value"],
            #     counter,
            # )
            # writer.add_scalar(
            #     "1.Total_reward/3.Episode_length",
            #     info["episode_length"],
            #     counter,
            # )
            # writer.add_scalar(
            #     "1.Total_reward/4.MuZero_reward",
            #     info["muzero_reward"],
            #     counter,
            # )
            # writer.add_scalar(
            #     "1.Total_reward/5.Opponent_reward",
            #     info["opponent_reward"],
            #     counter,
            # )
            # writer.add_scalar(
            #     "2.Workers/1.Self_played_games",
            #     info["num_played_games"],
            #     counter,
            # )
            # writer.add_scalar(
            #     "2.Workers/2.Training_steps", info["training_step"], counter
            # )
            # writer.add_scalar(
            #     "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
            # )
            # writer.add_scalar(
            #     "2.Workers/4.Reanalysed_games",
            #     info["num_reanalysed_games"],
            #     counter,
            # )
            # writer.add_scalar(
            #     "2.Workers/5.Training_steps_per_self_played_step_ratio",
            #     info["training_step"] / max(1, info["num_played_steps"]),
            #     counter,
            # )
            # writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
            # writer.add_scalar(
            #     "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
            # )
            # writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
            # writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
            # writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
            print(
                f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                end="\r",
            )
            counter += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


    if config.save_model:
        # Persist replay buffer to disk
        path = config.results_path / "replay_buffer.pkl"
        print(f"\n\nPersisting replay buffer games to disk at {path}")
        pickle.dump(
            {
                "buffer": buffer,
                "num_played_games": checkpoint["num_played_games"],
                "num_played_steps": checkpoint["num_played_steps"],
                "num_reanalysed_games": checkpoint["num_reanalysed_games"],
            },
            open(path, "wb"),
        )

if __name__ == "__main__":
    checkpoint = {
                "weights": None,
                "optimizer_state": None,
                "total_reward": 0,
                "muzero_reward": 0,
                "opponent_reward": 0,
                "episode_length": 0,
                "mean_value": 0,
                "training_step": 0,
                "lr": 0,
                "total_loss": 0,
                "value_loss": 0,
                "reward_loss": 0,
                "policy_loss": 0,
                "num_played_games": 0,
                "num_played_steps": 0,
                "num_reanalysed_games": 0,
                "terminate": False,
    }

    buffer = {}
    config = tictactoe.MuZeroConfig()
    config.results_path.mkdir(parents=True, exist_ok=True)
    Game = tictactoe.Game

    ray.init()

    cpu_actor = CPUActor.remote()
    cpu_weights = cpu_actor.get_initial_weights.remote(config)
    checkpoint["weights"], summary = copy.deepcopy(ray.get(cpu_weights))

    training_worker = trainer.Trainer.options(num_gpus=1).remote(checkpoint, config)

    shared_storage_worker = shared_storage.SharedStorage.remote(
        checkpoint,
        config,
    )
    shared_storage_worker.set_info.remote("terminate", False)

    replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
        checkpoint, buffer, config
    )
    self_play_worker =self_play.SelfPlay.remote(
                    checkpoint,
                    Game,
                    config,
                    config.seed,
    )

    self_play_worker.continuous_self_play.remote(
            shared_storage_worker, replay_buffer_worker
    )

    training_worker.continuous_update_weights.remote(
        replay_buffer_worker, shared_storage_worker
    )
    logging_loop(checkpoint,Game,config, shared_storage_worker, buffer)
