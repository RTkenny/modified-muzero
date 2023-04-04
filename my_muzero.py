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

def logging_loop():

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
    time.sleep(5)