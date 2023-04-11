import replay_buffer_sp
import trainer_sp
import self_play_sp
import shared_storage_sp
import models
from torch.utils.tensorboard import SummaryWriter

import numpy
import torch

from games import tictactoe

def print_info():
    writer = SummaryWriter(config.results_path)

    print(
        "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
    )
    pass

if __name__ == "__main__":
    buffer = {}
    Game = tictactoe.Game
    config = tictactoe.MuZeroConfig()
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
    config.results_path.mkdir(parents=True, exist_ok=True)
    model = models.MuZeroNetwork(config)
    weight = model.get_weights()
    checkpoint["weights"] = weight

    shared = shared_storage_sp.SharedStorage(checkpoint, config)
    replaybuffer = replay_buffer_sp.ReplayBuffer(checkpoint, buffer, config)
    selfplay = self_play_sp.SelfPlay(checkpoint, Game, config, config.seed)
    train= trainer_sp.Trainer(checkpoint, config)

    print(shared.get_info("training_step"))

    while shared.get_info("training_step") <= config.training_steps:
        selfplay.nums_self_play(20, shared, replaybuffer)
        train.nums_update_weights(10, replaybuffer, shared)
        print('trainging_step:{}'.format(shared.get_info("training_step")), 'played_games:{}'.format(shared.get_info("num_played_games")))
