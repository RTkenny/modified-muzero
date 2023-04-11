import replay_buffer_sp
import trainer_sp
import self_play_sp
import models

import numpy
import torch

from games import tictactoe

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
    model = models.MuZeroNetwork(config)
    weigths = model.get_weights()
    checkpoint["weights"] = weigths