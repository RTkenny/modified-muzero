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
Game = tictactoe.Game()

ray.init()
training_worker = trainer.Trainer.remote(checkpoint, config)

shared_storage_worker = shared_storage.SharedStorage.remote(
    checkpoint,
    config,
)
shared_storage_worker.set_info.remote("terminate", False)

replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
    checkpoint, replay_buffer, config
)
self_play_workers =self_play.SelfPlay.remote(
                checkpoint,
                Game,
                config,
                config,
)