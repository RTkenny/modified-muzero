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

import diagnose_model
import models
import replay_buffer
import self_play
import shared_storage
import trainer

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
                config.seed + seed,
            )