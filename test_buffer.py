import time

import models
import ray
import shared_storage
import replay_buffer
import self_play
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

    # checkpoint = torch.load('./test_checkpoint/model.checkpoint')
    # checkpoint["num_played_games"] = 0
    # checkpoint["num_played_steps"] = 0

    shared_storage_worker = shared_storage.SharedStorage.remote(
        checkpoint,
        config,
    )
    shared_storage_worker.set_info.remote("terminate", False)
    replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
        checkpoint, buffer, config
    )
    self_play_worker = self_play.SelfPlay.remote(
        checkpoint,
        Game,
        config,
        config.seed,
    )

    self_play_worker.continuous_self_play.remote(
        shared_storage_worker, replay_buffer_worker
    )

    while(ray.get(shared_storage_worker.get_info.remote("num_played_games")) <= 50):
        print(ray.get(shared_storage_worker.get_info.remote("num_played_games")))
        time.sleep(0.5)

    batch = ray.get(replay_buffer_worker.get_batch.remote())