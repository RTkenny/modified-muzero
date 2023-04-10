import models
import shared_storage
import replay_buffer
import self_play
import torch
from games import tictactoe
if __name__ == "__main__":
    buffer = {}
    Game = tictactoe.Game
    config = tictactoe.MuZeroConfig()
    checkpoint = torch.load('./test_checkpoint/model.checkpoint')
    checkpoint["num_played_games"] = 0
    checkpoint["num_played_steps"] = 0
    # shared_storage_worker = shared_storage.SharedStorage.remote(
    #     checkpoint,
    #     config,
    # )
    # shared_storage_worker.set_info.remote("terminate", False)
    # replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
    #     checkpoint, buffer, config
    # )
    # self_play_worker = self_play.SelfPlay.remote(
    #     checkpoint,
    #     Game,
    #     config,
    #     config.seed,
    # )
    #
    # self_play_worker.continuous_self_play.remote(
    #     shared_storage_worker, replay_buffer_worker
    # )
    # replay_buffer_worker.get_batch.remote()