import self_play_sp
import copy
from games import tictactoe
import models
import torch

if __name__ == "__main__":
    config = tictactoe.MuZeroConfig()
    model = models.MuZeroNetwork(config)
    weigths = model.get_weights()
    Game = tictactoe.Game
    checkpoint = torch.load('./test_checkpoint/model.checkpoint')
    # checkpoint = {
    #             "weights": weigths,
    #             "optimizer_state": None,
    #             "total_reward": 0,
    #             "muzero_reward": 0,
    #             "opponent_reward": 0,
    #             "episode_length": 0,
    #             "mean_value": 0,
    #             "training_step": 0,
    #             "lr": 0,
    #             "total_loss": 0,
    #             "value_loss": 0,
    #             "reward_loss": 0,
    #             "policy_loss": 0,
    #             "num_played_games": 0,
    #             "num_played_steps": 0,
    #             "num_reanalysed_games": 0,
    #             "terminate": False,
    # }
    # checkpoint["weights"] = copy.deepcopy(weigths)
    self_play_worker = self_play_sp.SelfPlay(checkpoint, Game, config, config.seed)
    gamehistory = self_play_worker.play_game(1, config.temperature_threshold, True, 'self', 0)

