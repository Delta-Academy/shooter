import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).parent.parent.resolve()


class ChooseMoveCheckpoint:
    def __init__(self, checkpoint_name: str, choose_move: Callable):
        self.neural_network = copy.deepcopy(load_checkpoint(checkpoint_name))
        self._choose_move = choose_move

    def __call__(self, state: np.ndarray) -> int:
        return self._choose_move(state, self.neural_network)


def checkpoint_model(model: nn.Module, checkpoint_name: str) -> None:
    torch.save(model, HERE / checkpoint_name)


def load_checkpoint(checkpoint_name: str) -> nn.Module:
    return torch.load(HERE / checkpoint_name)


def load_network(team_name: str, network_folder: Path = HERE) -> nn.Module:
    net_path = network_folder / f"{team_name}_network.pt"
    assert (
        net_path.exists()
    ), f"Network saved using TEAM_NAME='{team_name}' doesn't exist! ({net_path})"
    model = torch.load(net_path)
    model.eval()
    return model


def save_network(network: nn.Module, team_name: str) -> None:
    assert isinstance(
        network, nn.Module
    ), f"train() function outputs an network type: {type(network)}"
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"
    net_path = HERE / f"{team_name}_network.pt"
    n_retries = 5
    for attempt in range(n_retries):
        try:
            torch.save(network, net_path)
            load_network(team_name)
            return
        except Exception:
            if attempt == n_retries - 1:
                raise
