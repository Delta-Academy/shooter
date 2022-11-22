import random

import numpy as np
import torch
from torch import nn

from check_submission import check_submission
from game_mechanics import (
    ShooterEnv,
    choose_move_randomly,
    human_player,
    load_network,
    play_shooter,
    save_network,
)

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> nn.Module:
    """
    TODO: Write this function to train your algorithm.

    Returns:
        A pytorch network to be used by choose_move. You can architect
        this however you like but your choose_move function must be able
        to use it.
    """
    raise NotImplementedError("You need to implement this function!")


def choose_move(
    state: np.ndarray,
    neural_network: nn.Module,
) -> int:  # <--------------- Please do not change these arguments!
    """Called during competitive play. It acts greedily given current state and neural network
    function dictionary. It returns a single action to take.

    Args:
        state: State of the game as a np array, length = 18.
        neural_network: The pytorch network output by train().

    Returns:
        move (int): The move you want to given the state of the game.
                    Should be in {0,1,2,3,4,5}.
    """
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    # Example workflow, feel free to edit this! ###
    my_network = train()
    save_network(my_network, TEAM_NAME)

    # Make sure this does not error! Or your
    # submission will not work in the tournament!
    check_submission(TEAM_NAME)

    my_network = load_network(TEAM_NAME)

    def choose_move_no_network(state) -> int:
        """The arguments in play_pong_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, neural_network=my_network)

    # Removing the barriers and making the game half-sized will make it easier to train!
    include_barriers = False
    half_game_size = True

    # The code below plays a single game against your bot.
    # You play as the pink ship
    play_shooter(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=1,
        render=True,
        include_barriers=include_barriers,
        half_game_size=half_game_size,
    )
