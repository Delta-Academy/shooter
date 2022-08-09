from typing import Any, Dict

import numpy as np
from torch import nn
from tqdm import tqdm

from check_submission import check_submission
from game_mechanics import (
    ShooterEnv,
    choose_move_randomly,
    load_network,
    play_shooter,
    save_network,
)

TEAM_NAME = "Team Jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> nn.Module:
    """
    TODO: Write this function to train your algorithm.

    Returns:
    """
    return nn.Linear(1, 1)


def choose_move(state: Any, neural_network: nn.Module) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state:

    Returns:
    """
    return choose_move_randomly(state)


def n_games() -> None:
    n = 1000
    n_actions = []
    rewards = []
    for _ in tqdm(range(n)):
        game = ShooterEnv(choose_move_randomly, render=False)
        state, reward, done, info = game.reset()
        while not done:
            action = choose_move_randomly(state)
            state, reward, done, info = game.step(action)

        rewards.append(reward)
        n_actions.append(game.n_actions)
    assert np.mean(n_actions) > 250, "Maybe too few actions, check"
    print(f"Average reward = {np.mean(rewards)}")


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    file = train()
    save_network(file, TEAM_NAME)

    # check_submission(
    #     TEAM_NAME
    # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    my_value_fn = load_network(TEAM_NAME)

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_value_fn(state: Any) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_value_fn)

    play_shooter(
        your_choose_move=choose_move_no_value_fn,
        opponent_choose_move=choose_move_randomly,
        game_speed_multiplier=1,
        render=True,
    )
    # n_games()
