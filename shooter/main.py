import time
from typing import Any, Dict

import gym
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

TEAM_NAME = "Team Jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> nn.Module:

    env = ShooterEnv(choose_move_randomly, render=False)
    model = PPO("MlpPolicy", env, verbose=2)
    model.learn(total_timesteps=100_000)

    env = ShooterEnv(choose_move_randomly, render=False)

    n_test_games = 100
    n_wins = 0

    for _ in range(n_test_games):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        if reward == 1:
            n_wins += 1

    print(n_wins / n_test_games)

    model.save("Meaty_model")


def test():
    model = PPO.load("Meaty_model")
    env = ShooterEnv(choose_move_randomly, render=True)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(0.4)


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
    # file = train()
    # save_network(file, TEAM_NAME)

    # # check_submission(
    # #     TEAM_NAME
    # # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # my_value_fn = load_network(TEAM_NAME)
    test()

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    # def choose_move_no_value_fn(state: Any) -> int:
    #     """The arguments in play_game() require functions that only take the state as input.

    #     This converts choose_move() to that format.
    #     """
    #     return choose_move(state, my_value_fn)

    # play_shooter(
    #     your_choose_move=choose_move_no_value_fn,
    #     opponent_choose_move=choose_move_randomly,
    #     game_speed_multiplier=1,
    #     render=True,
    # )
    # # n_games()
