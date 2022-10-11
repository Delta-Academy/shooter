import copy
import cProfile
import time
from typing import Any, Dict

import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn
from tqdm import tqdm

from check_submission import check_submission
from game_mechanics import (
    ShooterEnv,
    choose_move_randomly,
    human_player,
    load_network,
    play_shooter,
    save_network,
)

## Delete me before merging

TEAM_NAME = "Team Jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

INCLUDE_BARRIERS = True


class ChooseMoveCheckpoint:
    def __init__(self, model):
        self.model = copy.deepcopy(model)

    def choose_move(self, state):
        action, _ = self.model.predict(state, deterministic=False)
        return action


def train() -> nn.Module:

    ####### train ##########
    # model = PPO.load("/Users/jamesrowland/Code/shooter/checkpoints/Meaty_model2")

    # env = ShooterEnv(opponent_choose_move, render=False, include_barriers=INCLUDE_BARRIERS)
    # env = ShooterEnv(choose_move_randomly, render=False, include_barriers=INCLUDE_BARRIERS)
    # model = PPO("MlpPolicy", env, verbose=2, ent_coef=0.05)

    # model = PPO("MlpPolicy", env, verbose=2, ent_coef=0.05)

    n = 4
    model = PPO.load(f"/Users/jamesrowland/Code/shooter/checkpoints/Meaty_model{n}.zip")

    opponent_choose_move = ChooseMoveCheckpoint(model).choose_move
    # opponent_choose_move = ChooseMoveCheckpoint(model).choose_move
    env = ShooterEnv(opponent_choose_move, render=False, include_barriers=INCLUDE_BARRIERS)

    model.set_env(env)

    model.learn(total_timesteps=300_000)
    model.save(f"/Users/jamesrowland/Code/shooter/checkpoints/Meaty_model{n+1}.zip")

    ####### test ##########
    test_env = ShooterEnv(choose_move_randomly, render=False, include_barriers=INCLUDE_BARRIERS)

    n_test_games = 100
    n_wins = 0

    for _ in range(n_test_games):
        obs = test_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = test_env.step(action)

        print(reward)
        if reward == 1:

            n_wins += 1

    print(f"fraction games won = {n_wins / n_test_games}")

    return n_wins / n_test_games


def test_graphics():

    model = PPO.load("/Users/jamesrowland/Code/shooter/checkpoints/Meaty_model4.zip")
    old_model = PPO.load("/Users/jamesrowland/Code/shooter/checkpoints/Meaty_model2.zip")

    def model_predict_wrapper(state: np.ndarray) -> int:

        return model.predict(state, deterministic=False)[0]

    n_games = 10
    env = ShooterEnv(
        model_predict_wrapper,
        render=True,
        game_speed_multiplier=100,
        include_barriers=INCLUDE_BARRIERS,
    )
    for game in range(n_games):
        # obs, _, done, _ = env.reset()
        obs = env.reset()
        done = False
        time.sleep(100)
        while not done:
            # action, _ = model.predict(obs, deterministic=False)
            # action = human_player(obs)
            action = choose_move_randomly(obs)
            # action = choose_move_randomly(obs)
            obs, reward, done, info = env.step(action)
            time.sleep(0.1)

        time.sleep(2)


def choose_move(state: Any, neural_network: nn.Module) -> int:
    """Called during competitive play.

    It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.
    Args:
        state:
    Returns:
    """

    model = PPO.load("jimmy_baselines_model")
    return model.predict(state)[0]
    # done = False

    # n_games = 3
    # env = ShooterEnv(human_player, render=True)
    # for game in range(n_games):
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         print(obs)
    #         # action, _states = model.predict(obs, deterministic=True)
    #         action = 3
    #         obs, reward, done, info = env.step(action)
    #         time.sleep(0.1)
    #     time.sleep(2)

    # return choose_move_randomly(state)


def n_games() -> None:
    n = 250
    n_actions = []
    rewards = []
    steps_count = []
    model = PPO.load("/Users/jamesrowland/Code/shooter/checkpoints/Meaty_model4.zip")
    for _ in tqdm(range(n)):
        n_steps = 0
        game = ShooterEnv(choose_move_randomly, render=False)

        obs = game.reset()
        # state, reward, done, info = game.reset()
        done = False
        while not done:
            n_steps += 1
            # action = choose_move_randomly(state)
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = game.step(action)

        rewards.append(reward)
        n_actions.append(game.n_actions)
        steps_count.append(n_steps)
    # assert np.mean(n_actions) > 250, "Maybe too few actions, check"
    print(f"Average reward = {np.mean(rewards)}")
    print(f"Average n steps = {np.mean(steps_count)}")


if __name__ == "__main__":

    # ## Example workflow, feel free to edit this! ###
    # np.set_printoptions(suppress=True)
    # train()

    # do_a_train = False
    # if do_a_train:

    #     t1 = time.time()
    #     performance_verbose = train()

    #     t2 = time.time()
    #     print(f"time: {t2 - t1}")

    # save_network(file, TEAM_NAME)

    # # check_submission(
    # #     TEAM_NAME
    # # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # my_value_fn = load_network(TEAM_NAME)
    # test_graphics()

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
    # train()
    test_graphics()
    # n_games()
    # cProfile.run("n_games()", "profile.prof")
