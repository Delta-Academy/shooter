import time
from typing import Any

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
    load_network,
    play_shooter,
    save_network,
)

TEAM_NAME = "Team Jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


class CustomCallback(BaseCallback):

    """A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.rewards = []
        self.count = 0
        self.fixed_env = ShooterEnv(choose_move_randomly, render=False)
        self.n_steps = 0
        self.track_performance = []
        self.loss = []
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.model._total_timesteps)

    def _on_rollout_start(self) -> None:
        """Called every n_steps."""

        # self.count += 1
        # n_test_games = 10
        # n_wins = 0

        # for _ in range(n_test_games):
        #     obs = self.fixed_env.reset()
        #     done = False
        #     while not done:
        #         action, _states = self.model.predict(obs, deterministic=True)
        #         obs, reward, done, info = self.fixed_env.step(action)

        #     if reward == 1:
        #         n_wins += 1

        # self.track_performance.append(n_wins / n_test_games)

    def _on_step(self) -> None:
        """Called every step()"""
        # self.rewards.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
        # self.n_steps += 1
        self.pbar.update(1)
        self.pbar.refresh()
        # self.loss.append(self.model.policy_gradient_loss)
        self.rewards.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))


def train() -> nn.Module:
    """
    TODO: Write this function to train your algorithm.

    Returns:
    """
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


def train_new() -> nn.Module:
    def model_predict_wrapper(obs):
        return model.predict(obs, deterministic=True)[0]

    env = ShooterEnv(choose_move_randomly, render=False)
    # env = ShooterEnv(model_predict_wrapper, render=False)

    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4)

    callback = CustomCallback()
    model.learn(total_timesteps=100_000, callback=callback)
    plt.plot(callback.rewards)
    # model.learn(total_timesteps=10_000, callback=callback)

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

    print(f"Trained bot performance: {n_wins / n_test_games}")
    model.save("Meaty_model")
    1 / 0


def test():
    def model_predict_wrapper(obs):
        return model.predict(obs, deterministic=True)[0]

    def choose_move_test(state):
        print(state[2])
        # Rotate clockwise
        return 0

        # Move Forward
        # return 2

    model = PPO.load("Meaty_model")
    # env = ShooterEnv(choose_move_randomly, render=True)
    # env = ShooterEnv(model_predict_wrapper, render=True)
    env = ShooterEnv(model_predict_wrapper, render=True)
    obs = env.reset()
    done = False
    time.sleep(3)
    while not done:
        # action, _states = model.predict(obs, deterministic=True)
        # action = choose_move_test(obs)
        action = np.random.randint(3)

        obs, reward, done, info = env.step(action)
        time.sleep(0.2)


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
    # save_network(file, TEAM_NAME)

    # # check_submission(
    # #     TEAM_NAME
    # # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # my_value_fn = load_network(TEAM_NAME)
    # test()

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
