import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import gym
import numpy as np
import pygame
import torch
from torch import nn

from models import BARRIERS, Barrier, Bullet, DummyScreen, GameObject, Spaceship
from utils import get_random_position, load_sprite, print_text

GAME_SIZE = (600, 450)
# GAME_SIZE = (400, 300)
# GAME_SIZE = (200, 150)


HERE = Path(__file__).parent.resolve()


def play_shooter(
    your_choose_move: Callable[[np.ndarray], int],
    opponent_choose_move: Callable[[np.ndarray], int],
    game_speed_multiplier: float = 1,
    render: bool = False,
) -> float:
    """Play a game where moves are chosen by `your_choose_move()` and `opponent_choose_move()`.

    . You can render the game by setting `render=True`.

    Args:
        your_choose_move: function that chooses move (takes state as input)
        opponent_choose_move: function that picks your opponent's next move
        verbose: whether to print board states to console. Useful for debugging
        game_speed_multiplier: multiplies the speed of the game. High == fast
                               (only has an effect when verbose=True)

    Returns: total_return, which is the sum of return from the game
    """
    total_return = 0.0
    game = ShooterEnv(opponent_choose_move, render=render)
    state, reward, done, info = game.reset()
    while not done:
        action = your_choose_move(state)
        state, reward, done, info = game.step(action)
        total_return += reward
        if render:
            time.sleep(0.01)
    return total_return


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


def choose_move_randomly(state: np.ndarray) -> int:
    return np.random.randint(4)


class ShooterEnv(gym.Env):
    def __init__(self, opponent_choose_move: Callable, render: bool):

        self.render = render
        self.opponent_choose_move = opponent_choose_move
        if self.render:
            self.init_graphics()
        else:
            self.screen = DummyScreen(GAME_SIZE)

        self.reset()
        self.num_envs = 1
        self.action_space = gym.spaces.Discrete(4)
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.n_observations,))
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(self.n_observations,))

        self.metadata = ""
        if self.render:
            self._draw()

    def reset(self) -> Tuple[np.ndarray, float, bool, Dict]:
        self.message = ""
        player1_starting_pos = (GAME_SIZE[0] // 4, GAME_SIZE[1] // 2)
        player2_starting_pos = (int(GAME_SIZE[0] // (4 / 3)), GAME_SIZE[1] // 2)

        # player1_starting_pos = (np.random.randint(GAME_SIZE[0]), np.random.randint(GAME_SIZE[1]))
        # player2_starting_pos = (np.random.randint(GAME_SIZE[0]), np.random.randint(GAME_SIZE[1]))
        self.player1 = Spaceship(player1_starting_pos, player=1, graphical=self.render)
        self.player2 = Spaceship(player2_starting_pos, player=2, graphical=self.render)
        self.done = False
        self.n_actions = 0
        # Players should see themselves as in the same place after reset
        # assert np.all(self.observation_player2 == self.observation_player1)
        return self.observation_player1

    def init_graphics(self) -> None:
        pygame.init()
        pygame.display.set_caption("Space Rocks")
        self.screen = pygame.display.set_mode(GAME_SIZE)
        self.background = load_sprite("space", False)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 64)

    def _step(self, action: int, player: Spaceship) -> None:
        """Takes a single step for one player."""

        assert (
            isinstance(action, (int, np.int64)) and 0 <= action <= 3
        ), "Action should be an integer 0-3"
        # if action in {0, 1} and player == self.player2:
        #     # Turning is in the reversed direction
        #     action = int(not action)

        self._take_action(action, player)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self._step(action, self.player1)

        self._step(self.opponent_choose_move(self.observation_player2), self.player2)

        winners = self._process_game_logic()

        if winners is None or len(winners) > 1:  # Continuing game / Reservoir dogs ending
            reward = 0
        else:
            reward = 1 if winners[0] == self.player1 else -1

        if self.render:
            self._draw()

        # assert np.all(self.observation_player2 == self.observation_player1)
        return self.observation_player1, reward, self.done, {}

    @property
    def total_game_bullets(self) -> int:
        return self.player1.NUM_BULLETS * 2

    @property
    def n_observations(self):
        return (2 + self.total_game_bullets) * 3

    @property
    def observation_player1(self) -> np.ndarray:

        observation_player1 = np.zeros(self.n_observations)
        for idx, object in enumerate(
            [self.player1, self.player2, *self.player1.bullets, *self.player2.bullets]
        ):
            observation_player1[idx * 3 : (idx + 1) * 3] = np.array(
                [
                    object.position[0] / GAME_SIZE[0],  # Divide by the max value
                    object.position[1] / GAME_SIZE[1],
                    object.angle / 360,
                ]
            )

        return observation_player1

    @property
    def observation_player2(self) -> np.ndarray:
        observation_player2 = np.zeros(self.n_observations)

        for idx, object in enumerate(
            [self.player2, self.player1, *self.player2.bullets, *self.player1.bullets]
        ):
            observation_player2[idx * 3 : (idx + 1) * 3] = np.array(
                [
                    object.position[0] / GAME_SIZE[0],  # Divide by the max value
                    object.position[1] / GAME_SIZE[1],
                    object.angle / 360,
                ]
            )

        return observation_player2

    def main_loop(self) -> None:
        if self.render:
            while not self.done:
                time.sleep(0.1)
                # if play with keyboard  TODO: Implement flag
                # self._handle_input()
                assert self.player1.radius == 20
                assert self.player2.radius == 20
                self._take_action(np.random.randint(4), self.player1)
                self._take_action(np.random.randint(4), self.player2)
                self._process_game_logic()
                self._draw()

        else:
            while not self.done:
                assert self.player1.radius == 20
                assert self.player2.radius == 20
                self._take_action(np.random.randint(4), self.player1)
                self._take_action(np.random.randint(4), self.player2)
                self._process_game_logic()

    def _handle_input(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                quit()
            elif self.player1 and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.player1.shoot()

        is_key_pressed = pygame.key.get_pressed()

        if self.player1:
            if is_key_pressed[pygame.K_RIGHT]:
                self.player1.rotate(clockwise=True)
            elif is_key_pressed[pygame.K_LEFT]:
                self.player1.rotate(clockwise=False)
            if is_key_pressed[pygame.K_UP]:
                self.player1.accelerate()

    def _take_action(self, action: int, player: Spaceship) -> None:
        self.n_actions += 1
        player.velocity *= 0
        if action == 0:
            player.rotate(clockwise=True)
        elif action == 1:
            player.rotate(clockwise=False)
        elif action == 2:
            player.move_forward()
        elif action == 3:
            player.shoot()

    def _process_game_logic(self) -> Optional[List[Spaceship]]:

        for game_object in self._get_game_objects():
            game_object.move(self.screen)

        # Can get both players winning reservoir dogs style
        winners = []

        for bullet in self.player1.bullets:
            # Remove
            assert bullet.radius == 5
            if bullet.collides_with(self.player2):
                self.done = True
                self.message += "Player 1 wins!"
                if self.render:
                    self.player2.dead = True
                    self._draw()

                winners.append(self.player1)

        for bullet in self.player2.bullets:
            assert bullet.radius == 5
            if bullet.collides_with(self.player1):
                self.done = True
                self.message += "Player 2 wins!"
                winners.append(self.player2)
                if self.render:
                    self.player1.dead = True
                    self._draw()

        # Remove
        assert len(self.player1.bullets) <= 2
        assert len(self.player2.bullets) <= 2
        for bullet in self.player1.bullets:
            # Remove
            assert self.screen.get_rect() == pygame.Rect(0, 0, GAME_SIZE[0], GAME_SIZE[1])
            if not self.screen.get_rect().collidepoint(bullet.position) or bullet.hit_barrier:
                self.player1.bullets.remove(bullet)

        for bullet in self.player2.bullets:
            if not self.screen.get_rect().collidepoint(bullet.position) or bullet.hit_barrier:
                self.player2.bullets.remove(bullet)

        if winners:
            return winners
        return None

    def _draw(self) -> None:
        assert not isinstance(self.screen, DummyScreen), "Don't call _draw() with a dummy screen"
        self.screen.blit(self.background, (0, 0))

        for game_object in self._get_game_objects():
            game_object.draw(self.screen)

        if self.message:
            print_text(self.screen, self.message, self.font)

        pygame.display.flip()
        self.clock.tick(60)

    def _get_game_objects(self) -> List[GameObject]:

        game_objects = []
        if not self.player1.dead:
            game_objects.extend([self.player1, *self.player1.bullets])

        if not self.player2.dead:
            game_objects.extend([self.player2, *self.player2.bullets])

        game_objects.extend(BARRIERS)
        return game_objects
