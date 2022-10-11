import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import gym
import gym.spaces
import numpy as np
import pygame
import torch
from torch import nn

from models import (
    DOWN,
    GAME_SIZE,
    LEFT,
    RIGHT,
    UP,
    DummyScreen,
    GameObject,
    Spaceship,
    get_barriers,
)
from shooter_utils import load_sprite, print_text

SPAWN_POINTS = [
    (int(GAME_SIZE[0] * 0.1), GAME_SIZE[1] // 2),
    (int(GAME_SIZE[0] * 0.9), GAME_SIZE[1] // 2),
    (GAME_SIZE[0] // 2, int(GAME_SIZE[1] * 0.1)),
    (GAME_SIZE[0] // 2, int(GAME_SIZE[1] * 0.9)),
]

SPAWN_ORIENTATIONS = [RIGHT, LEFT, DOWN, UP]


HERE = Path(__file__).parent.resolve()

BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)


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
        render: whether to render the game graphically
        game_speed_multiplier: multiplies the speed of the game. High == fast
                               (only has an effect when verbose=True)

    Returns: total_return, which is the sum of return from the game
    """
    total_return = 0.0
    env = ShooterEnv(
        opponent_choose_move, render=render, game_speed_multiplier=game_speed_multiplier
    )

    state, _, done, _ = env.reset()
    while not done:
        action = your_choose_move(state)
        state, reward, done, _ = env.step(action)
        total_return += reward
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
    def __init__(
        self,
        opponent_choose_move: Callable,
        render: bool = False,
        game_speed_multiplier: float = 1,
        include_barriers: bool = True,
    ):

        self._render = render
        self.opponent_choose_move = opponent_choose_move
        self.game_speed_multiplier = game_speed_multiplier
        if self._render:
            self.init_graphics()
        else:
            self.screen = DummyScreen(GAME_SIZE)

        self.num_envs = 1
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(18,))
        self.action_space = gym.spaces.Discrete(4)  # type: ignore
        self.include_barriers = include_barriers
        self.barriers = get_barriers() if include_barriers else []
        self.reset()
        if self._render:
            self._draw()

    def reset(self) -> Tuple[np.ndarray, float, bool, Dict]:
        self.message = ""
        spawn_idx = list(range(len(SPAWN_POINTS)))
        random.shuffle(spawn_idx)

        player1_idx = spawn_idx.pop()
        player2_idx = spawn_idx.pop()

        self.player1 = Spaceship(
            SPAWN_POINTS[player1_idx],
            SPAWN_ORIENTATIONS[player1_idx],
            player=1,
            graphical=self._render,
            include_barriers=self.include_barriers,
        )
        self.player2 = Spaceship(
            SPAWN_POINTS[player2_idx],
            SPAWN_ORIENTATIONS[player2_idx],
            player=2,
            graphical=self._render,
            include_barriers=self.include_barriers,
        )
        self.done = False
        self.n_actions = 0
        # return self.observation_player1, 0.0, False, {}
        return self.observation_player1

    def init_graphics(self) -> None:
        pygame.init()
        pygame.display.set_caption("Space Shooter")
        self.screen = pygame.display.set_mode(GAME_SIZE)
        self.background = load_sprite("space", False)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 64)

    def _step(self, action: int, player: Spaceship) -> None:
        """Takes a single step for one player."""

        assert isinstance(action, (int, np.int64)) and action in range(  # type: ignore
            4
        ), f"Action should be an integer 0-3. Got {action}"
        self._take_action(action, player)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if action is not None:
            self._step(action, self.player1)

        opponent_move = self.opponent_choose_move(state=self.observation_player2)
        if opponent_move is not None:
            self._step(opponent_move, self.player2)

        winners = self._process_game_logic()

        if winners is None or len(winners) > 1:  # Continuing game / Reservoir dogs ending
            reward = 0
        else:

            reward = 1 if winners[0] == self.player1 else -1

        if self._render:
            self._draw()
            time.sleep(0.05 / self.game_speed_multiplier)

        return self.observation_player1, reward, self.done, {}

    @property
    def total_game_bullets(self) -> int:
        return self.player1.NUM_BULLETS * 2

    @property
    def n_observations(self) -> int:
        return (2 + self.total_game_bullets) * 3

    @property
    def observation_player1(self) -> np.ndarray:

        observation_player1 = np.zeros(self.n_observations)
        for idx, object in enumerate(
            [self.player1, self.player2, *self.player1.bullets, *self.player2.bullets]
        ):
            observation_player1[idx * 3 : (idx + 1) * 3] = np.array(
                [
                    self.normalise(object.position[0], GAME_SIZE[0]),
                    self.normalise(object.position[1], GAME_SIZE[1]),
                    self.normalise(object.angle % 360, 360),
                ]
            )

        return observation_player1

    @staticmethod
    def normalise(x: float, max_x: float) -> float:
        """Normalise x to be between -1 and 1."""
        return 2 * (x / max_x) - 1

    @property
    def observation_player2(self) -> np.ndarray:
        observation_player2 = np.zeros(self.n_observations)

        for idx, object in enumerate(
            [self.player2, self.player1, *self.player2.bullets, *self.player1.bullets]
        ):
            observation_player2[idx * 3 : (idx + 1) * 3] = np.array(
                [
                    self.normalise(object.position[0], GAME_SIZE[0]),
                    self.normalise(object.position[1], GAME_SIZE[1]),
                    self.normalise(object.angle % 360, 360),
                ]
            )

        return observation_player2

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
                self.message = "Player 1 wins!"
                self.player2.dead = True
                if self._render:
                    self._draw()

                winners.append(self.player1)

        for bullet in self.player2.bullets:
            if bullet.collides_with(self.player1):
                self.done = True
                self.message += "Player 2 wins!"
                winners.append(self.player2)
                self.player1.dead = True
                if self._render:
                    self._draw()

        for bullet in self.player1.bullets:
            if (
                not self.screen.get_rect().collidepoint((bullet.position[0], bullet.position[1]))
                or bullet.hit_barrier
            ):
                self.player1.bullets.remove(bullet)

        for bullet in self.player2.bullets:
            if (
                not self.screen.get_rect().collidepoint(bullet.position[0], bullet.position[1])
                or bullet.hit_barrier
            ):
                self.player2.bullets.remove(bullet)

        return winners or None

    def _draw(self) -> None:
        assert not isinstance(self.screen, DummyScreen), "Don't call _draw() with a dummy screen"
        self.screen.blit(self.background, (0, 0))
        self.screen.fill((BLACK_COLOR))

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

        game_objects.extend(self.barriers)
        return game_objects


def human_player(*arg, **kwargs) -> Optional[int]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            quit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            return 3
    is_key_pressed = pygame.key.get_pressed()
    if is_key_pressed[pygame.K_RIGHT]:
        return 0
    elif is_key_pressed[pygame.K_LEFT]:
        return 1
    return 2 if is_key_pressed[pygame.K_UP] else None
