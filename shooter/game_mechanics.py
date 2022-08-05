import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pygame
import torch
from torch import nn

from models import Bullet, DummyScreen, Spaceship
from utils import get_random_position, load_sprite, print_text

GAME_SIZE = (800, 600)
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
    # if verbose:
    #     time.sleep(1 / game_speed_multiplier)

    while not done:
        action = your_choose_move(state)
        state, reward, done, info = game.step(action)
        print(done)
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


class ShooterEnv:
    def __init__(self, opponent_choose_move: Callable, render: bool):

        self.render = render
        self.opponent_choose_move = opponent_choose_move
        if self.render:
            self.init_graphics()
        else:
            self.screen = DummyScreen(GAME_SIZE)

        self.reset()

    def reset(self) -> Tuple[np.ndarray, float, bool, Dict]:
        self.message = ""
        self.asteroids: List[Bullet] = []
        self.player1 = Spaceship(
            (GAME_SIZE[0] // 4, GAME_SIZE[1] // 2), player=1, graphical=self.render
        )
        self.player2 = Spaceship(
            (GAME_SIZE[0] // (4 / 3), GAME_SIZE[1] // 2),
            player=2,
            graphical=self.render,
        )
        self.done = False
        self.n_actions = 0
        return self.observation, 0, False, {}

    def init_graphics(self):
        pygame.init()
        pygame.display.set_caption("Space Rocks")
        self.screen = pygame.display.set_mode(GAME_SIZE)
        self.background = load_sprite("space", False)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 64)

    def _step(self, action: int, player: Spaceship) -> int:
        """Takes a single step and returns the reward."""

        assert isinstance(action, int) and 0 <= action <= 4, "Action should be an integer 0-4"
        self._take_action(action, player)
        winner = self._process_game_logic()
        if winner is not None:
            self.done = True
            return int(winner == player)
        return 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        reward = self._step(action, self.player1)

        reward += (
            self._step(self.opponent_choose_move(self.observation), self.player2)
            * -1  # TODO: Flip the observation
        )

        return self.observation, reward, self.done, {}

    @property
    def observation(self) -> np.ndarray:
        observation = np.zeros(6 * 3)
        for idx, object in enumerate(
            [self.player1, self.player2, *self.player1.bullets, *self.player1.bullets]
        ):
            fill = np.array([object.position[0], object.position[1], object.angle])
            observation[idx * 3 : idx * 3 + 3] = fill
        return observation

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

    def _process_game_logic(self) -> Optional[Spaceship]:

        for game_object in self._get_game_objects():
            game_object.move(self.screen)

        for bullet in self.player1.bullets:
            assert bullet.radius == 5
            if bullet.collides_with(self.player2):
                self.message = "Player 1 wins!"
                winner = self.player1
                return winner
            # Does this give player1 a slight advantage cos the
            # collision is checked first?
        for bullet in self.player2.bullets:
            assert bullet.radius == 5
            if bullet.collides_with(self.player1):
                self.message = "Player 2 wins!"
                winner = self.player2
                return winner

        return None

        # I think this removes collided bullets but not sure
        # for bullet in self.bullets[:]:
        #     if not self.screen.get_rect().collidepoint(bullet.position):
        #         self.bullets.remove(bullet)

    def _draw(self):
        self.screen.blit(self.background, (0, 0))

        for game_object in self._get_game_objects():
            game_object.draw(self.screen)

        if self.message:
            print_text(self.screen, self.message, self.font)

        pygame.display.flip()
        self.clock.tick(60)

    def _get_game_objects(self):

        game_objects = []
        if self.player1:
            game_objects.extend([self.player1, *self.player1.bullets])

        if self.player2:
            game_objects.extend([self.player2, *self.player2.bullets])

        # if self.player2:
        #     game_objects.append(self.player2)

        return game_objects

    # def _step(self, move: Optional[Tuple[int, int]], verbose: bool = False) -> int:
    #     """Takes 1 turn, internal to this class.

    #     Do not call
    #     """
    #     assert not self.done, "Game is over, call .reset() to start a new game"

    #     if move is None:
    #         assert not has_legal_move(
    #             self._board, self._player
    #         ), f"Your move is None, but you must make a move when a legal move is available!"
    #         if verbose:
    #             print(f"Player {self._player} has no legal move, switching player")
    #         self.switch_player()
    #         return 0

    #     assert is_legal_move(self._board, move, self._player), f"Move {move} is not valid!"

    #     self.running_tile_count += 1
    #     self._board = _make_move(self._board, move, self._player)

    #     # Check for game completion
    #     tile_difference = self.tile_count[self._player] - self.tile_count[self._player * -1]
    #     self.done = self.game_over
    #     self.winner = (
    #         None
    #         if self.tile_count[1] == self.tile_count[-1] or not self.done
    #         # mypy sad, probably bug: github.com/python/mypy/issues/9765
    #         else max(self.tile_count, key=self.tile_count.get)  # type: ignore
    #     )
    #     won = self.done and tile_difference > 0

    #     # Currently just if won, many alternatives
    #     reward = 1 if won else 0

    #     if verbose:
    #         print(f"Player {self._player} places counter at row {move[0]}, column {move[1]}")
    #         print(self)
    #         if self.done:
    #             if won:
    #                 print(f"Player {self._player} has won!\n")
    #             elif self.running_tile_count == self.board_dim**2 and tile_difference == 0:
    #                 print("Board full. It's a tie!")
    #             else:
    #                 print(f"Player {self._player * -1} has won!\n")

    #     self.switch_player()
    #     return reward

    # def step(
    #     self, move: Optional[Tuple[int, int]], verbose: bool = False
    # ) -> Tuple[np.ndarray, int, bool, Dict[str, int]]:
    #     """Called by user - takes 2 turns, yours and your opponent's"""

    #     reward = self._step(move, verbose)

    #     if not self.done:
    #         # Negative sign is because both players should see themselves as player 1
    #         opponent_action = self._opponent_choose_move(-self._board)
    #         opponent_reward = self._step(opponent_action, verbose)  # Can be None, fix
    #         # Negative sign is because the opponent's victory is your loss
    #         reward -= opponent_reward

    #     if self.done:
    #         if np.sum(self._board == 1) > np.sum(self._board == -1):
    #             reward = 1
    #         elif np.sum(self._board == 1) < np.sum(self._board == -1):
    #             reward = -1
    #         else:
    #             reward = 0
    #     else:
    #         reward = 0

    #     return self._board, reward, self.done, self.info
