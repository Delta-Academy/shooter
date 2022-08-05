import time

import numpy as np
import pygame

from models import DummyScreen, Spaceship
from utils import get_random_position, load_sprite, print_text

GAME_SIZE = (800, 600)
# GAME_SIZE = (400, 300)
# GAME_SIZE = (200, 150)


class SpaceRocks:
    def __init__(self, graphical: bool):

        self.graphical = graphical
        if self.graphical:
            self.init_graphics()
        else:
            self.screen = DummyScreen(GAME_SIZE)

        self.reset()

    def reset(self):
        self.message = ""
        self.asteroids = []
        self.bullets = []
        self.player1 = Spaceship(
            (GAME_SIZE[0] // 4, GAME_SIZE[1] // 2), self.bullets.append, graphical=self.graphical
        )
        self.player2 = Spaceship(
            (GAME_SIZE[0] // (4 / 3), GAME_SIZE[1] // 2),
            self.bullets.append,
            player=2,
            graphical=self.graphical,
        )
        self.done = False
        self.n_actions = 0

    def init_graphics(self):
        pygame.init()
        pygame.display.set_caption("Space Rocks")
        self.screen = pygame.display.set_mode(GAME_SIZE)
        self.background = load_sprite("space", False)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 64)

    def main_loop(self):
        if self.graphical:
            while True:
                # self._handle_input()
                self._process_action(np.random.randint(4), self.player1)
                self._process_action(np.random.randint(4), self.player2)
                self._process_game_logic()
                self._draw()

        else:
            while not self.done:
                self._process_action(np.random.randint(4), self.player1)
                self._process_action(np.random.randint(4), self.player2)
                self._process_game_logic()

    def _handle_input(self):
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

    def _process_action(self, action: int, player: Spaceship):
        if self.done:
            return
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

    def _process_game_logic(self):

        if self.done:
            return

        for game_object in self._get_game_objects():
            game_object.move(self.screen)

        for bullet in self.bullets[:]:
            if bullet.collides_with(self.player2):
                self.player2 = None
                self.bullets.remove(bullet)
                self.message = "Player 1 wins!"
                self.done = True
                return
            # Does this give player1 a slight advantage cos the
            # collision is checked first?
            elif bullet.collides_with(self.player1):
                self.player1 = None
                self.bullets.remove(bullet)
                self.message = "Player 2 wins!"
                self.done = True
                return

        # Don't know what this does
        # for bullet in self.bullets[:]:
        #     if not self.screen.get_rect().collidepoint(bullet.position):
        #         self.bullets.remove(bullet)

        if not self.player2 and self.player1:
            self.message = "You won!"

    def _draw(self):
        self.screen.blit(self.background, (0, 0))

        for game_object in self._get_game_objects():
            game_object.draw(self.screen)

        if self.message:
            print_text(self.screen, self.message, self.font)

        pygame.display.flip()
        self.clock.tick(60)

    def _get_game_objects(self):
        game_objects = [*self.asteroids, *self.bullets]

        if self.player1:
            game_objects.append(self.player1)

        if self.player2:
            game_objects.append(self.player2)

        return game_objects
