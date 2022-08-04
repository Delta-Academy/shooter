import pygame

from models import Asteroid, Spaceship
from utils import get_random_position, load_sprite, print_text


class SpaceRocks:
    MIN_ASTEROID_DISTANCE = 250

    def __init__(self):
        self._init_pygame()
        self.screen = pygame.display.set_mode((800, 600))
        self.background = load_sprite("space", False)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 64)
        self.message = ""

        self.asteroids = []
        self.bullets = []
        self.player1 = Spaceship((400, 300), self.bullets.append)
        self.player2 = Spaceship((400, 300), self.bullets.append)

        for _ in range(1):
            while True:
                position = get_random_position(self.screen)
                if position.distance_to(self.player1.position) > self.MIN_ASTEROID_DISTANCE:
                    break

            self.asteroids.append(Asteroid(position, self.asteroids.append))

    def main_loop(self):
        while True:
            self._handle_input()
            self._process_game_logic()
            self._draw()

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Space Rocks")

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
            else:
                self.player1.velocity *= 0

    def _process_game_logic(self):
        for game_object in self._get_game_objects():
            game_object.move(self.screen)

        if self.player1:
            for asteroid in self.asteroids:
                if asteroid.collides_with(self.player1):
                    self.player1 = None
                    self.message = "You lost!"
                    break

        for bullet in self.bullets[:]:
            for asteroid in self.asteroids[:]:
                if asteroid.collides_with(bullet):
                    self.asteroids.remove(asteroid)
                    self.bullets.remove(bullet)
                    asteroid.split()
                    break

        for bullet in self.bullets[:]:
            if not self.screen.get_rect().collidepoint(bullet.position):
                self.bullets.remove(bullet)

        if not self.asteroids and self.player1:
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
