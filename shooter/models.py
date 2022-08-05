from abc import ABC, abstractmethod
from collections import abc
from typing import Any, List, Tuple, Union

import pygame
from pygame.math import Vector2
from pygame.surface import Surface
from pygame.transform import rotozoom

from utils import get_random_velocity, load_sound, load_sprite, wrap_position

UP = Vector2(0, -1)


class DummyScreen:
    def __init__(self, size: Tuple):
        self.size = size

    def get_size(self):
        return self.size[0], self.size[1]


class DummySprite(ABC):
    @abstractmethod
    def get_width(self) -> int:
        pass


class DummyShip(DummySprite):
    def get_width(self) -> int:
        return 40


class DummyBullet(DummySprite):
    def get_width(self) -> int:
        return 10


class DummySound:
    def play(self) -> None:
        pass


class GameObject:
    def __init__(
        self, starting_position: Tuple[int, int], sprite: Union[Surface, DummySprite], velocity: int
    ) -> None:
        self.set_position(starting_position)
        self.sprite = sprite
        self.radius = sprite.get_width() / 2
        self.velocity = Vector2(velocity)
        self.face_up()

    def set_position(self, position: Tuple[int, int]) -> None:
        self.position = Vector2(position)

    def face_up(self) -> None:
        self.direction = Vector2(UP)

    @property
    def angle(self) -> int:
        return int(self.direction.angle_to(UP))

    def draw(self, surface: pygame.Surface) -> None:
        blit_position = self.position - Vector2(self.radius)
        surface.blit(self.sprite, blit_position)

    def move(self, surface: pygame.Surface) -> None:
        self.position = wrap_position(self.position + self.velocity, surface)

    def collides_with(self, other_obj: "GameObject") -> bool:
        distance = self.position.distance_to(other_obj.position)
        return distance < self.radius + other_obj.radius


class Spaceship(GameObject):
    ANGLE_TURN = 45
    ACCELERATION = 0.1
    BULLET_SPEED = 60
    NUM_BULLETS = 2  # Limit the number on the screen at one time

    def __init__(self, starting_position, player: int, graphical=True) -> None:

        # Make a copy of the original UP vector
        self.starting_position = starting_position
        self.graphical = graphical
        self.player = player

        if self.graphical:
            super().__init__(starting_position, load_sprite("spaceship"), Vector2(0))
            self.laser_sound = load_sound("laser")
        else:
            super().__init__(starting_position, DummyShip(), Vector2(0))
            self.laser_sound = DummySound()

        self.reset()

    def reset(self) -> None:
        self.set_position(self.starting_position)
        self.face_up()

        self.bullets: List[Bullet] = []

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Spaceship):
            raise NotImplemented
        return self.player == other.player

    def rotate(self, clockwise=True):
        sign = 1 if clockwise else -1
        angle = self.ANGLE_TURN * sign
        self.direction.rotate_ip(angle)

    def accelerate(self):
        self.velocity += self.direction * self.ACCELERATION

    def move_forward(self):
        distance = self.radius
        self.position += self.direction * distance

    def draw(self, surface):
        angle = self.direction.angle_to(UP)
        rotated_surface = rotozoom(self.sprite, angle, 1.0)
        rotated_surface_size = Vector2(rotated_surface.get_size())
        blit_position = self.position - rotated_surface_size * 0.5
        surface.blit(rotated_surface, blit_position)

    def shoot(self):
        # Limit number of bullets
        if len(self.bullets) == self.NUM_BULLETS:
            return
        bullet_velocity = self.direction * self.BULLET_SPEED + self.velocity
        bullet = Bullet(self.position, bullet_velocity, self.graphical)
        self.bullets.append(bullet)
        self.laser_sound.play()


# class Asteroid(GameObject):
#     def __init__(self, position, create_asteroid_callback, size=3):
#         self.create_asteroid_callback = create_asteroid_callback
#         self.size = size

#         size_to_scale = {3: 1.0, 2: 0.5, 1: 0.25}
#         scale = size_to_scale[size]
#         sprite = rotozoom(load_sprite("asteroid"), 0, scale)

#         super().__init__(position, sprite, get_random_velocity(1, 3))

#     def split(self):
#         if self.size > 1:
#             for _ in range(2):
#                 asteroid = Asteroid(self.position, self.create_asteroid_callback, self.size - 1)
#                 self.create_asteroid_callback(asteroid)


class Bullet(GameObject):
    def __init__(self, position, velocity, graphical):
        if graphical:
            super().__init__(position, load_sprite("bullet"), velocity)
        else:
            super().__init__(position, DummyBullet(), velocity)

    def move(self, surface):
        self.position = self.position + self.velocity
