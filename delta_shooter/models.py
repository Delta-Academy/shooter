from abc import ABC, abstractmethod
from typing import Any, List, Literal, Tuple, Union

import numpy as np
import pygame
from pygame.math import Vector2
from pygame.surface import Surface
from pygame.transform import rotozoom

from shooter_utils import edge_barriers, load_sound, load_sprite

UP = Vector2(0, -1)
DOWN = Vector2(0, 1)
RIGHT = Vector2(1, 0)
LEFT = Vector2(-1, 0)


# GAME_SIZE = (300, 225)
# GAME_SIZE = (600, 450)
GAME_SIZE = (500, 400)


BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)

# Types of coordinates used throughout
Coord = Union[Vector2, Tuple[int, int]]


class DummyScreen:
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        self.rect = pygame.Rect(0, 0, size[0], size[1])

    def get_size(self) -> Tuple[int, int]:
        return self.size[0], self.size[1]

    def get_rect(self) -> pygame.Rect:
        return self.rect


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


Orientation = Literal["horizontal", "vertical"]


class GameObject:
    def __init__(
        self,
        starting_position: Coord,
        sprite: Union[Surface, DummySprite],
        velocity: Union[int, Vector2],
    ) -> None:
        self.set_position(starting_position)
        self.sprite = sprite
        self.radius = int(sprite.get_width() / 2)
        self.velocity = Vector2(velocity)
        self.face_up()

    def set_position(self, position: Coord) -> None:
        self.position = Vector2(position)

    def set_orientation(self, orientation: Vector2) -> None:
        self.direction = Vector2(orientation)

    def face_up(self) -> None:
        self.set_orientation(UP)

    @property
    def angle(self) -> int:
        return round(self.direction.angle_to(UP))

    def draw(self, surface: pygame.surface.Surface) -> None:
        assert isinstance(self.sprite, pygame.Surface)
        blit_position = self.position - Vector2(self.radius)
        surface.blit(self.sprite, blit_position)

    def move(self, surface: Union[pygame.surface.Surface, DummyScreen]) -> None:
        self.position = edge_barriers(self.position + self.velocity, self.radius, surface)

    def collides_with(self, other_obj: "GameObject") -> bool:
        """Fudge factor stops bullets skipping over objects."""

        fudge_factor = 1.5
        distance = self.position.distance_to(other_obj.position)
        return distance < (self.radius * fudge_factor) + (other_obj.radius * fudge_factor)


class Spaceship(GameObject):
    ANGLE_TURN = 15
    ACCELERATION = 0.1
    BULLET_SPEED = 60
    SHOOTING_JITTER = 2.5  # Add randomness to shot direction
    NUM_BULLETS = 2  # Limit the number on the screen at one time

    def __init__(
        self,
        starting_position: Tuple[int, int],
        starting_orientation: Vector2,
        player: int,
        graphical: bool = True,
        include_barriers: bool = True,
    ) -> None:

        self.starting_position = starting_position
        self.starting_orientation = starting_orientation
        self.graphical = graphical
        self.player = player
        self.dead = False
        self.name = "spaceship"

        if self.graphical:
            super().__init__(
                starting_position, load_sprite(f"spaceship_player{player}"), Vector2(0)
            )
            try:
                self.laser_sound = load_sound("laser")
            except pygame.error:
                self.laser_sound = DummySound()
        else:
            super().__init__(starting_position, DummyShip(), Vector2(0))
            self.laser_sound = DummySound()

        self.reset()
        self.include_barriers = include_barriers
        self.barriers = get_barriers() if self.include_barriers else []

    def reset(self) -> None:
        self.set_position(self.starting_position)
        self.set_orientation(self.starting_orientation)

        self.bullets: List[Bullet] = []

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Spaceship):
            raise NotImplemented
        return self.player == other.player

    def rotate(self, clockwise: bool = True) -> None:
        sign = 1 if clockwise else -1
        angle = self.ANGLE_TURN * sign
        self.direction.rotate_ip(angle)

    def accelerate(self) -> None:
        self.velocity += self.direction * self.ACCELERATION

    def move_forward(self) -> None:
        distance = self.radius
        new_position = self.position + self.direction * distance
        for barrier in self.barriers:
            if barrier.hit_barrier(self.position, new_position, self.radius):
                return
        self.position = new_position

    def draw(self, surface: pygame.surface.Surface) -> None:
        angle = self.direction.angle_to(UP)
        assert isinstance(self.sprite, pygame.Surface)
        rotated_surface = rotozoom(self.sprite, angle, 1.0)
        rotated_surface_size = Vector2(rotated_surface.get_size())
        blit_position = self.position - rotated_surface_size * 0.5
        surface.blit(rotated_surface, blit_position)

    def shoot(self) -> None:
        # Limit number of bullets
        if len(self.bullets) == self.NUM_BULLETS:
            return
        bullet_velocity = (
            self.direction * self.BULLET_SPEED
            + self.velocity
            + Vector2(np.random.normal(0, self.SHOOTING_JITTER))
        )
        bullet = Bullet(self.position, bullet_velocity, self.graphical, self.include_barriers)
        self.bullets.append(bullet)
        self.laser_sound.play()


class Bullet(GameObject):
    def __init__(
        self,
        position: Coord,
        velocity: Union[int, Vector2],
        graphical: bool,
        include_barriers: bool = True,
    ):
        self.hit_barrier = False
        if graphical:
            super().__init__(position, load_sprite("bullet"), velocity)
        else:
            super().__init__(position, DummyBullet(), velocity)
        self.name = "bullet"
        self.barriers = get_barriers() if include_barriers else []

    def move(self, surface: Any) -> None:
        new_position = self.position + self.velocity
        for barrier in self.barriers:
            if barrier.hit_barrier(self.position, new_position, self.radius):
                self.set_position((-100, -100))
                self.hit_barrier = True
                return
        self.set_position(new_position)


def ccw(A: Coord, B: Coord, C: Coord) -> bool:
    """Check if points are in a counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A: Coord, B: Coord, C: Coord, D: Coord) -> bool:
    """Return true if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class Barrier:
    WIDTH = 6  # Width of barrier (needs to be even)

    def __init__(self, orientation: Orientation, length: int, center: Tuple[int, int]):
        self.orientation = orientation
        # Prevents mypy isssues with potentially unbound vars
        assert orientation in {"vertical", "horizontal"}, "Invalid Orientation"
        self.length = length
        self.center = center
        self.name = "barrier"

        if orientation == "vertical":
            self.corner1 = (self.center[0] - self.WIDTH // 2, self.center[1] - self.length // 2)
            self.corner2 = (self.center[0] - self.WIDTH // 2, self.center[1] + self.length // 2)
            self.corner3 = (self.center[0] + self.WIDTH // 2, self.center[1] - self.length // 2)
            self.corner4 = (self.center[0] + self.WIDTH // 2, self.center[1] + self.length // 2)

        else:
            self.corner1 = (self.center[0] - self.length // 2, self.center[1] - self.WIDTH // 2)
            self.corner2 = (self.center[0] + self.length // 2, self.center[1] - self.WIDTH // 2)
            self.corner3 = (self.center[0] - self.length // 2, self.center[1] + self.WIDTH // 2)
            self.corner4 = (self.center[0] + self.length // 2, self.center[1] + self.WIDTH // 2)

    def hit_barrier(
        self,
        pos: Coord,
        new_pos: Coord,
        radius: int,
    ) -> bool:
        # Check points around the front half of the object for intersection

        # Check if the new_pos is inside the barrier
        if (
            new_pos[0] > self.corner1[0] - radius
            and new_pos[0] < self.corner4[0] + radius
            and new_pos[1] > self.corner1[1] - radius
            and new_pos[1] < self.corner4[1] + radius
        ):
            return True

        # Check if passing through the barrier
        return bool(
            intersect(self.corner1, self.corner2, pos, new_pos)
            or intersect(self.corner3, self.corner4, pos, new_pos)
        )

    def draw(self, screen: pygame.surface.Surface) -> None:
        pygame.draw.line(screen, WHITE_COLOR, self.corner1, self.corner2, width=self.WIDTH)

    def move(self, screen: pygame.surface.Surface) -> None:
        pass


def get_barriers() -> List[Barrier]:

    barrier_length = int(GAME_SIZE[1] * 0.3)
    return [
        Barrier(
            orientation="vertical",
            center=(int(GAME_SIZE[0] * 0.2), int(GAME_SIZE[1] * 0.5)),
            length=barrier_length,
        ),
        Barrier(
            orientation="vertical",
            center=(int(GAME_SIZE[0] * 0.8), int(GAME_SIZE[1] * 0.5)),
            length=barrier_length,
        ),
        Barrier(
            orientation="horizontal",
            center=(int(GAME_SIZE[0] * 0.5), int(GAME_SIZE[1] * 0.2)),
            length=barrier_length,
        ),
        Barrier(
            orientation="horizontal",
            center=(int(GAME_SIZE[0] * 0.5), int(GAME_SIZE[1] * 0.8)),
            length=barrier_length,
        ),
    ]
