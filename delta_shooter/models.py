from abc import ABC, abstractmethod
from typing import Any, List, Literal, Tuple, Union

import pygame
from pygame.math import Vector2
from pygame.surface import Surface
from pygame.transform import rotozoom

from shooter_utils import edge_barriers, get_random_velocity, load_sound, load_sprite, wrap_position

UP = Vector2(0, -1)
DOWN = Vector2(0, 1)
RIGHT = Vector2(1, 0)
LEFT = Vector2(-1, 0)


GAME_SIZE = (600, 450)


class DummyScreen:
    def __init__(self, size: Tuple):
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
        starting_position: Tuple[int, int],
        sprite: Union[Surface, DummySprite],
        velocity: int,
    ) -> None:
        self.set_position(starting_position)
        self.sprite = sprite
        self.radius = sprite.get_width() / 2
        self.velocity = Vector2(velocity)
        self.face_up()

    def set_position(self, position: Tuple[int, int]) -> None:
        self.position = Vector2(position)

    def set_orientation(self, orientation: Vector2):
        self.direction = Vector2(orientation)

    def face_up(self) -> None:
        self.set_orientation(UP)

    @property
    def angle(self) -> int:
        # TODO: Make consistent with pong?
        return round(self.direction.angle_to(UP))

    def draw(self, surface: pygame.Surface) -> None:
        blit_position = self.position - Vector2(self.radius)
        surface.blit(self.sprite, blit_position)

    def move(self, surface: pygame.Surface) -> None:
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
    NUM_BULLETS = 2  # Limit the number on the screen at one time

    def __init__(
        self,
        starting_position: Tuple[int, int],
        starting_orientation: Vector2,
        player: int,
        graphical: bool = True,
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
            self.laser_sound = load_sound("laser")
        else:
            super().__init__(starting_position, DummyShip(), Vector2(0))
            self.laser_sound = DummySound()

        # self.radius *= 1.5
        self.reset()

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
        for barrier in BARRIERS:
            if barrier.hit_barrier(self.position, new_position, self.radius):
                return
        self.position = new_position

    def draw(self, surface: pygame.Surface) -> None:
        angle = self.direction.angle_to(UP)
        rotated_surface = rotozoom(self.sprite, angle, 1.0)
        rotated_surface_size = Vector2(rotated_surface.get_size())
        blit_position = self.position - rotated_surface_size * 0.5
        surface.blit(rotated_surface, blit_position)

    def shoot(self) -> None:
        # Limit number of bullets
        if len(self.bullets) == self.NUM_BULLETS:
            return
        bullet_velocity = self.direction * self.BULLET_SPEED + self.velocity
        bullet = Bullet(self.position, bullet_velocity, self.graphical)
        self.bullets.append(bullet)
        self.laser_sound.play()


class Bullet(GameObject):
    def __init__(self, position, velocity, graphical):
        self.hit_barrier = False
        if graphical:
            super().__init__(position, load_sprite("bullet"), velocity)
        else:
            super().__init__(position, DummyBullet(), velocity)
        self.name = "bullet"

    def move(self, surface):
        new_position = self.position + self.velocity
        for barrier in BARRIERS:
            if barrier.hit_barrier(self.position, new_position, self.radius):
                if barrier.orientation == "vertical":
                    self.set_position((barrier.center[0], new_position[1]))
                    self.hit_barrier = True
                    return
                elif barrier.orientation == "horizontal":
                    self.set_position((new_position[0], barrier.center[1]))
                    self.hit_barrier = True
                    return
        self.set_position(new_position)


class Barrier(GameObject):
    def __init__(self, orientation: Orientation, length: int, center: Tuple[int, int]):
        self.orientation = orientation
        self.length = length
        self.center = center
        self.name = "barrier"

        if orientation == "vertical":
            self.end1 = (self.center[0], self.center[1] - self.length // 2)
            self.end2 = (self.center[0], self.center[1] + self.length // 2)

        elif orientation == "horizontal":
            self.end1 = (self.center[0] - self.length // 2, self.center[1])
            self.end2 = (self.center[0] + self.length // 2, self.center[1])

    def hit_barrier(self, pos: Tuple, new_pos, radius) -> bool:
        """There's probably a way to generalise this to diagonal barriers with linear algebra but
        cba.

        (wow this function is janky)
        """
        x, y = pos
        x_new, y_new = new_pos
        if self.orientation == "vertical":
            y_hit = min(y, y_new) > self.end1[1] - radius and max(y, y_new) < self.end2[1] + radius
            x_hit = (
                min(x, x_new) - radius < self.center[0] and max(x, x_new) + radius > self.center[0]
            )
        elif self.orientation == "horizontal":
            x_hit = min(x, x_new) > self.end1[0] - radius and max(x, x_new) < self.end2[0] + radius
            y_hit = (
                min(y, y_new) - radius < self.center[1] and max(y, y_new) + radius > self.center[1]
            )

        if y_hit and x_hit:
            return True
        return False

    def draw(self, screen):
        pygame.draw.line(screen, (255, 255, 255), self.end1, self.end2)
        # pygame.display.flip()

    def move(self, screen):
        pass


# TODO: Move me
BARRIER_LENGTH = int(GAME_SIZE[1] * 0.2)
BARRIERS = [
    Barrier(
        orientation="vertical",
        center=(int(GAME_SIZE[0] * 0.2), int(GAME_SIZE[1] * 0.5)),
        length=BARRIER_LENGTH,
    ),
    Barrier(
        orientation="vertical",
        center=(int(GAME_SIZE[0] * 0.8), int(GAME_SIZE[1] * 0.5)),
        length=BARRIER_LENGTH,
    ),
    Barrier(
        orientation="horizontal",
        center=(int(GAME_SIZE[0] * 0.5), int(GAME_SIZE[1] * 0.2)),
        length=BARRIER_LENGTH,
    ),
    Barrier(
        orientation="horizontal",
        center=(int(GAME_SIZE[0] * 0.5), int(GAME_SIZE[1] * 0.8)),
        length=BARRIER_LENGTH,
    ),
]
