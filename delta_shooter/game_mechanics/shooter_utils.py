import random
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union

import pygame
from pygame import Color
from pygame.image import load
from pygame.math import Vector2
from pygame.mixer import Sound
from pygame.surface import Surface

# To avoid circular import
if TYPE_CHECKING:
    from models import DummyScreen

ASSET_PATH = Path(__file__).parent.resolve() / "assets"


def load_sprite(name: str, with_alpha: bool = True) -> Surface:
    path = ASSET_PATH / f"sprites/{name}.png"
    loaded_sprite = load(path)
    return loaded_sprite.convert_alpha() if with_alpha else loaded_sprite.convert()


def load_sound(name: str) -> Sound:
    path = ASSET_PATH / f"sounds/{name}.wav"
    return Sound(str(path))


def edge_barriers(
    position: Union[Vector2, Tuple[int, int]],
    radius: int,
    surface: Union[pygame.surface.Surface, "DummyScreen"],
) -> Vector2:
    x, y = position[0], position[1]

    w, h = surface.get_size()

    x = max(0 + radius, x)
    x = min(w - radius, x)

    y = max(0 + radius, y)
    y = min(h - radius, y)
    return Vector2(x, y)


def get_random_position(surface: pygame.surface.Surface) -> Vector2:
    return Vector2(
        random.randrange(surface.get_width()),
        random.randrange(surface.get_height()),
    )


def get_random_velocity(min_speed: int, max_speed: int) -> Vector2:
    speed = random.randint(min_speed, max_speed)
    angle = random.randrange(0, 360)
    return Vector2(speed, 0).rotate(angle)


def print_text(
    surface: pygame.surface.Surface,
    text: str,
    font: pygame.font.Font,
    color: Color = Color("tomato"),
) -> None:

    text_surface = font.render(text, False, color)

    rect = text_surface.get_rect()
    size = surface.get_size()
    rect.center = (size[0] // 2, size[1] // 2)

    surface.blit(text_surface, rect)
