import random
from pathlib import Path

from pygame import Color
from pygame.image import load
from pygame.math import Vector2
from pygame.mixer import Sound
from pygame.surface import Surface

HERE = Path(__file__).parent.resolve()


def load_sprite(name: str, with_alpha: bool = True) -> Surface:
    path = HERE / f"assets/sprites/{name}.png"
    loaded_sprite = load(path)

    if with_alpha:
        return loaded_sprite.convert_alpha()
    else:
        return loaded_sprite.convert()


def load_sound(name: str) -> Sound:
    path = HERE / f"assets/sounds/{name}.wav"
    return Sound(str(path))


def wrap_position(position, surface):
    x, y = position
    w, h = surface.get_size()
    return Vector2(x % w, y % h)


def edge_barriers(position, radius, surface):
    x, y = position
    w, h = surface.get_size()

    x = max(0 + radius, x)
    x = min(w - radius, x)

    y = max(0 + radius, y)
    y = min(h - radius, y)
    return Vector2(x, y)


def get_random_position(surface):
    return Vector2(
        random.randrange(surface.get_width()),
        random.randrange(surface.get_height()),
    )


def get_random_velocity(min_speed, max_speed):
    speed = random.randint(min_speed, max_speed)
    angle = random.randrange(0, 360)
    return Vector2(speed, 0).rotate(angle)


def print_text(surface, text, font, color=Color("tomato")):
    text_surface = font.render(text, False, color)

    rect = text_surface.get_rect()
    rect.center = Vector2(surface.get_size()) / 2

    surface.blit(text_surface, rect)
