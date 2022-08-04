import pytest

from game_mechanics import Env


def test_env():
    """Take me out in real games."""
    with pytest.raises(NotImplementedError):
        env = Env()
