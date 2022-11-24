# Test that no reward is given for two bots that don't shoot


import random

from delta_shooter.game_mechanics import ShooterEnv, choose_move_randomly


def non_shooter(state) -> int:
    return random.choice([0, 1, 2, 4, 5])


def test_no_shoot_no_reward() -> None:
    env = ShooterEnv(
        non_shooter,
        render=False,
        include_barriers=False,
        half_sized_game=True,
    )

    state, _, done, _ = env.reset()

    for _ in range(10_000):
        action = non_shooter(state)
        state, reward, done, _ = env.step(action)
        assert not done
        assert reward == 0


def shoot_and_forward(state) -> int:
    return random.choice([2, 3])


def test_barriers_no_reward_half_size() -> None:

    env = ShooterEnv(
        shoot_and_forward,
        render=False,
        include_barriers=True,
        half_sized_game=True,
    )

    for _ in range(10):  # Try different spawns
        state, _, done, _ = env.reset()
        for _ in range(500):
            action = shoot_and_forward(state)
            state, reward, done, _ = env.step(action)
            assert not done
            assert reward == 0


def test_barriers_no_reward_full_size() -> None:

    env = ShooterEnv(
        shoot_and_forward,
        render=False,
        include_barriers=True,
        half_sized_game=False,
    )

    for _ in range(10):  # Try different spawns
        state, _, done, _ = env.reset()
        for _ in range(500):
            action = shoot_and_forward(state)
            state, reward, done, _ = env.step(action)
            assert not done
            assert reward == 0


def test_shooting_ends_game() -> None:

    env = ShooterEnv(
        choose_move_randomly,
        render=False,
        include_barriers=False,
        half_sized_game=True,
    )

    for _ in range(50):  # Try different spawns
        state, _, done, _ = env.reset()
        # failed if get stuck in infite loop
        while not done:
            action = choose_move_randomly(state)
            state, reward, done, _ = env.step(action)

        assert done
        if reward == 0:
            assert env.player1.dead and env.player2.dead
        else:
            assert reward in {-1, 1}


def test_reward_player_wins() -> None:

    env = ShooterEnv(
        non_shooter,
        render=False,
        include_barriers=False,
        half_sized_game=True,
    )

    for _ in range(10):  # Try different spawns
        state, _, done, _ = env.reset()
        # failed if get stuck in infite loop
        while not done:
            action = choose_move_randomly(state)
            state, reward, done, _ = env.step(action)

        assert done
        assert reward == 1


def test_reward_player_loses() -> None:

    env = ShooterEnv(
        choose_move_randomly,
        render=False,
        include_barriers=False,
        half_sized_game=True,
    )

    for _ in range(10):  # Try different spawns
        state, _, done, _ = env.reset()
        # failed if get stuck in infite loop
        while not done:
            action = non_shooter(state)
            state, reward, done, _ = env.step(action)

        assert done
        assert reward == -1
