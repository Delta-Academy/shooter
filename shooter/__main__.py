import cProfile
import time

from tqdm import tqdm

from game_mechanics import SpaceRocks


def n_games():
    n = 100
    space_rocks = SpaceRocks(graphical=False)
    t1 = time.time()

    n_actions = []
    for _ in tqdm(range(n)):
        space_rocks.main_loop()
        n_actions.append(space_rocks.n_actions)
        space_rocks.reset()

    t2 = time.time()
    print(f"Time per game: {(t2 - t1) / n}")
    print(f"Average number of actions: {sum(n_actions) / n}")


def graphical():

    space_rocks = SpaceRocks(graphical=True)
    space_rocks.main_loop()
    print(f"Number of actions = {space_rocks.n_actions}")


if __name__ == "__main__":

    graphical()

    # cProfile.run("n_games()", "profile.prof")
