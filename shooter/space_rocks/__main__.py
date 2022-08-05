import cProfile
import time

from tqdm import tqdm

from game import SpaceRocks


def n_games():
    n = 25
    space_rocks = SpaceRocks(graphical=False)
    t1 = time.time()
    for _ in tqdm(range(n)):
        space_rocks.main_loop()
        space_rocks.reset()

    t2 = time.time()
    print((t2 - t1) / n)


def graphical():

    space_rocks = SpaceRocks(graphical=True)
    space_rocks.main_loop()


if __name__ == "__main__":

    # graphical()
    cProfile.run("n_games()", "profile.prof")
