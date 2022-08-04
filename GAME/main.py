from typing import Any, Dict

from check_submission import check_submission
from game_mechanics import Env, choose_move_randomly, load_pkl, play_game, save_pkl

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> Dict:
    """
    TODO: Write this function to train your algorithm.

    Returns:
    """
    raise NotImplementedError("You need to implement this function!")


def choose_move(state: Any, user_file: Any, verbose: bool = False) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state:

    Returns:
    """
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    file = train()
    save_pkl(file, TEAM_NAME)

    check_submission(
        TEAM_NAME
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    my_value_fn = load_pkl(TEAM_NAME)

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_value_fn(state: Any) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_value_fn)

    play_game(
        your_choose_move=choose_move_no_value_fn,
        opponent_choose_move=choose_move_randomly,
        game_speed_multiplier=1,
        render=True,
        verbose=False,
    )
