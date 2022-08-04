from typing import Any, Callable, Dict, Tuple


def save_pkl(file: Any, team_name: str):
    """Save a user PKL."""
    pass


def load_pkl(team_name: str):
    """Load a user PKL."""
    pass


def choose_move_randomly(state):
    pass


def play_game(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier=1,
    render=True,
    verbose=False,
):
    pass


class Env:
    def __init__(self):
        """Initialises the object.

        Is called when you call `environment = Env()`.

        It sets everything up in the starting state for the episode to run.
        """
        self.reset()

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Given an action to take:
            1. sample the next state and update the state
            2. get the reward from this timestep
            3. determine whether the episode has terminated

        Args:
            action: The action to take. Determined by user code
                that runs the policy

        Returns:
            Tuple of:
                1. state (Any): The updated state after taking the action
                2. reward (float): Reward at this timestep
                3. done (boolean): Whether the episode is over
                4. info (dict): Dictionary of extra information
        """
        raise NotImplementedError()

    def reset(self) -> Tuple[Any, float, bool, Dict]:
        """Resets the environment (resetting state, total return & whether the episode has
        terminated) so it can be re-used for another episode.

        Returns:
            Same type output as .step(). Tuple of:
                1. state (Any): The state after resetting the environment
                2. reward (None): None at this point, since no reward is
                   given initially
                3. done (boolean): Always `True`, since the episode has just
                   been reset
                4. info (dict): Dictionary of any extra information
        """
        raise NotImplementedError()
