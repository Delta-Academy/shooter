from pathlib import Path

import delta_utils.check_submission as checker
from torch import nn

from game_mechanics import ShooterEnv, load_network


def check_submission(team_name: str) -> None:

    example_state = ShooterEnv(lambda x: x, render=False).reset()

    expected_choose_move_return_type = int
    game_mechanics_expected_hash = (
        "08c245cda934111fd2dea916c86440e0e608108f060b1b4b5f78d9914c379ffc"
    )
    expected_pkl_output_type = nn.Module
    pkl_file = load_network(team_name)

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=pkl_file,
        pkl_checker_function=lambda x: x,
        game_mechanics_hash=game_mechanics_expected_hash,
        current_folder=Path(__file__).parent.resolve(),
    )
