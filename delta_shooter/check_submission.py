from pathlib import Path

import delta_utils.check_submission as checker
from torch import nn

from game_mechanics import ShooterEnv, load_network


def check_submission(team_name: str) -> None:

    example_state, _, _, _ = ShooterEnv(lambda x: x, render=False).reset()

    expected_choose_move_return_type = int
    game_mechanics_expected_hash = (
        "2925cc219430589b7b976fbc298cbe7f3819ed21ed5055410528606f8cef2400"
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
