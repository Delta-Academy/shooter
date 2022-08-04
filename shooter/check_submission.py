from pathlib import Path

import delta_utils.check_submission as checker


def check_submission(team_name: str) -> None:
    example_state = ...
    expected_choose_move_return_type = ...
    game_mechanics_expected_hash = ...
    expected_pkl_output_type = ...
    pkl_file = ...

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=pkl_file,
        pkl_checker_function=checker.pkl_checker_value_dict,
        game_mechanics_hash=game_mechanics_expected_hash,
        current_folder=Path(__file__).parent.resolve(),
    )
