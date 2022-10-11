from pathlib import Path

from delta_utils.check_submission import hash_game_mechanics

print(f"Game mechanics hash:\n{hash_game_mechanics(Path('.'))}")
