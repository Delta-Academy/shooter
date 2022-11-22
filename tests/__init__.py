import sys
from pathlib import Path

parent = Path(__file__).parent.parent.resolve()

sys.path.append(str(parent / "delta_shooter"))
