"""Make the repo root importable so tests can `import pipeline.…` / `training.…`
regardless of the directory pytest is invoked from."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
