"""Shared loader for the CLI scripts under test.

`scripts/` is deliberately not a package — the scripts are run as
`python scripts/foo.py` and each inserts the repo root on `sys.path` itself. So
tests load them by file path. The module must be registered in `sys.modules`
before `exec_module`, otherwise `@dataclass` fails resolving its own module's
namespace while the module is still executing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def extract_script():
    return load_script("extract_under_test", "scripts/extract_and_align_lines.py")


@pytest.fixture(scope="session")
def split_script():
    return load_script("split_under_test", "scripts/split_dataset.py")
