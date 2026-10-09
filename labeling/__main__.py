"""Allow ``python -m labeling.ui`` via ``python -m labeling`` with subcommand.

Also supports: ``python -m labeling.ui`` (preferred).
"""

from __future__ import annotations

import sys


def main() -> None:
    # ``python -m labeling`` → ui; ``python -m labeling.ui`` handled by ui.py
    from labeling.ui import main as ui_main

    ui_main(sys.argv[1:])


if __name__ == "__main__":
    main()
