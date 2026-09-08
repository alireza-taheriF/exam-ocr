"""DEPRECATED — re-export shim.

This file previously held a third copy of `normalize_text` / `cer`. It now
re-exports `training.eval.metrics`, which delegates normalization to
`pipeline.normalizer`. Import from `training.eval.metrics` directly in new code.

The original rationale ("used in Colab without full repo") no longer holds: the
Colab notebook clones the repo, so `pipeline/` and `training/eval/` are both on
disk. Keeping a divergent copy is how the ZWNJ-to-space bug survived in five
places at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.eval.metrics import (  # noqa: F401
    accuracy_from_cer,
    cer,
    corpus_cer,
    corpus_wer,
    levenshtein,
    normalize_text,
    wer,
)

__all__ = [
    "normalize_text",
    "cer",
    "wer",
    "corpus_cer",
    "corpus_wer",
    "levenshtein",
    "accuracy_from_cer",
]
