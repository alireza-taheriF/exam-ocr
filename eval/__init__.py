"""Evaluation harness for Persian handwritten exam OCR.

Primary metric: mean sentence-level similarity on ``gold_test`` (never used for
training or tuning). See ``eval/metrics.py`` and README § Evaluation.
"""

from __future__ import annotations

__all__ = ["normalize", "metrics"]
