"""Make the ``gomoku_ml`` package importable when running
``python -m pytest ml/tests`` from the repo root."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
