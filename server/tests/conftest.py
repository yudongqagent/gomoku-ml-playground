"""Test setup: put server/ on sys.path so `app.*` imports work."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
