#!/usr/bin/env python3
"""Print the architecture registry as JSON.

The training server calls this to populate the UI's architecture picker:
each entry has a name, a description, and a JSON-serializable parameter
schema with defaults (see gomoku_ml.architectures).

Usage:
    python ml/scripts/list_architectures.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml.architectures import list_architectures_json  # noqa: E402


def main() -> int:
    print(list_architectures_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
