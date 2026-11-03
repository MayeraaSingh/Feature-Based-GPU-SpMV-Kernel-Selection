"""
conftest.py
===========
Root-level pytest configuration.

Adds the project root to sys.path so that tests can import
`data_pipeline` and other project packages regardless of how pytest
is invoked (e.g., from the project root or from the `tests/` directory).
"""

import sys
from pathlib import Path

# Ensure the project root is always importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
