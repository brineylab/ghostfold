#!/usr/bin/env python3
"""Compatibility wrapper for the packaged Neff calculator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ghostfold.entrypoints import neff_main


if __name__ == "__main__":
    neff_main()
