#!/usr/bin/env python3
"""Compatibility wrapper for the packaged mask_msa entrypoint."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ghostfold.entrypoints import mask_msa_main


if __name__ == "__main__":
    mask_msa_main()
