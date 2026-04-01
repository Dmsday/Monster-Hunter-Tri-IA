"""
train.py — Backward-compatible entry point.

The training logic has been decomposed into the `train/` package.
This file simply delegates to `train.runner.main()`.

Usage unchanged:
    python train.py --timesteps 100000
    python train.py --resume ./models/exp/checkpoint.zip
    python train.py --num-agents 4 --num-instances 4
"""

import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')   # Suppress duplicate oneDNN info message

import sys
import traceback
from info.module_logger import get_module_logger

logger = get_module_logger('train')

if __name__ == "__main__":
    try:
        from train.runner import main
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted (Ctrl+C) — shutting down cleanly")
        sys.exit(0)
    except Exception as critical_error:
        logger.error(f"CRITICAL ERROR: {critical_error}")
        traceback.print_exc()
        sys.exit(1)
