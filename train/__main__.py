"""
Entry point <--> allows running the package directly via `python -m train`.
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
