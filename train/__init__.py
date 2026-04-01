"""
train — Monster Hunter Tri RL training package.

Decomposed from the monolithic train.py into focused modules:

    cli.py           – Command-line argument definitions
    dolphin.py       – Dolphin process management (launch, detect, cleanup)
    allocation.py    – Agent-to-instance allocation & validation
    logging_setup.py – TrainingLogger construction & handler wiring
    environment.py   – Environment creation (single / multi-instance)
    agents.py        – PPO agent creation & checkpoint loading
    callbacks.py     – SB3 callbacks (GUI update, checkpoints, logging)
    runner.py        – main() orchestrator (startup, train loop, save, cleanup)
    __main__.py      – Entry point (python -m train)
"""

from train.runner import main  # noqa: F401

__all__ = ["main"]
