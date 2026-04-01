"""
logging_setup.py — Training logger construction and handler wiring.

Exports:
    build_loggers(args, allocation_result, session_timestamp)
        -> (list[TrainingLogger], TrainingLogger)
    teardown_loggers(training_loggers)
    reconnect_module_loggers(log_level)
    build_session_config(args, env_config) -> dict
"""

import sys
import logging

from info.module_logger import set_global_log_level, _GLOBAL_FILE_HANDLERS
from info.advanced_logging import TrainingLogger
from info.module_logger import get_module_logger

logger = get_module_logger('train.logging')


def build_loggers(
    args,
    allocation_result: dict,
    session_timestamp: str,
) -> tuple:
    """
    Create one TrainingLogger per (agent, instance) pair.

    Directory layout:
      - 1 agent, 1 instance  -> logs/exp/ts/              (flat)
      - 1 agent, N instances -> logs/exp/ts/agent_0/env_N/ (per-env)
      - N agents, M instances -> logs/exp/ts/agent_N/env_M/ (full matrix)

    Returns:
        (all_loggers: list[TrainingLogger], primary_logger: TrainingLogger)
    """
    # Build instance -> owning agent mapping
    inst_to_agent = {}
    if allocation_result and 'allocation' in allocation_result:
        for aid, instances in allocation_result['allocation'].items():
            for iid in instances:
                inst_to_agent[iid] = aid
    else:
        inst_to_agent = {0: 0}

    loggers_by_env = {}

    # Simple case: single agent, single instance — flat log structure
    if args.num_instances == 1 and args.num_agents == 1:
        tl = TrainingLogger(
            experiment_name=args.name,
            base_dir="./logs",
            console_log_level=args.log_level,
            agent_id=None,
            instance_id=None,
            num_agents=1,
            session_timestamp=session_timestamp,
        )
        loggers_by_env[0] = tl
        return list(loggers_by_env.values()), tl

    # Multi-instance or multi-agent: one logger per environment
    for env_idx in range(args.num_instances):
        aid = inst_to_agent.get(env_idx, 0)
        tl = TrainingLogger(
            experiment_name=args.name,
            base_dir="./logs",
            console_log_level=args.log_level,
            agent_id=aid,
            instance_id=env_idx,
            num_agents=args.num_agents,
            session_timestamp=session_timestamp,  # Share timestamp across all loggers
        )
        loggers_by_env[env_idx] = tl

    all_loggers = list(loggers_by_env.values())
    return all_loggers, all_loggers[0]


def teardown_loggers(training_loggers: list) -> None:
    """Close file handlers from a previous logger set before rebuilding."""
    for tl in training_loggers:
        # Flush and close the training data file
        try:
            if hasattr(tl, 'training_data_file'):
                tl.training_data_file.flush()
                tl.training_data_file.close()
        except Exception:
            pass

        # Remove file handlers from all mh_* loggers
        for handler in getattr(tl, '_file_handlers', []):
            for name in list(logging.root.manager.loggerDict.keys()):
                if name.startswith('mh_'):
                    lg = logging.getLogger(name)
                    if handler in lg.handlers:
                        lg.removeHandler(handler)
            if handler in _GLOBAL_FILE_HANDLERS:
                _GLOBAL_FILE_HANDLERS.remove(handler)


def reconnect_module_loggers(log_level: str) -> None:
    """
    Reconnect file handlers to all mh_* loggers after TrainingLogger creation.

    Critical design rule:
      - Logger level is ALWAYS DEBUG (so file handlers receive everything)
      - Only the stdout StreamHandler respects --log-level
      - File handler levels are NEVER changed here (set by TrainingLogger):
          console.log  = DEBUG
          errors.log   = ERROR
          reward_debug = DEBUG
    """
    set_global_log_level(log_level)  # sets loggers to DEBUG, stdout to log_level

    # Reconnect the advanced_console_capture handler to all mh_* loggers
    # so console.log/errors.log receive output from every module.
    adv = logging.getLogger('advanced_console_capture')
    if not adv.handlers:
        return

    for name in logging.root.manager.loggerDict:
        if not name.startswith('mh_'):
            continue
        lg = logging.getLogger(name)
        # Ensure logger level stays at DEBUG (set_global_log_level already did this,
        # but be explicit in case something else changed it)
        lg.setLevel(logging.DEBUG)

        for handler in adv.handlers:
            if handler not in lg.handlers:
                lg.addHandler(handler)
                # Do NOT change the handler's level — it was set correctly
                # by TrainingLogger._setup_loggers()


def build_session_config(args, env_config: dict) -> dict:
    """Build a session-summary dict suitable for JSON persistence."""
    return {
        'command': ' '.join(sys.argv),
        'cli_args': vars(args),
        'runtime': {
            'experiment_name': args.name,
            'input_method': 'DLL injection (keyboard)',  # Always DLL, no vgamepad
            'device': 'pending',     # resolved later
            'timesteps_actual': args.timesteps,
            'scenario': 'pending',   # resolved later
            'num_agents': args.num_agents,
            'num_instances': args.num_instances,
        },
        'environment': env_config,
    }