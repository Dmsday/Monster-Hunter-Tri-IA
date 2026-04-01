"""
runner.py — Main training orchestrator.

Replaces the monolithic main() from the original train.py.
Coordinates: CLI -> logging -> Dolphin -> environments -> agents -> train loop -> cleanup.
"""

import os
import sys
import time
import atexit
import traceback
import threading
from datetime import datetime

import torch
import psutil
import tkinter as tk
from stable_baselines3.common.vec_env import VecNormalize

from info.module_logger import get_module_logger, set_global_log_level
from info.agent_context import AgentContext

from core.controller.action_heads import HEAD_NAME_TO_IDX

from train.cli import build_parser, post_process_args
from train.dolphin import (
    resolve_dolphin_path,
    launch_dolphin_instances_via_powershell,
    wait_for_dolphin_windows,
    read_pids_from_temp,
    clean_pid_files,
    close_existing_dolphin_instances,
    cleanup_dolphin_processes,
    register_signal_handlers,
    global_dolphin_pids,
)
from train.allocation import calculate_agent_allocation,  validate_multi_agent_args
from train.logging_setup import (
    build_loggers,
    teardown_loggers,
    reconnect_module_loggers,
    build_session_config,
)
from train.environment import (
    resolve_env_flags,
    create_single_instance_env,
    create_multi_instance_envs,
    wrap_vec_normalize,
    validate_multi_instance_windows,
    run_vision_test,
    run_exploration_map_test,
)
from train.agents import load_or_create_single_agent, create_multi_agents
from train.callbacks import build_callbacks, ProgressWindowCallback

from GUI.training_gui import TrainingGUI

logger = get_module_logger('train')

# Global flag to prevent double cleanup (atexit + finally)
_cleanup_done = False

# ======================================================================
#  OPTIONAL MULTI-AGENT IMPORTS
# ======================================================================

try:
    from multi.multi_agent_scheduler import MultiAgentScheduler
    from multi.multi_agent_trainer import MultiAgentTrainer
    from multi.genetic_trainer import GeneticTrainer
    MULTI_AGENT_AVAILABLE = True
except ImportError as _exc:
    MultiAgentScheduler = MultiAgentTrainer = GeneticTrainer = None
    MULTI_AGENT_AVAILABLE = False
    logger.warning(f"Multi-agent modules not available: {_exc}")


# ======================================================================
#  MAIN ENTRY POINT
# ======================================================================

def main():
    global _cleanup_done

    # -- 1. Parse CLI arguments --------------------------------------------
    args = post_process_args(build_parser().parse_args())
    use_vision, use_memory = resolve_env_flags(args)

    # Validate multi-agent configuration before any resource allocation
    if args.num_agents > 1 or args.num_instances > 1:
        validate_multi_agent_args(args)  # Raises on invalid config

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -- 2. Logging setup (level only — handler reconnection after loggers exist)
    set_global_log_level(args.log_level)

    env_config = {
        'mode': args.env,
        'use_vision': use_vision,
        'use_memory': use_memory,
        'grayscale': args.grayscale,
        'frame_stack': 4,
        'frame_size': '84x84',
        'multi_instance': args.num_instances > 1,
        'num_instances': args.num_instances,
        'num_agents': args.num_agents,
    }
    session_config = build_session_config(args, env_config)

    # Generate experiment name if not provided
    if args.name is None:
        args.name = f"mh_{session_ts}"

    logs_dir = f"./logs/{args.name}"
    os.makedirs(logs_dir, exist_ok=True)

    # Models directory : timestamped subfolder (like logs),
    # except when resuming --> reuse existing directory
    if args.resume:
        models_dir = f"./models/{args.name}"
    else:
        models_dir = f"./models/{args.name}/{session_ts}"
    os.makedirs(models_dir, exist_ok=True)

    _log_startup_info(args, use_vision, use_memory)

    # -- 3. Compute device -------------------------------------------------
    device = _resolve_device(args)

    # -- 4. Start GUI ------------------------------------------------------
    gui = _create_gui(args)

    # -- 5. Create environment(s) ------------------------------------------
    base_env = None
    env = None
    allocation_result = None
    training_loggers = []
    training_logger = None

    try:
        if args.num_instances == 1:
            # --- Single instance ---
            base_env, env, allocation_result = create_single_instance_env(args, gui)

            training_loggers, training_logger = build_loggers(args, allocation_result, session_ts)
            reconnect_module_loggers(args.log_level)
            for tl in training_loggers:
                tl.log_config(session_config)

            run_vision_test(base_env, training_logger)
            run_exploration_map_test(base_env, training_logger)
        else:
            # --- Multi instance ---
            allocation_result = _setup_multi_instance(args, gui, session_config, session_ts)
            if allocation_result is None:
                return  # setup failed — already cleaned up

            training_loggers, training_logger = build_loggers(args, allocation_result, session_ts)
            reconnect_module_loggers(args.log_level)
            for tl in training_loggers:
                tl.log_config(session_config)

            env = create_multi_instance_envs(args, allocation_result, gui)

        # --- VecNormalize wrapper ---
        env = wrap_vec_normalize(env, args, logs_dir, training_logger)

    except (RuntimeError, ValueError, ImportError, OSError) as exc:
        _handle_env_creation_failure(exc, args, gui, training_logger, training_loggers)
        _cleanup_done = True
        return

    # --- Validate multi-instance window assignment ---
    if args.num_instances > 1:
        if not validate_multi_instance_windows(env, args.num_instances):
            logger.warning("Window title mismatch detected — games may still be loading")
            logger.warning("Training will continue, but verify that each instance "
                           "captures the correct window")

    # -- 6. Create agent(s) ------------------------------------------------
    agents = []
    scheduler = None
    previous_timesteps = 0

    try:
        if args.num_agents == 1:
            agent, previous_timesteps = load_or_create_single_agent(
                args, env, device, training_logger, base_env)
            agents = [agent]
        else:
            agents = create_multi_agents(args, env, device, logs_dir)

            if not MULTI_AGENT_AVAILABLE:
                raise ImportError("MultiAgentScheduler module not available")

            scheduler = MultiAgentScheduler(
                agents=agents,
                allocation=allocation_result['allocation'],
                mode=args.multi_agent_mode,
                block_size=args.block_size,
                weighted_eval_freq=getattr(args, 'weighted_eval_freq', 100),
            )
            previous_timesteps = 0

    except Exception as exc:
        logger.error(f"Agent creation error: {exc}")
        traceback.print_exc()
        _log_error_all(exc, "Agent creation", training_loggers, training_logger)
        _safe_close(gui, env)
        return

    # -- 7. Build callbacks ------------------------------------------------
    callbacks, gui_callback, logging_callbacks = build_callbacks(
        args, gui, env, models_dir, training_logger, training_loggers, allocation_result)

    # -- 8. Pre-start countdown / wait for user ----------------------------
    try:
        if not _wait_for_start(args, gui):
            _safe_close(gui, env)
            _cleanup_done = True
            return
    except KeyboardInterrupt:
        _handle_prestart_ctrl_c(args, gui, env, allocation_result)
        _cleanup_done = True
        return

    # Clean up PID files now that training is confirmed
    if args.num_instances > 1:
        clean_pid_files(args.num_instances)
    # Guard against GUI thread crash before we reach this point
    if gui and hasattr(gui, 'stop_button') and gui.running:
        try:
            gui.stop_button.config(state=tk.NORMAL)
        except Exception as _e:
            logger.warning(f"Could not enable stop button (GUI may have crashed): {_e}")

    # -- 9. Training loop --------------------------------------------------
    try:
        timesteps_to_train = args.debug_steps or args.timesteps
        logger.warning("Starting training...")

        if len(agents) == 1:
            _train_single_agent(agents[0], timesteps_to_train, callbacks, gui)
        else:
            _train_multi_agent(
                agents, env, scheduler, timesteps_to_train,
                args, gui, gui_callback, logging_callbacks,
                models_dir, training_logger, allocation_result)

        # Display final exploration statistics
        _log_final_exploration_map(env)

        # Save final model(s)
        _save_final(agents, env, models_dir, training_logger)

    except KeyboardInterrupt:
        _handle_training_ctrl_c(agents, env, models_dir, training_logger, training_loggers)

    except Exception as exc:
        _log_error_all(exc, "Training loop", training_loggers, training_logger)
        logger.error(f"Training error: {exc}")
        traceback.print_exc()

    finally:
        _final_cleanup(args, env, gui, allocation_result, training_loggers)


# ======================================================================
#  MULTI-INSTANCE SETUP
# ======================================================================

def _setup_multi_instance(args, gui, session_config, session_ts) -> dict | None:
    """
    Launch Dolphin, read PIDs, compute allocation.
    Returns allocation_result or None on failure.
    """
    logger.info(f"Multi-instance mode: {args.num_instances} instances")

    dolphin_path = resolve_dolphin_path(args.dolphin_path)
    if not os.path.exists(dolphin_path):
        logger.error(f"Dolphin path not found: {dolphin_path}")
        _safe_close(gui, None)
        return None

    # Close any existing Dolphin processes
    if not close_existing_dolphin_instances():
        logger.error("Could not close existing Dolphin instances — aborting")
        _safe_close(gui, None)
        return None

    # Launch new instances
    ok = launch_dolphin_instances_via_powershell(
        num_instances=args.num_instances,
        dolphin_path=dolphin_path,
        minimize_dolphin=True,
        minimize_game=False,
    )
    if not ok:
        logger.error("Failed to launch Dolphin instances")
        _safe_close(gui, None)
        return None

    # Read PIDs from temp files
    pids = read_pids_from_temp(args.num_instances)
    valid_pids = [p for p in pids if p is not None and p > 0]
    if not valid_pids:
        logger.error("No Dolphin PIDs retrieved — PowerShell may have failed")
        _safe_close(gui, None)
        return None

    # Register PIDs globally for cleanup
    global_dolphin_pids.clear()
    global_dolphin_pids.extend(valid_pids)
    register_signal_handlers()

    if gui is not None:
        gui._dolphin_pids_ref = global_dolphin_pids

    atexit.register(_emergency_atexit, valid_pids)

    # Wait for game windows to appear
    if not wait_for_dolphin_windows(args.num_instances, timeout=args.dolphin_timeout,
                                     check_interval=10):
        logger.error("Not all Dolphin windows detected within timeout")
        _safe_close(gui, None)
        return None

    # Compute agent-to-instance allocation
    allocation_result = calculate_agent_allocation(
        num_agents=args.num_agents,
        num_instances=args.num_instances,
        allocation_mode=args.allocation_mode,
        allocation_map=args.allocation_map,
        multi_agent_mode=args.multi_agent_mode,
    )
    allocation_result['dolphin_pids'] = pids
    return allocation_result


# ======================================================================
#  TRAINING DISPATCHERS
# ======================================================================

def _train_single_agent(agent, timesteps, callbacks, gui):
    """Run SB3 learn() for a single PPO agent."""
    if gui is not None:
        gui.set_total_timesteps(timesteps)
    else:
        callbacks.append(ProgressWindowCallback(total_timesteps=timesteps, num_envs=1))

    agent.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=False,
    )


def _train_multi_agent(agents, env, scheduler, timesteps, args, gui,
                       gui_callback, logging_callbacks, models_dir,
                       training_logger, allocation_result):
    """Run the multi-agent training loop."""
    if not MULTI_AGENT_AVAILABLE:
        logger.error("Multi-agent modules not available — fallback to single agent")
        agents[0].learn(total_timesteps=timesteps, progress_bar=True,
                        reset_num_timesteps=False)
        return

    # Genetic mode
    if args.multi_agent_mode == 'genetic':
        if GeneticTrainer is None:
            logger.error("GeneticTrainer not available")
            return
        gt = GeneticTrainer(
            agents=agents, env=env,
            elite_ratio=args.genetic_elite_ratio,
            mutation_rate=args.genetic_mutation_rate,
            episodes_per_eval=10,
        )
        results = gt.train(num_generations=args.genetic_generations, progress_bar=True)
        agents[:] = results['final_agents']
        return

    # Standard multi-agent modes (independent, round_robin, majority_vote)
    trainer = MultiAgentTrainer(
        agents=agents, env=env, scheduler=scheduler,
        steps_per_agent=args.steps_per_agent,
        callback=gui_callback,
        agent_callbacks=logging_callbacks,
        scenario=str(allocation_result['scenario']),
        allocation=allocation_result['allocation'],
        models_dir=models_dir,
    )

    if gui is not None:
        gui.set_total_timesteps(timesteps)

    trainer.train(
        total_timesteps=timesteps,
        progress_bar=True,
        show_progress_window=(gui is None),
    )


# ======================================================================
#  SAVE
# ======================================================================

def _save_final(agents, env, models_dir, training_logger):
    """Save final model(s) and VecNormalize wrapper."""
    if len(agents) == 1:
        path = os.path.join(models_dir, "final_model")
        agents[0].save(path)
        logger.info(f"Model saved: {path}.zip ({agents[0].num_timesteps:,} steps)")
    else:
        for aid, agent in enumerate(agents):
            d = os.path.join(models_dir, f"agent_{aid}")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "final_model")
            agent.save(path)
            logger.info(f"  Agent {aid}: {path}.zip ({agent.num_timesteps:,} steps)")

    _save_vec_normalize(env, models_dir, training_logger)
    logger.warning("TRAINING COMPLETE")


def _save_vec_normalize(env, models_dir, training_logger, prefix="vec_normalize"):
    """Save the VecNormalize wrapper state if the env is wrapped."""
    try:
        path = os.path.join(models_dir, f"{prefix}.pkl")
        if isinstance(env, VecNormalize):
            env.save(path)
            logger.info(f"VecNormalize saved: {path}")
    except (OSError, AttributeError) as exc:
        if training_logger:
            training_logger.log_error(exc, context=f"Save {prefix}")
        logger.error(f"VecNormalize save failed: {exc}")


# ======================================================================
#  CLEANUP & ERROR HANDLING
# ======================================================================

def _final_cleanup(args, env, gui, allocation_result, training_loggers=None):
    """Always-runs cleanup executed in the finally block.

    Order matters — close env first (stops frame capture threads),
    THEN kill Dolphin (so no thread tries to capture from dead windows).
    Training loggers are closed last so they can capture all shutdown events.
    """
    global _cleanup_done

    had_dolphin = False
    dolphin_ok = True

    # -- 1. Suppress frame_capture AND dll capture warnings during shutdown
    import logging as _logging
    _fc_logger = _logging.getLogger('mh_frame_capture')
    _fc_original_level = _fc_logger.level
    _fc_logger.setLevel(_logging.ERROR)
    # Also suppress dll capture: Dolphin may already be dead (signal handler ran first)
    _dll_logger = _logging.getLogger('mh_dolphin_capture_dll')
    _dll_original_level = _dll_logger.level
    _dll_logger.setLevel(_logging.ERROR)

    # -- 2. Release controller inputs (give Dolphin one poll cycle) --------
    try:
        if env is not None:
            if hasattr(env, 'envs'):
                for e in env.envs:
                    ctrl = getattr(e, 'controller', None)
                    if ctrl and hasattr(ctrl, 'reset_all'):
                        ctrl.reset_all()
            elif hasattr(env, 'controller'):
                ctrl = getattr(env, 'controller', None)
                if ctrl and hasattr(ctrl, 'reset_all'):
                    ctrl.reset_all()
        time.sleep(0.08)
    except Exception:
        pass

    # -- 3. Clean up controller resources ----------------------------------
    try:
        if env is not None:
            ctrl = None
            if hasattr(env, 'get_attr'):
                try:
                    ctrls = env.get_attr('controller')
                    while isinstance(ctrls, list) and ctrls:
                        ctrls = ctrls[0]
                    ctrl = ctrls
                except (AttributeError, IndexError):
                    pass
            if ctrl is None and hasattr(env, 'envs'):
                try:
                    ctrl = env.envs[0].controller
                except (AttributeError, IndexError):
                    pass
            if ctrl and hasattr(ctrl, 'cleanup'):
                ctrl.cleanup()
    except Exception:
        pass

    # -- 4. Close environment (stops frame capture threads) ----------------
    try:
        if env is not None:
            env.close()
    except Exception:
        pass

    # -- 5. Close Dolphin instances — skip if signal handler already did it
    import train.dolphin as _dmod
    if _dmod.global_cleanup_done:
        # Signal handler already killed all instances — nothing to do
        logger.debug("Skipping Dolphin cleanup: already done by signal handler")
        had_dolphin = True
        dolphin_ok = True
    else:
        try:
            pids = []
            if allocation_result:
                pids = allocation_result.get('dolphin_pids', [])
            if not pids:
                pids = list(global_dolphin_pids)
            valid = [p for p in pids if p is not None and p > 0]
            if valid:
                had_dolphin = True
                cleanup_dolphin_processes(valid, emergency=False)
                time.sleep(0.5)
                still = [p for p in valid if psutil.pid_exists(p)]
                dolphin_ok = len(still) == 0
        except Exception as exc:
            logger.error(f"Dolphin cleanup error: {exc}")
            dolphin_ok = False

    # -- 6. Clean leftover PID files ---------------------------------------
    try:
        if args.num_instances > 1:
            clean_pid_files(args.num_instances)
    except Exception:
        pass

    if had_dolphin and not dolphin_ok:
        logger.error("Some Dolphin instances may still be running — check Task Manager")

    # -- 7. Restore suppressed log levels ----------------------------------
    _fc_logger.setLevel(_fc_original_level)
    _dll_logger.setLevel(_dll_original_level)

    # -- 8. Close training loggers (writes "no errors" banner + session summary)
    try:
        if training_loggers:
            for tl in training_loggers:
                try:
                    tl.close()
                except Exception:
                    pass
    except Exception:
        pass

    # -- 9. Close GUI ------------------------------------------------------
    try:
        if gui is not None:
            gui.close()
    except Exception:
        pass

    _cleanup_done = True
    import train.dolphin as _dmod
    _dmod.global_cleanup_done = True

    print("\nCleanup completed — training terminated")
    os._exit(0)


def _handle_env_creation_failure(error, args, gui, training_logger, training_loggers):
    """Log the error and clean up Dolphin after environment creation failure."""
    if training_logger:
        training_logger.log_error(error, context="Environment creation")
    else:
        logger.error(f"Environment creation failed: {error}")
    traceback.print_exc()

    pids = list(global_dolphin_pids)
    if pids:
        cleanup_dolphin_processes(pids, emergency=True)

    _safe_close(gui, None)


def _handle_training_ctrl_c(agents, env, models_dir, training_logger, training_loggers):
    """Save interrupted model(s) when the user presses Ctrl+C during training."""
    if training_logger:
        training_logger.log_warning("Training interrupted by user (Ctrl+C)")
    logger.warning("Interrupted (Ctrl+C)")

    for aid, agent in enumerate(agents):
        try:
            if len(agents) == 1:
                path = os.path.join(models_dir, f"interrupted_{agent.num_timesteps}steps")
            else:
                d = os.path.join(models_dir, f"agent_{aid}")
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"interrupted_{agent.num_timesteps}steps")
            agent.save(path)
            logger.info(f"Interrupted model saved: {path}.zip ({agent.num_timesteps:,} steps)")
        except Exception as exc:
            logger.error(f"Save error (agent {aid}): {exc}")

    _save_vec_normalize(env, models_dir, training_logger, prefix="interrupted_vec_normalize")


def _handle_prestart_ctrl_c(args, gui, env, allocation_result):
    """Clean up when the user cancels before training starts."""
    logger.info("Cancelled before training start")

    pids = list(global_dolphin_pids)
    if args.num_instances > 1:
        file_pids = read_pids_from_temp(args.num_instances)
        pids.extend([p for p in file_pids if p and p > 0 and p not in pids])
        clean_pid_files(args.num_instances)

    if pids:
        cleanup_dolphin_processes(pids, emergency=False)

    _safe_close(gui, env)


def _safe_close(gui, env):
    """Attempt to close the environment and GUI without raising."""
    try:
        if env is not None:
            env.close()
    except Exception:
        pass
    try:
        if gui is not None:
            gui.close()
    except Exception:
        pass


def _log_error_all(error, context, training_loggers, training_logger):
    """Log an error to every available TrainingLogger."""
    if training_loggers:
        for tl in training_loggers:
            if tl.agent_id is not None:
                AgentContext.set_current_agent(tl.agent_id)
            tl.log_error(error, context=context)
    elif training_logger:
        training_logger.log_error(error, context=context)


def _emergency_atexit(pids):
    """atexit handler — runs if the process terminates unexpectedly."""
    global _cleanup_done
    if _cleanup_done:
        return
    logger.warning("Emergency atexit cleanup triggered")
    try:
        cleanup_dolphin_processes(pids, emergency=True)
    except Exception:
        pass
    _cleanup_done = True


# ======================================================================
#  PRE-START WAIT
# ======================================================================

def _wait_for_start(args, gui) -> bool:
    """
    Wait for the user to press ENTER or auto-start after a countdown.
    Returns True to proceed, False to cancel.
    """
    if gui:
        return _wait_with_gui(gui)
    else:
        return _wait_no_gui()


def _wait_with_gui(gui) -> bool:
    """GUI mode: ENTER to start immediately, or auto-start after 10 seconds."""
    # If GUI crashed (window destroyed), fall back to headless countdown
    if not gui.running and gui.window is None:
        logger.warning("GUI crashed — falling back to headless 5s auto-start")
        return _wait_no_gui()

    logger.warning("Press ENTER to start training (auto-start in 10s)...")

    start_flag = {'value': False, 'by_user': False}
    shutdown = {'value': False}

    def _wait_enter():
        try:
            input()
        except (EOFError, KeyboardInterrupt, UnicodeDecodeError):
            return
        except Exception:
            return
        if not shutdown['value']:
            start_flag['value'] = True
            start_flag['by_user'] = True

    thread = threading.Thread(target=_wait_enter, daemon=True)
    thread.start()

    for countdown in range(10, 0, -1):
        if start_flag['value']:
            break
        if gui.should_stop():
            shutdown['value'] = True
            thread.join(timeout=1.0)
            return False
        print(f"\rAuto-start in {countdown}s... ", end='', flush=True)
        time.sleep(1)

    if start_flag['by_user']:
        logger.warning("Manual start (ENTER pressed)")
    else:
        logger.warning("Automatic start (timeout)")

    return True


def _wait_no_gui() -> bool:
    """No-GUI mode: auto-start after 5 seconds, Ctrl+C to cancel."""
    logger.info("Auto-start in 5 seconds (Ctrl+C to cancel)")
    try:
        for countdown in range(5, 0, -1):
            print(f"\r  Starting in {countdown}s...  ", end='', flush=True)
            time.sleep(1)
        print("\r  Starting now...           ")
        return True
    except KeyboardInterrupt:
        logger.info("Start cancelled by user")
        return False


# ======================================================================
#  HELPERS
# ======================================================================

def _resolve_device(args) -> str:
    """Determine whether to use GPU or CPU."""
    if args.cpu:
        logger.info("Device: CPU (forced)")
        return 'cpu'
    try:
        if torch.cuda.is_available():
            logger.info(f"Device: GPU — {torch.cuda.get_device_name(0)}")
            return 'cuda'
    except (ImportError, AttributeError, OSError):
        pass
    logger.info("Device: CPU")
    return 'cpu'


def _create_gui(args):
    """Create and start the training GUI, or return None if disabled."""
    if args.no_gui:
        return None
    try:
        gui = TrainingGUI(title=f"MH Training — {args.name}")

        # Pass disabled head indices to GUI for action tab display
        _disabled_indices = set()
        for name in getattr(args, 'disabled_heads', ['menu']):
            idx = HEAD_NAME_TO_IDX.get(name)
            if idx is not None:
                _disabled_indices.add(idx)
        gui.set_disabled_heads(_disabled_indices)
        gui.start()
        logger.info("GUI started")
        return gui
    except Exception as exc:
        logger.error(f"GUI initialization failed: {exc}")
        return None


def _log_startup_info(args, use_vision, use_memory):
    """Print a summary of the training configuration to the log."""
    logger.info(f"Experiment : {args.name or 'auto'}")
    logger.info(f"Timesteps  : {args.timesteps:,}")
    if args.resume:
        logger.info(f"Resume     : {args.resume}")
    mode_map = {'visual': 'Vision only', 'memory': 'Memory only', 'hybrid': 'Hybrid (vision + memory)'}
    logger.info(f"Env mode   : {mode_map[args.env]}")
    logger.info(f"Grayscale  : {'Yes' if args.grayscale else 'No'}")
    logger.info(f"Input      : DLL injection (focus-free)")
    if args.debug_steps:
        logger.info(f"DEBUG mode : limited to {args.debug_steps} steps")


def _log_final_exploration_map(env):
    """Display the final exploration statistics at the end of training."""
    try:
        if hasattr(env, 'get_attr'):
            rc = env.get_attr('reward_calc')[0]
            if rc and hasattr(rc, 'exploration_tracker'):
                info = rc.exploration_tracker.get_detailed_map_info()
                for line in info.split('\n'):
                    if line.strip():
                        logger.info(line)
    except (AttributeError, KeyError, IndexError, TypeError):
        pass