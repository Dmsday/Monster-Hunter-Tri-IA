"""
environment.py — Environment creation for single and multi-instance modes.

Exports:
    resolve_env_flags(args)       -> (use_vision, use_memory)
    create_single_instance_env()  -> (base_env, DummyVecEnv, allocation_result)
    create_multi_instance_envs()  -> DummyVecEnv
    wrap_vec_normalize()          -> VecNormalize | DummyVecEnv
    validate_multi_instance_windows() -> bool
    run_vision_test()
    run_exploration_map_test()
"""

import os
import threading

import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.mh_env import MonsterHunterEnv
from environment.realtime_display import SurveillanceWindow  # Extracted to its own module
import environment.realtime_display as _rt_display_module     # _surveillance_win lives here now
from info.agent_context import AgentContext, EnvContext
from info.module_logger import get_module_logger

from train.dolphin import (
    is_mhtri_window_open,
    resolve_dolphin_path,
    launch_dolphin_instances_via_powershell,
    wait_for_dolphin_windows,
    read_pids_from_temp,
    register_signal_handlers,
    global_dolphin_pids,
)

logger = get_module_logger('train.environment')


# ======================================================================
#  HELPERS
# ======================================================================

def resolve_env_flags(args):
    """Derive (use_vision, use_memory) booleans from the --env flag."""
    return (
        args.env in ('visual', 'hybrid'),
        args.env in ('memory', 'hybrid'),
    )


# ======================================================================
#  SINGLE-INSTANCE CREATION
# ======================================================================

def create_single_instance_env(args, gui=None):
    """
    Create the environment for single-instance mode.
    Auto-launches Dolphin if no MH Tri window is found.

    Returns:
        (base_env, vec_env, allocation_result)
    """
    use_vision, use_memory = resolve_env_flags(args)

    # Auto-launch Dolphin when no game window exists
    if not is_mhtri_window_open():
        logger.info("No Dolphin / MH Tri window detected — auto-launching...")
        dolphin_path = resolve_dolphin_path(args.dolphin_path)

        ok = launch_dolphin_instances_via_powershell(
            num_instances=1, dolphin_path=dolphin_path,
            minimize_dolphin=True, minimize_game=False,
        )
        if ok:
            logger.info("Dolphin launched — waiting for game window...")
            wait_for_dolphin_windows(1, timeout=args.dolphin_timeout, check_interval=5)

        # Register the single PID for cleanup
        pids = read_pids_from_temp(1)
        valid = [p for p in pids if p is not None and p > 0]
        if valid:
            global_dolphin_pids.clear()
            global_dolphin_pids.extend(valid)
            register_signal_handlers()
    else:
        logger.info("Existing MH Tri window detected — connecting")

    # Create the base environment
    disabled_heads = getattr(args, 'disabled_heads', ['menu'])

    base_env = MonsterHunterEnv(
        use_vision=use_vision,
        use_memory=use_memory,
        grayscale=args.grayscale,
        frame_stack=4,
        use_controller=True,
        controller_debug=False,
        use_advanced_rewards=True,
        save_state_slot=args.save_state,
        rt_vision=args.rtvision and use_vision,
        rt_minimap=args.rtminimap and args.rtvision and use_vision and use_memory,
        instance_id=0,
        disabled_heads=disabled_heads,
    )

    logger.info("Environment created")
    if isinstance(base_env.action_space, spaces.Discrete):
        n_actions = base_env.action_space.n
        if n_actions != 19:
            logger.error(f"Expected 19 actions, got {n_actions}")

    allocation_result = {
        'scenario': 'ONE_TO_ONE',
        'allocation': {0: [0]},
        'num_agents': 1,
        'num_instances': 1,
        'dolphin_pids': list(global_dolphin_pids),
    }

    vec_env = DummyVecEnv([lambda: base_env])
    return base_env, vec_env, allocation_result


# ======================================================================
#  MULTI-INSTANCE CREATION
# ======================================================================

def create_multi_instance_envs(args, allocation_result, gui=None):
    """
    Create N MonsterHunterEnv instances wrapped in DummyVecEnv.
    Assumes Dolphin instances are already running.

    Returns DummyVecEnv.
    """
    use_vision, use_memory = resolve_env_flags(args)
    disabled_heads = getattr(args, 'disabled_heads', ['menu'])

    def _make_env(env_idx: int):
        """Return a factory function for the given environment index."""
        # Determine which PPO agent owns this environment
        owner = 0
        if allocation_result and 'allocation' in allocation_result:
            for aid, insts in allocation_result['allocation'].items():
                if env_idx in insts:
                    owner = aid
                    break

        def _init():
            AgentContext.set_current_agent(owner)
            EnvContext.set_current_env(env_idx)
            env = MonsterHunterEnv(
                use_vision=use_vision,
                use_memory=use_memory,
                grayscale=args.grayscale,
                frame_stack=4,
                use_controller=True,
                controller_debug=False,
                use_advanced_rewards=True,
                save_state_slot=args.save_state,
                rt_vision=(args.rtvision and use_vision),
                rt_minimap=(args.rtminimap and args.rtvision and use_vision and use_memory and env_idx == 0),
                instance_id=env_idx,
                agent_id=owner,
                disabled_heads=disabled_heads,
            )
            # Set Dolphin base dir for cross-user save state recovery
            try:
                from train.dolphin import resolve_dolphin_path
                env.dolphin_base_dir = resolve_dolphin_path(args.dolphin_path)
            except Exception:
                pass  # Non-critical — recovery will just be unavailable
            return env
        return _init

    logger.info(f"Creating {args.num_instances} environments (threaded init)...")

    pre_created = [None] * args.num_instances
    init_errors = [None] * args.num_instances

    def _parallel_init(idx):
        try:
            # Resolve the owning agent for this env index
            owner = 0
            if allocation_result and 'allocation' in allocation_result:
                for aid, insts in allocation_result['allocation'].items():
                    if idx in insts:
                        owner = aid
                        break
            AgentContext.set_current_agent(owner)
            pre_created[idx] = _make_env(idx)()
        except Exception as exc:
            init_errors[idx] = exc

    threads = [threading.Thread(target=_parallel_init, args=(i,), daemon=False)
               for i in range(args.num_instances)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for idx, err in enumerate(init_errors):
        if err is not None:
            raise RuntimeError(f"Failed to initialize environment {idx}: {err}")

    # Surveillance window for real-time vision in multi-instance mode
    if args.rtvision and use_vision:
        _rt_display_module._surveillance_win = SurveillanceWindow(
            num_agents=args.num_instances,
            allocation=allocation_result.get('allocation') if allocation_result else None,
        )
        logger.info(f"Surveillance window initialized ({args.num_instances} agents)")

    vec_env = DummyVecEnv([lambda e=pre_created[i]: e for i in range(args.num_instances)])

    # Flag all environments as multi-instance for reduced action frames
    if args.num_instances > 1:
        try:
            vec_env.env_method('__setattr__', '_multi_instance_mode', True)
        except Exception:
            pass

    logger.info(f"{args.num_instances} environments created (DummyVecEnv)")

    if args.num_instances > 3:
        logger.info(
            f"DummyVecEnv steps {args.num_instances} envs sequentially per agent step. "
            f"This is normal - each env targets a separate Dolphin instance."
        )

    return vec_env


# ======================================================================
#  VECNORMALIZE WRAPPER
# ======================================================================

def wrap_vec_normalize(env, args, logs_dir, training_logger=None):
    """Apply VecNormalize. Loads from checkpoint when resuming."""
    vec_norm_path = os.path.join(logs_dir, "vec_normalize.pkl")

    if args.resume and os.path.exists(vec_norm_path) and not args.force_new_vecnormalize:
        logger.info("Loading VecNormalize from checkpoint...")
        try:
            env = VecNormalize.load(vec_norm_path, env)
            logger.info("VecNormalize loaded successfully")
            return env
        except Exception as exc:
            if training_logger:
                training_logger.log_error(exc, context="VecNormalize load")
            logger.error(f"VecNormalize load failed: {exc} — creating new wrapper")

    if args.force_new_vecnormalize:
        logger.info("--force-new-vecnormalize: creating fresh VecNormalize")

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=100.0,
        gamma=0.997,
    )
    logger.info("VecNormalize wrapper created")
    return env


# ======================================================================
#  VALIDATION & TESTS
# ======================================================================

def validate_multi_instance_windows(env, num_instances: int) -> bool:
    """
    Verify that each environment captured the correct MHTri-N window.
    Returns True if no collision was detected.
    """
    try:
        titles = env.env_method('get_window_title')
        collision = False
        for i, title in enumerate(titles):
            expected = f"MHTri-{i}"
            ok = expected in title
            logger.info(f"  Instance {i}: '{title}' [{'OK' if ok else 'MISMATCH'}]")
            if not ok:
                collision = True
        return not collision
    except Exception as exc:
        logger.error(f"Window validation error: {exc}")
        return True  # non-fatal — proceed anyway


def run_vision_test(base_env, training_logger=None):
    """Capture a test frame and save a crop-verification image."""
    if not base_env.use_vision:
        return
    logger.info("Testing vision pipeline...")
    try:
        frame = base_env.frame_capture.capture_frame()
        processed = base_env.preprocessor.preprocess_frame(frame)
        stacked = base_env.preprocessor.process_and_stack(frame)
        logger.debug(f"  Raw: {frame.shape}, Processed: {processed.shape}, Stacked: {stacked.shape}")

        debug_dir = os.path.join(".", "vision", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        base_env.preprocessor.visualize_crop(
            frame, os.path.join(debug_dir, "crop_verification_training.png"))
    except Exception as exc:
        if training_logger:
            training_logger.log_error(exc, context="Vision test")
        logger.error(f"Vision test error: {exc}")


def run_exploration_map_test(base_env, training_logger=None):
    """Generate and save a test minimap image."""
    if not (base_env.use_memory and base_env.state_fusion and base_env.memory):
        return
    logger.info("Testing exploration map...")
    try:
        state = base_env.memory.read_game_state()
        x, y, z = state.get('player_x', 0.0), state.get('player_y', 0.0), state.get('player_z', 0.0)
        zone = state.get('current_zone', 0) or 0

        test_map = base_env.state_fusion.create_exploration_map_with_channels((x, y, z), zone)
        logger.debug(f"  Minimap shape: {test_map.shape}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(test_map[:, :, 0], cmap='viridis'); axes[0].set_title("Visits")
        axes[1].imshow(test_map[:, :, 1], cmap='hot');     axes[1].set_title("Player position")
        axes[2].imshow(test_map[:, :, 2], cmap='Blues');    axes[2].set_title("Recent cubes")
        plt.tight_layout()

        debug_dir = os.path.join(".", "vision", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        plt.savefig(os.path.join(debug_dir, "minimap_test.png"), dpi=100)
        plt.close()
    except Exception as exc:
        if training_logger:
            training_logger.log_error(exc, context="Exploration map test")
        logger.error(f"Exploration map test error: {exc}")