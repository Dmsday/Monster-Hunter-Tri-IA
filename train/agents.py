"""
agents.py — PPO agent creation and checkpoint loading.

Exports:
    load_or_create_single_agent() -> (PPO, int)
    create_multi_agents()         -> list[PPO]
    SingleEnvProxy                (gym.Env stub for multi-agent PPO init)
"""

import os
import sys
import time

import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from agent.ppo_agent import create_ppo_agent
from info.module_logger import get_module_logger

logger = get_module_logger('train.agents')


# ======================================================================
#  SINGLE AGENT
# ======================================================================

def load_or_create_single_agent(args, env, device, training_logger=None, base_env=None):
    """
    Load a PPO agent from --resume or create a fresh one.

    Returns:
        (agent: PPO, previous_timesteps: int)
    """
    if args.resume:
        agent, prev = _load_model(args.resume, env, device, training_logger)
        if agent is not None:
            return agent, prev

    # Create a new agent from scratch
    logger.info("Creating new PPO agent...")
    n_steps = 256 if args.small_rollout else 512
    batch_size = 64 if args.small_rollout else 256

    agent = create_ppo_agent(
        environment_new=env,
        learning_rate=args.lr,
        n_steps=n_steps,
        batch_size=batch_size,
        features_dim=256,
        device=device,
        verbose=1,
        tensorboard_log=f"./logs/{args.name}",
    )

    logger.info("PPO agent created")

    # Run a quick CNN forward-pass test if vision is enabled
    if base_env is not None and getattr(base_env, 'use_vision', False):
        _test_cnn_features(agent, base_env, device, training_logger)

    return agent, 0


def _load_model(path, env, device, training_logger=None):
    """
    Attempt to load a PPO checkpoint.

    Returns (agent, previous_timesteps) on success, or exits on failure.
    """
    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        if training_logger:
            training_logger.log_error(FileNotFoundError(path), context="Model loading")
        sys.exit(1)

    logger.info(f"Loading model: {path}")
    time.sleep(2.0)

    try:
        agent = PPO.load(path, env=env, device=device)
        logger.info(f"Model loaded successfully ({agent.num_timesteps:,} previous timesteps)")
        return agent, agent.num_timesteps
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        if training_logger:
            training_logger.log_error(exc, context="Checkpoint loading")
        logger.error(f"Model loading error: {exc}")
        sys.exit(1)


def _test_cnn_features(agent, base_env, device, training_logger=None):
    """Run a single forward pass through the CNN feature extractor to verify it works."""
    if not isinstance(base_env.observation_space, spaces.Dict):
        return

    logger.info("Testing CNN feature extraction...")
    try:
        with torch.no_grad():
            test_obs = base_env.reset()[0]
            obs_tensor = {}
            for key, value in test_obs.items():
                if isinstance(value, dict):
                    obs_tensor[key] = {
                        k: torch.FloatTensor(v).unsqueeze(0).to(device)
                        for k, v in value.items()
                    }
                else:
                    obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(device)

            features = agent.policy.extract_features(obs_tensor, agent.policy.features_extractor)
            logger.debug(f"  CNN output shape: {features.shape}")
    except Exception as exc:
        if training_logger:
            training_logger.log_error(exc, context="CNN feature test")
        logger.error(f"CNN feature test error: {exc}")


# ======================================================================
#  MULTI AGENT
# ======================================================================

class SingleEnvProxy(gym.Env):
    """
    Minimal gym.Env proxy so SB3's _patch_env() accepts it.

    PPO only reads observation_space and action_space at init time to size
    the policy network and rollout buffer. The actual rollout collection is
    driven by MultiAgentTrainer, so reset() / step() are never called.
    """

    def __init__(self, full_env):
        super().__init__()
        self.observation_space = full_env.observation_space
        self.action_space = full_env.action_space
        self.num_envs = 1

    def reset(self, seed=None, options=None):
        return None, {}

    def step(self, action):
        return None, 0.0, False, False, {}

    def render(self):
        pass


# ✅ MODIFIÉ
def create_multi_agents(args, env, device, logs_dir) -> list:
    """
    Create *args.num_agents* independent PPO agents.

    If --resume points to a directory containing agent_N/final_model.zip or
    interrupted_agent_N_*steps.zip files, loads them and fills remaining
    slots with fresh agents.

    Handles all cases:
        - N saved == N requested  → 1:1 load
        - N saved <  N requested  → load all saved, create fresh for the rest
        - N saved >  N requested  → pick the N with the most training steps

    Returns a list of PPO instances.
    """
    n_steps = 512 if args.small_rollout else 2048
    batch_size = 64 if args.small_rollout else 256
    proxy = SingleEnvProxy(env)

    # ------------------------------------------------------------------
    # RESUME PATH: discover saved models
    # ------------------------------------------------------------------
    if args.resume:
        saved_models = _discover_saved_models(args.resume)

        if not saved_models:
            logger.warning(
                f"No agent models found in '{args.resume}' "
                f"— creating {args.num_agents} fresh agents"
            )
        else:
            return _load_multi_agents(
                saved_models=saved_models,
                num_agents=args.num_agents,
                proxy=proxy,
                lr=args.lr,
                n_steps=n_steps,
                batch_size=batch_size,
                device=device,
                logs_dir=logs_dir,
            )

    # ------------------------------------------------------------------
    # FRESH: create all agents from scratch
    # ------------------------------------------------------------------
    logger.info(f"Creating {args.num_agents} fresh PPO agents...")
    agents = []

    for aid in range(args.num_agents):
        agent = create_ppo_agent(
            environment_new=proxy,
            learning_rate=args.lr,
            n_steps=n_steps,
            batch_size=batch_size,
            features_dim=256,
            device=device,
            verbose=0,
            tensorboard_log=os.path.join(logs_dir, f"agent_{aid}"),
        )
        agents.append(agent)
        logger.info(f"  Agent {aid + 1}/{args.num_agents} created (fresh)")

    logger.info(f"All {len(agents)} agents ready")
    return agents


def _discover_saved_models(resume_path: str) -> list:
    """
    Scan a directory (or single file) for saved PPO agent .zip files.

    Looks for:
        1. agent_N/final_model.zip      (normal end)
        2. *_agent_N_*steps.zip         (interrupted / checkpoint / error)
        3. Single .zip file             (--resume path/to/model.zip)

    Returns:
        List of (agent_id, path, num_steps) sorted by agent_id.
        Empty list if nothing found.
    """
    import re

    # Case 1: resume path is a single .zip file (e.g. from single-agent)
    if os.path.isfile(resume_path):
        logger.info(f"Resume from single model file: {resume_path}")
        return [(0, resume_path, 0)]

    if not os.path.isdir(resume_path):
        # Maybe they forgot .zip extension
        if os.path.isfile(resume_path + '.zip'):
            return [(0, resume_path + '.zip', 0)]
        logger.warning(f"Resume path not found: {resume_path}")
        return []

    found = {}  # agent_id -> (path, steps)

    # Scan agent_N/ subdirectories for final_model.zip
    for entry in os.listdir(resume_path):
        subdir = os.path.join(resume_path, entry)
        if os.path.isdir(subdir) and entry.startswith('agent_'):
            try:
                aid = int(entry.split('_')[1])
            except (IndexError, ValueError):
                continue

            final = os.path.join(subdir, 'final_model.zip')
            if os.path.isfile(final):
                # Read num_timesteps from the model to rank them
                steps = _read_model_steps(final)
                found[aid] = (final, steps)
                continue

            # Look for checkpoint/interrupted files in subdir
            best = _find_best_zip_in_dir(subdir)
            if best:
                found[aid] = best

    # Scan root directory for interrupted_agent_N_XXXsteps.zip or similar
    pattern = re.compile(r'(?:interrupted|checkpoint|error)_agent_(\d+)_(\d+)steps\.zip')
    for fname in os.listdir(resume_path):
        m = pattern.match(fname)
        if m:
            aid = int(m.group(1))
            steps = int(m.group(2))
            fpath = os.path.join(resume_path, fname)
            # Only use if we don't already have a final_model for this agent
            if aid not in found or steps > found[aid][1]:
                found[aid] = (fpath, steps)

    result = [(aid, path, steps) for aid, (path, steps) in found.items()]
    result.sort(key=lambda x: x[0])  # Sort by agent_id

    if result:
        logger.info(f"Discovered {len(result)} saved model(s):")
        for aid, path, steps in result:
            logger.info(f"  Agent {aid}: {os.path.basename(path)} ({steps:,} steps)")

    return result


def _find_best_zip_in_dir(directory: str):
    """Find the .zip with the most steps in a directory. Returns (path, steps) or None."""
    import re
    best_path, best_steps = None, -1
    pattern = re.compile(r'(\d+)\s*steps')

    for fname in os.listdir(directory):
        if not fname.endswith('.zip'):
            continue
        fpath = os.path.join(directory, fname)
        m = pattern.search(fname)
        if m:
            steps = int(m.group(1))
        else:
            steps = _read_model_steps(fpath)
        if steps > best_steps:
            best_steps = steps
            best_path = fpath

    return (best_path, max(best_steps, 0)) if best_path else None


def _read_model_steps(zip_path: str) -> int:
    """Quick-read num_timesteps from a saved PPO model without fully loading it."""
    try:
        import zipfile, io, torch as _torch
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # SB3 stores data in 'data' file inside the zip
            if 'data' in zf.namelist():
                raw = zf.read('data')
                data = _torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
                return int(data.get('num_timesteps', 0))
    except Exception:
        pass
    return 0

def _load_multi_agents(
    saved_models: list,
    num_agents: int,
    proxy,
    lr: float,
    n_steps: int,
    batch_size: int,
    device: str,
    logs_dir: str,
) -> list:
    """
    Load saved models and adapt to the requested agent count.

    Strategy:
        - N saved == N agents  → load 1:1
        - N saved <  N agents  → load all, CLONE the best to fill remaining
        - N saved >  N agents  → recursively MERGE the two weakest until
                                  count matches (keeps originals on disk)
    """
    import copy as _copy

    n_saved = len(saved_models)
    agents = []

    # ------------------------------------------------------------------
    # EXACT MATCH
    # ------------------------------------------------------------------
    if n_saved == num_agents:
        logger.info(f"Loading {num_agents} agents (exact match)")
        for aid, (_, path, steps) in enumerate(saved_models):
            agent = _load_single_agent(path, proxy, device)
            agent.learning_rate = lr
            agents.append(agent)
            logger.info(f"  Agent {aid}: loaded from {os.path.basename(path)} ({steps:,} steps)")

    # ------------------------------------------------------------------
    # FEWER SAVED THAN NEEDED → clone best performers
    # ------------------------------------------------------------------
    elif n_saved < num_agents:
        n_clones = num_agents - n_saved

        # Rank saved models by reward (estimated from steps as proxy,
        # or use actual reward if available via session_summary)
        ranked = _rank_by_reward(saved_models)

        logger.info(
            f"Loading {n_saved} saved agent(s), cloning top performers "
            f"to fill {n_clones} extra slot(s)"
        )

        # Load all saved models
        loaded = []
        for aid, (orig_aid, path, steps) in enumerate(saved_models):
            agent = _load_single_agent(path, proxy, device)
            agent.learning_rate = lr
            agents.append(agent)
            loaded.append((aid, agent, steps))
            logger.info(f"  Agent {aid}: loaded ({steps:,} steps)")

        # Clone from the best models in round-robin order
        for i in range(n_clones):
            # Pick source: cycle through top performers
            source_idx = ranked[i % len(ranked)]
            source_agent = agents[source_idx]
            cloned = _clone_agent(source_agent, proxy, device)
            cloned.learning_rate = lr
            new_aid = n_saved + i
            agents.append(cloned)
            logger.info(
                f"  Agent {new_aid}: CLONED from agent {source_idx} "
                f"(inherits trained weights)"
            )

    # ------------------------------------------------------------------
    # MORE SAVED THAN NEEDED → recursive merge of weakest pairs
    # ------------------------------------------------------------------
    else:
        logger.info(
            f"Found {n_saved} saved models but only need {num_agents} "
            f"— merging weakest pairs recursively"
        )

        # Load all models with their reward ranking
        pool = []  # [(agent, orig_id, reward_rank)]
        ranked = _rank_by_reward(saved_models)

        for orig_aid, path, steps in saved_models:
            agent = _load_single_agent(path, proxy, device)
            agent.learning_rate = lr
            rank_pos = ranked.index(orig_aid) if orig_aid in ranked else len(ranked)
            pool.append({
                'agent': agent,
                'orig_ids': [orig_aid],
                'rank': rank_pos,
                'steps': steps,
            })

        # Recursive merge: always merge the two worst until count matches
        merge_round = 0
        while len(pool) > num_agents:
            merge_round += 1
            # Sort by rank descending (worst = highest rank number = end)
            pool.sort(key=lambda x: x['rank'], reverse=True)

            worst_a = pool.pop(0)  # Worst
            worst_b = pool.pop(0)  # Second worst

            merged = _merge_agents(worst_a['agent'], worst_b['agent'])
            merged.learning_rate = lr

            merged_ids = worst_a['orig_ids'] + worst_b['orig_ids']
            # Merged agent gets the better rank of the two parents
            merged_rank = min(worst_a['rank'], worst_b['rank'])

            pool.append({
                'agent': merged,
                'orig_ids': merged_ids,
                'rank': merged_rank,
                'steps': max(worst_a['steps'], worst_b['steps']),
            })

            logger.info(
                f"  Merge #{merge_round}: agents {worst_a['orig_ids']} + "
                f"{worst_b['orig_ids']} → merged agent "
                f"(averaged weights)"
            )

        # Sort by rank (best first) and assign final IDs
        pool.sort(key=lambda x: x['rank'])
        for new_aid, entry in enumerate(pool):
            agents.append(entry['agent'])
            origin = entry['orig_ids']
            label = f"original {origin[0]}" if len(origin) == 1 else f"merge of {origin}"
            logger.info(f"  Agent {new_aid}: {label}")

    logger.info(
        f"All {len(agents)} agents ready "
        f"({min(n_saved, num_agents)} loaded"
        f"{f', {num_agents - n_saved} cloned' if n_saved < num_agents else ''}"
        f"{f', {n_saved - num_agents} merged' if n_saved > num_agents else ''})"
    )
    return agents


def _rank_by_reward(saved_models: list) -> list:
    """
    Rank saved models by reward from session_summary.json if available,
    otherwise by steps. Returns list of agent_ids ordered best-first.
    """
    scores = {}

    for orig_aid, path, steps in saved_models:
        # Try to find session_summary.json next to the model
        model_dir = os.path.dirname(path)
        summary_path = os.path.join(model_dir, 'session_summary.json')

        if os.path.isfile(summary_path):
            try:
                import json
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                ep_stats = summary.get('episode_statistics', {})
                reward_mean = ep_stats.get('reward', {}).get('mean', 0.0)
                scores[orig_aid] = reward_mean
                continue
            except Exception:
                pass

        # Fallback: use steps as proxy (more steps = probably better)
        scores[orig_aid] = steps

    # Sort by score descending (best first), return agent_ids
    ranked_ids = sorted(scores, key=lambda aid: scores[aid], reverse=True)

    if any(isinstance(v, float) for v in scores.values()):
        logger.info("Agent ranking (by reward):")
    else:
        logger.info("Agent ranking (by steps, no reward data found):")

    for i, aid in enumerate(ranked_ids):
        logger.info(f"  #{i+1}: Agent {aid} (score: {scores[aid]:.1f})")

    return ranked_ids


def _clone_agent(source: 'PPO', proxy, device: str) -> 'PPO':
    """
    Deep-clone a PPO agent: copy all network weights into a fresh instance.
    The clone shares the same architecture but is a fully independent object.
    """
    import copy as _copy
    import io as _io

    # Save source to an in-memory buffer, then reload as a new instance
    buffer = _io.BytesIO()
    source.save(buffer)
    buffer.seek(0)
    cloned = PPO.load(buffer, env=proxy, device=device)
    return cloned


def _merge_agents(agent_a: 'PPO', agent_b: 'PPO') -> 'PPO':
    """
    Merge two PPO agents by averaging their policy network weights.
    Returns agent_a with averaged weights (agent_b is not modified).

    This preserves learned features from both agents while creating
    a single blended policy — similar to federated averaging.
    """
    import copy as _copy
    import torch as _torch

    state_a = agent_a.policy.state_dict()
    state_b = agent_b.policy.state_dict()

    averaged = _copy.deepcopy(state_a)
    for key in averaged:
        if key in state_b:
            averaged[key] = (
                state_a[key].float() + state_b[key].float()
            ).div(2.0).to(state_a[key].dtype)

    agent_a.policy.load_state_dict(averaged)

    logger.debug("Weight merge complete (50/50 average)")
    return agent_a

def _load_single_agent(path: str, proxy, device: str):
    """Load a single PPO agent, attaching it to the proxy env."""
    try:
        agent = PPO.load(path, env=proxy, device=device)
        return agent
    except Exception as exc:
        logger.error(f"Failed to load {path}: {exc}")
        raise RuntimeError(f"Cannot load agent from {path}: {exc}")
