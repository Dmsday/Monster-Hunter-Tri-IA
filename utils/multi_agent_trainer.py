"""
Trainer custom pour SCÉNARIO 3 : Multi-agents partageant des instances
Implémente la boucle collect/train avec scheduler
"""

import time
import torch
import numpy as np
from typing import List, Optional, Dict
from collections import defaultdict

from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from utils.multi_agent_scheduler import MultiAgentScheduler
from utils.module_logger import get_module_logger

logger = get_module_logger('multi_agent_trainer')


class MultiAgentTrainer:
    """
    Gère l'entraînement de plusieurs agents PPO partageant des instances

    Architecture :
    - N agents PPO (chacun avec son modèle)
    - M instances (M < N)
    - Scheduler pour gérer qui joue quand
    - Collection séquentielle par instance, parallèle entre instances
    """

    def __init__(
            self,
            agents: List[PPO],
            env: VecEnv,
            scheduler: MultiAgentScheduler,
            steps_per_agent: int = 2048,
            callback: Optional[BaseCallback] = None,
            scenario: str = "INSTANCE_SHARING",
            allocation: Dict[int, List[int]] = None,
    ):
        """
        Args:
            agents: Liste des agents PPO
            env: Environnement vectorisé
            scheduler: Scheduler pour gérer les tours
            steps_per_agent: Steps à collecter par agent avant update
            callback: Callback SB3 (optionnel)
            scenario: Type de scénario (pour weighted rebalancing)
            allocation: Allocation initiale (pour référence)
        """
        self.agents = agents
        self.env = env
        self.scheduler = scheduler
        self.steps_per_agent = steps_per_agent
        self.callback = callback
        self.scenario = scenario
        self.allocation = allocation

        # Stats
        self.total_timesteps = 0
        self.episode_counts = [0] * len(agents)
        self.episode_rewards = [[] for _ in range(len(agents))]

        logger.info("MultiAgentTrainer créé")
        logger.info(f"   Agents : {len(agents)}")
        logger.info(f"   Instances : {env.num_envs}")
        logger.info(f"   Steps/agent : {steps_per_agent}")
        logger.info(f"   Mode : {scheduler.mode}")
        logger.info(f"   Scénario : {scenario}")

    @staticmethod
    def _obs_to_tensor(obs, device):
        """
        Convertit observation en tensor pour le policy

        Args:
            obs: Observation (dict ou array)
            device: Device torch

        Returns:
            Tensor ou dict de tensors
        """
        if isinstance(obs, dict):
            obs_tensor = {}
            for key, value in obs.items():
                obs_tensor[key] = torch.as_tensor(value).unsqueeze(0).to(device)
            return obs_tensor
        else:
            return torch.as_tensor(obs).unsqueeze(0).to(device)

    def train(self, total_timesteps: int, progress_bar: bool = True):
        """
        Main training loop

        Args:
            total_timesteps: Total steps to perform (distributed across agents)
            progress_bar: Display progress bar (currently unused, reserved for future tqdm integration)
        """
        # Note: progress_bar parameter reserved for future progress tracking implementation
        # Currently logging is used instead of progress bars

        logger.info("")
        logger.info("=" * 70)
        logger.info("🚀 STARTING MULTI-AGENT TRAINING")
        logger.info("=" * 70)
        logger.info(f"Total timesteps visés : {total_timesteps:,}")
        logger.info(f"Timesteps par agent : {total_timesteps // len(self.agents):,}")
        logger.info(f"Mode : {self.scheduler.mode}")
        logger.info("")

        # Reset environnement
        observations = self.env.reset()

        # Compteurs
        steps_collected = defaultdict(int)  # Steps par agent
        episodes_done = defaultdict(int)  # Épisodes par agent
        start_time = time.time()
        last_log_time = start_time

        # Buffers temporaires pour stocker les expériences
        # Note : En PPO, les buffers sont gérés par l'agent lui-même
        # On utilise donc les buffers natifs de SB3

        # Boucle principale
        while self.total_timesteps < total_timesteps:
            # ================================================================
            # PHASE 1 : COLLECTION
            # ================================================================
            logger.debug("Phase de collection...")

            for collect_step in range(self.steps_per_agent):
                # Pour chaque instance
                actions = []
                agents_used = []

                for env_idx in range(self.env.num_envs):
                    obs = observations[env_idx]

                    # Demander action au scheduler
                    action, agent_used = self.scheduler.get_action(env_idx, obs)

                    actions.append(action)
                    agents_used.append(agent_used)

                # Exécuter actions
                new_observations, rewards, dones, infos = self.env.step(np.array(actions))

                # STOCKER dans le buffer de chaque agent
                for env_idx in range(self.env.num_envs):
                    agent_used = agents_used[env_idx]

                    if agent_used >= 0:
                        # Agent spécifique
                        agent = self.agents[agent_used]

                        # Predict value and log_prob (keep as tensors for buffer)
                        with torch.no_grad():
                            obs_tensor = self._obs_to_tensor(observations[env_idx], agent.device)
                            action_tensor = torch.as_tensor([actions[env_idx]]).to(agent.device)

                            # Evaluate action - returns tensors
                            values, log_prob, _ = agent.policy.evaluate_actions(obs_tensor, action_tensor)

                            # Keep as tensors but flatten (buffer expects 1D tensors)
                            values = values.flatten()  # Remove .cpu().numpy()
                            log_prob = log_prob.flatten()  # Remove .cpu().numpy()

                        # Add to buffer (SB3 expects tensors, not numpy)
                        agent.rollout_buffer.add(
                            obs=observations[env_idx],
                            action=np.array([actions[env_idx]]),
                            reward=np.array([rewards[env_idx]]),
                            episode_start=np.array([dones[env_idx]]),
                            value=values,  # Now a tensor
                            log_prob=log_prob  # Now a tensor
                        )

                        steps_collected[agent_used] += 1

                        # Compter épisodes terminés
                        if dones[env_idx]:
                            episodes_done[agent_used] += 1

                            # Mettre à jour score pour weighted mode
                            if 'episode' in infos[env_idx]:
                                episode_reward = infos[env_idx]['episode']['r']
                                self.scheduler.update_agent_score(agent_used, episode_reward)

                    else:
                        # Mode majority_vote : tous les agents
                        for aid in range(len(self.agents)):
                            agent = self.agents[aid]

                            # Each agent stores the transition (keep tensors for buffer)
                            with torch.no_grad():
                                obs_tensor = self._obs_to_tensor(observations[env_idx], agent.device)
                                action_tensor = torch.as_tensor([actions[env_idx]]).to(agent.device)

                                values, log_prob, _ = agent.policy.evaluate_actions(obs_tensor, action_tensor)

                                # Keep as tensors, just flatten
                                values = values.flatten()  # Remove .cpu().numpy()
                                log_prob = log_prob.flatten()  # Remove .cpu().numpy()

                            agent.rollout_buffer.add(
                                obs=observations[env_idx],
                                action=np.array([actions[env_idx]]),
                                reward=np.array([rewards[env_idx]]),
                                episode_start=np.array([dones[env_idx]]),
                                value=values,  # Now a tensor
                                log_prob=log_prob  # Now a tensor
                            )

                            steps_collected[aid] += 1

                            if dones[env_idx]:
                                episodes_done[aid] += 1
                                if 'episode' in infos[env_idx]:
                                    episode_reward = infos[env_idx]['episode']['r']
                                    self.scheduler.update_agent_score(aid, episode_reward)

                # Update observations
                observations = new_observations

                # Callback
                if self.callback:
                    if not self.callback.on_step():
                        logger.warning("Callback a demandé l'arrêt")
                        return

                self.total_timesteps += self.env.num_envs

                # Periodic logging (replaces progress bar for now)
                current_time = time.time()
                if current_time - last_log_time > 10.0:
                    elapsed = current_time - start_time
                    fps = self.total_timesteps / elapsed

                    # Calculate progress percentage
                    progress_pct = (self.total_timesteps / total_timesteps) * 100

                    # Simple text-based progress indicator (until tqdm integration)
                    if progress_bar:
                        # Show progress bar in logs
                        bar_length = 30
                        filled = int(bar_length * self.total_timesteps / total_timesteps)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        logger.info(f"[{bar}] {progress_pct:.1f}%")

                    logger.info("")
                    logger.info(f"📊 Progress: {self.total_timesteps:,}/{total_timesteps:,} steps")
                    logger.info(f"   FPS : {fps:.1f}")
                    logger.info(f"   Steps par agent : {dict(steps_collected)}")
                    logger.info(f"   Épisodes : {dict(episodes_done)}")

                    last_log_time = current_time

                # Vérifier limite
                if self.total_timesteps >= total_timesteps:
                    break

            if self.total_timesteps >= total_timesteps:
                break

            # ================================================================
            # PHASE 2 : UPDATE
            # ================================================================
            logger.info("Phase d'update des agents...")

            for agent_id, agent in enumerate(self.agents):
                if steps_collected[agent_id] < self.steps_per_agent:
                    logger.debug(f"Agent {agent_id} : Pas assez de steps ({steps_collected[agent_id]})")
                    continue

                logger.debug(f"Update agent {agent_id} ({steps_collected[agent_id]} steps)")

                try:
                    # Forcer update via rollout buffer
                    # Le buffer a déjà été rempli pendant collect_rollouts
                    # On doit juste déclencher l'optimization

                    # 1. Calculer advantages si pas fait
                    if not agent.rollout_buffer.full:
                        logger.warning(
                            f"Agent {agent_id} : Buffer non plein ({agent.rollout_buffer.pos}/{agent.rollout_buffer.buffer_size})")
                        continue

                    # 2. Compute returns and advantages
                    with torch.no_grad():
                        # Calculate last value for bootstrap
                        last_obs = agent.rollout_buffer.observations[-1]
                        if isinstance(last_obs, dict):
                            # Dict obs → convert to torch
                            last_obs_torch = {}
                            for key, value in last_obs.items():
                                last_obs_torch[key] = torch.as_tensor(value).unsqueeze(0).to(agent.device)
                        else:
                            last_obs_torch = torch.as_tensor(last_obs).unsqueeze(0).to(agent.device)

                        last_values = agent.policy.predict_values(last_obs_torch)

                        # Keep as tensor (compute_returns_and_advantage accepts both)
                        # But flatten to 1D
                        last_values = last_values.flatten()  # Remove .cpu().numpy()

                    # Calculate advantages with GAE (accepts tensors)
                    agent.rollout_buffer.compute_returns_and_advantage(
                        last_values=last_values,  # Now a tensor
                        dones=agent.rollout_buffer.episode_starts[-1]
                    )

                    # 3. Lancer les epochs d'optimisation
                    for epoch in range(agent.n_epochs):
                        # Itérer sur les mini-batches
                        for rollout_data in agent.rollout_buffer.get(agent.batch_size):
                            actions = rollout_data.actions
                            if isinstance(agent.action_space, spaces.Discrete):
                                actions = actions.long().flatten()

                            # Forward pass
                            values, log_prob, entropy = agent.policy.evaluate_actions(
                                rollout_data.observations,
                                actions
                            )
                            values = values.flatten()

                            # Normalize advantages
                            advantages = rollout_data.advantages
                            if agent.normalize_advantage:
                                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                            # Policy loss (clipped)
                            ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                            policy_loss_1 = advantages * ratio
                            policy_loss_2 = advantages * torch.clamp(
                                ratio, 1 - agent.clip_range, 1 + agent.clip_range
                            )
                            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                            # Value loss (clipped)
                            if agent.clip_range_vf is not None:
                                values_pred = rollout_data.old_values + torch.clamp(
                                    values - rollout_data.old_values,
                                    -agent.clip_range_vf,
                                    agent.clip_range_vf
                                )
                                value_loss = (rollout_data.returns - values_pred).pow(2)
                            else:
                                value_loss = (rollout_data.returns - values).pow(2)
                            value_loss = value_loss.mean()

                            # Entropy loss
                            if entropy is None:
                                entropy_loss = -torch.mean(-log_prob)
                            else:
                                entropy_loss = -torch.mean(entropy)

                            # Total loss
                            loss = (
                                    policy_loss
                                    + agent.ent_coef * entropy_loss
                                    + agent.vf_coef * value_loss
                            )

                            # Optimization step
                            agent.policy.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                agent.policy.parameters(),
                                agent.max_grad_norm
                            )
                            agent.policy.optimizer.step()

                    # 4. Reset buffer
                    agent.rollout_buffer.reset()

                    logger.debug(f"   Agent {agent_id} : Update terminé")

                except Exception as agent_train_error:
                    logger.error(f"   Agent {agent_id} : Erreur update : {agent_train_error}")
                    import traceback
                    traceback.print_exc()

            # Rééquilibrage allocation (mode weighted)
            if self.scheduler.mode == 'weighted':
                total_episodes = sum(len(scores) for scores in self.scheduler.agent_scores.values())

                if total_episodes >= self.scheduler.weighted_eval_freq:
                    logger.info("Rééquilibrage allocation (mode weighted)...")

                    rebalanced = self.scheduler.rebalance_weighted_allocation(
                        scenario=self.scenario
                    )

                    if rebalanced:
                        logger.info("Allocation mise à jour selon performances")

                # Reset compteurs pour prochain cycle
            steps_collected = defaultdict(int)

            # Vérifier limite
            if self.total_timesteps >= total_timesteps:
                break

        # ================================================================
        # FIN
        # ================================================================
        total_time = time.time() - start_time

        logger.info("")
        logger.info("=" * 70)
        logger.info("ENTRAÎNEMENT TERMINÉ")
        logger.info("=" * 70)
        logger.info(f"Total timesteps : {self.total_timesteps:,}")
        logger.info(f"Temps écoulé : {total_time:.1f}s")
        logger.info(f"FPS moyen : {self.total_timesteps / total_time:.1f}")
        logger.info("")
        logger.info("Steps par agent :")

        # Afficher stats finales par agent
        for agent_id in range(len(self.agents)):
            total_steps_agent = self.agents[agent_id].num_timesteps
            logger.info(f"   Agent {agent_id} : {total_steps_agent:,} steps")

        logger.info("=" * 70)
        logger.info("")