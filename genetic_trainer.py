"""
Genetic Algorithm Trainer pour PPO Multi-Agent
Implémente évolution génétique selon spec v1.0
"""

import numpy as np
import copy
from tqdm import tqdm
import torch
from typing import List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from utils.module_logger import get_module_logger

logger = get_module_logger('genetic_trainer')


class GeneticTrainer:
    """
    Genetic trainer for multi-agent PPO

    Process:
        1. Evaluation: Each agent plays N episodes
        2. Selection: Keep elites + choose parents
        3. Reproduction: Mutations + Crossovers
    """


    def __init__(
            self,
            agents: List[PPO],
            env: VecEnv,
            elite_ratio: float = 0.25,
            mutation_rate: float = 0.3,
            episodes_per_eval: int = 10,
    ):
        """
        Args:
            agents: List of PPO agents
            env: Vectorized environment
            elite_ratio: Ratio of elites to keep
            mutation_rate: Mutation rate
            episodes_per_eval: Number of episodes per evaluation
        """
        self.agents = agents
        self.env = env
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.episodes_per_eval = episodes_per_eval

        self.num_agents = len(agents)
        self.num_elites = max(1, int(self.num_agents * elite_ratio))

        logger.info("GeneticTrainer initialisé")
        logger.info(f"   Agents : {self.num_agents}")
        logger.info(f"   Élites : {self.num_elites}")
        logger.info(f"   Mutation rate : {mutation_rate}")

    def evaluate_agent(self, agent: PPO, agent_id: int) -> float:
        """
        Evaluates agent over N episodes

        Returns:
            Average score (fitness) as Python float
        """
        episode_rewards = []

        for episode in range(self.episodes_per_eval):
            # Handle both tuple (obs, info) and legacy obs-only reset formats
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result  # New Gymnasium format
            else:
                obs = reset_result  # Legacy format

            done = False
            episode_reward = 0.0

            while not done:
                action, _ = agent.predict(obs, deterministic=False)

                # Robust step handling with explicit error reporting
                step_result = self.env.step(action)

                try:
                    if len(step_result) == 5:
                        # Gymnasium format: obs, reward, terminated, truncated, info
                        obs, reward, terminated, truncated, info = step_result
                    elif len(step_result) == 4:
                        # Legacy format: obs, reward, done, info
                        obs, reward, done_legacy, info = step_result
                        # Convert to Gymnasium format
                        terminated = done_legacy
                        truncated = np.array([False] * self.env.num_envs, dtype=bool)
                    else:
                        # Unexpected format = log error and raise
                        logger.error(f"Unexpected step() return format: {len(step_result)} values")
                        logger.error(f"Expected 4 (legacy) or 5 (Gymnasium) values")
                        logger.error(f"step_result types: {[type(x).__name__ for x in step_result]}")
                        raise ValueError(
                            f"env.step() returned {len(step_result)} values, expected 4 or 5. "
                            f"Check your VecEnv configuration."
                        )
                except ValueError as unpack_error:
                    # Explicit unpacking error handling
                    logger.error(f"Failed to unpack step() result: {unpack_error}")
                    logger.error(f"step_result length: {len(step_result)}")
                    logger.error(f"step_result content: {step_result}")
                    raise RuntimeError(
                        f"env.step() unpacking failed. "
                        f"Got {len(step_result)} values, expected 4 or 5."
                    ) from unpack_error

                episode_reward += reward[0]  # VecEnv returns array
                done = terminated[0] or truncated[0]

            episode_rewards.append(episode_reward)

        # Convert numpy.floating to native Python float for type safety
        fitness = float(np.mean(episode_rewards))
        logger.info(f"Agent {agent_id}: Fitness = {fitness:.2f} (±{np.std(episode_rewards):.2f})")

        return fitness

    def evaluate_generation(self) -> List[Tuple[int, float]]:
        """
        Évalue tous les agents

        Returns:
            Liste (agent_id, fitness) triée par fitness décroissant
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("ÉVALUATION GÉNÉRATION")
        logger.info("=" * 70)

        fitness_scores = []

        for agent_id, agent in enumerate(self.agents):
            logger.info(f"Évaluation Agent {agent_id}/{self.num_agents}...")
            fitness = self.evaluate_agent(agent, agent_id)
            fitness_scores.append((agent_id, fitness))

        # Trier par fitness décroissant
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info("")
        logger.info("Classement :")
        for rank, (agent_id, fitness) in enumerate(fitness_scores, 1):
            marker = "🏆" if rank <= self.num_elites else "  "
            logger.info(f"{marker} {rank}. Agent {agent_id} : {fitness:.2f}")

        logger.info("=" * 70)

        return fitness_scores

    @staticmethod
    def mutate_agent(agent: PPO, mutation_strength: float = 0.1) -> PPO:
        """
        Mute un agent (perturbation des poids)

        Args:
            agent: Agent source
            mutation_strength: Intensité mutation

        Returns:
            Nouvel agent muté
        """
        # Copier l'agent
        mutated = copy.deepcopy(agent)

        # Perturber les poids du policy network
        with torch.no_grad():
            for param in mutated.policy.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)

        return mutated

    @staticmethod
    def crossover_agents(parent1: PPO, parent2: PPO) -> PPO:
        """
        Croisement entre 2 agents (mélange poids)

        Args:
            parent1: Premier parent
            parent2: Second parent

        Returns:
            Nouvel agent enfant
        """
        # Copier parent1
        child = copy.deepcopy(parent1)

        # Mélanger les poids
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(
                    child.policy.parameters(),
                    parent1.policy.parameters(),
                    parent2.policy.parameters()
            ):
                if child_param.requires_grad:
                    # Moyenne pondérée aléatoire
                    alpha = np.random.uniform(0.3, 0.7)
                    child_param.data = alpha * p1_param.data + (1 - alpha) * p2_param.data

        return child

    def evolve_generation(self, fitness_scores: List[Tuple[int, float]]) -> List[PPO]:
        """
        Crée nouvelle génération via sélection/reproduction

        Args:
            fitness_scores: Scores triés

        Returns:
            Nouvelle liste d'agents
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("REPRODUCTION GÉNÉRATION")
        logger.info("=" * 70)

        new_agents = []

        # 1. CONSERVER ÉLITES
        elite_ids = [agent_id for agent_id, _ in fitness_scores[:self.num_elites]]
        logger.info(f"Conservation {self.num_elites} élites : {elite_ids}")

        for agent_id in elite_ids:
            new_agents.append(copy.deepcopy(self.agents[agent_id]))

        # 2. REMPLIR LE RESTE
        remaining = self.num_agents - self.num_elites
        logger.info(f"Génération {remaining} nouveaux agents...")

        for i in range(remaining):
            strategy = np.random.choice(['mutate', 'crossover'], p=[self.mutation_rate, 1 - self.mutation_rate])

            if strategy == 'mutate':
                # Mutation d'un élite
                parent_id = np.random.choice(elite_ids)
                logger.info(f"   [{i + 1}/{remaining}] Mutation de Agent {parent_id}")
                new_agent = self.mutate_agent(self.agents[parent_id])
            else:
                # Croisement entre 2 élites
                parent1_id, parent2_id = np.random.choice(elite_ids, size=2, replace=False)
                logger.info(f"   [{i + 1}/{remaining}] Croisement Agent {parent1_id} × Agent {parent2_id}")
                new_agent = self.crossover_agents(self.agents[parent1_id], self.agents[parent2_id])

            new_agents.append(new_agent)

        logger.info("=" * 70)

        return new_agents

    def train(self, num_generations: int, progress_bar: bool = True):
        """
        Genetic training loop

        Args:
            num_generations: Number of generations to train
            progress_bar: Display progress bar (if True, shows generation progress)
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("GENETIC TRAINING STARTED")
        logger.info("=" * 70)
        logger.info(f"Generations: {num_generations}")
        logger.info(f"Agents: {self.num_agents}")
        logger.info(f"Episodes/eval: {self.episodes_per_eval}")
        logger.info(f"Progress bar: {'Enabled' if progress_bar else 'Disabled'}")  # Log parameter
        logger.info("")

        best_fitness_history = []
        mean_fitness_history = []

        # Use progress_bar parameter for conditional display
        if progress_bar:
            try:
                generation_iterator = tqdm(
                    range(num_generations),
                    desc="Genetic Training",
                    unit="gen"
                )
            except ImportError:
                # tqdm not available - use plain range iterator
                logger.warning("tqdm not available - falling back to simple logging")
                generation_iterator = range(num_generations)
        else:
            generation_iterator = range(num_generations)

        for generation in generation_iterator:
            # Conditional logging based on progress_bar
            if not progress_bar:
                logger.info("")
                logger.info("=" * 70)
                logger.info(f"GENERATION {generation + 1}/{num_generations}")
                logger.info("=" * 70)

            # 1. ÉVALUATION
            fitness_scores = self.evaluate_generation()

            # Stats
            best_fitness = fitness_scores[0][1]
            mean_fitness = np.mean([f for _, f in fitness_scores])

            best_fitness_history.append(best_fitness)
            mean_fitness_history.append(mean_fitness)

            logger.info("")
            logger.info(f"Best fitness : {best_fitness:.2f}")
            logger.info(f"Mean fitness : {mean_fitness:.2f}")

            # 2. ÉVOLUTION (sauf dernière génération)
            if generation < num_generations - 1:
                self.agents = self.evolve_generation(fitness_scores)

        logger.info("")
        logger.info("=" * 70)
        logger.info("ENTRAÎNEMENT GÉNÉTIQUE TERMINÉ")
        logger.info("=" * 70)
        logger.info(f"Best final fitness : {best_fitness_history[-1]:.2f}")
        logger.info(f"Amélioration : {best_fitness_history[-1] - best_fitness_history[0]:+.2f}")
        logger.info("=" * 70)

        return {
            'best_fitness_history': best_fitness_history,
            'mean_fitness_history': mean_fitness_history,
            'final_agents': self.agents
        }