"""
Genetic Algorithm Trainer pour PPO Multi-Agent
Implémente évolution génétique selon spec v1.0
"""

import numpy as np
import copy
import torch
from typing import List, Dict, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from utils.module_logger import get_module_logger

logger = get_module_logger('genetic_trainer')


class GeneticTrainer:
    """
    Entraîneur génétique pour multi-agents PPO

    Processus :
    1. Évaluation : Chaque agent joue N épisodes
    2. Sélection : Conservation élites + choix parents
    3. Reproduction : Mutations + Croisements
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
            agents: Liste agents PPO
            env: Environnement vectorisé
            elite_ratio: Ratio d'élites conservées
            mutation_rate: Taux de mutation
            episodes_per_eval: Episodes par évaluation
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
        Évalue un agent sur N épisodes

        Returns:
            Score moyen (fitness)
        """
        episode_rewards = []

        for episode in range(self.episodes_per_eval):
            obs = self.env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action, _ = agent.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward[0]  # VecEnv retourne array
                done = terminated[0] or truncated[0]

            episode_rewards.append(episode_reward)

        fitness = np.mean(episode_rewards)
        logger.info(f"Agent {agent_id} : Fitness = {fitness:.2f} (±{np.std(episode_rewards):.2f})")

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

    def mutate_agent(self, agent: PPO, mutation_strength: float = 0.1) -> PPO:
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

    def crossover_agents(self, parent1: PPO, parent2: PPO) -> PPO:
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
        Boucle d'entraînement génétique

        Args:
            num_generations: Nombre de générations
            progress_bar: Afficher progression
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("DÉMARRAGE ENTRAÎNEMENT GÉNÉTIQUE")
        logger.info("=" * 70)
        logger.info(f"Générations : {num_generations}")
        logger.info(f"Agents : {self.num_agents}")
        logger.info(f"Episodes/eval : {self.episodes_per_eval}")
        logger.info("")

        best_fitness_history = []
        mean_fitness_history = []

        for generation in range(num_generations):
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"GÉNÉRATION {generation + 1}/{num_generations}")
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