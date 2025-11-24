"""
Scheduler pour gérer plusieurs agents PPO partageant des instances
Implémente les modes : independent, round_robin, majority_vote, genetic
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

from stable_baselines3 import PPO

logger = logging.getLogger('multi_agent_scheduler')


class MultiAgentScheduler:
    """
    Gère l'ordonnancement des actions de plusieurs agents sur instances partagées

    Modes supportés :
    - independent : Tour par tour, 1 step à la fois
    - round_robin : Tour par tour par blocs de N steps
    - majority_vote : Vote démocratique à chaque step
    - weighted : Allocation adaptative selon performances (avec rééquilibrage périodique)

    Note: Le mode genetic n'utilise pas le scheduler (gestion directe par GeneticTrainer)
    """

    def __init__(
            self,
            agents: List[PPO],
            allocation: Dict[int, List[int]],
            mode: str = 'independent',
            block_size: int = 100,
            weighted_eval_freq: int = 100,
    ):
        """
        Args:
            agents: Liste des agents PPO
            allocation: Dict {agent_id: [instance_ids]}
            mode: Mode de scheduling ('independent', 'round_robin', 'majority_vote')
            block_size: Taille des blocs pour round_robin
            weighted_eval_freq: Fréquence réévaluation (mode weighted)
        """
        self.agents = agents
        self.allocation = allocation
        self.mode = mode
        self.block_size = block_size
        self.weighted_eval_freq = weighted_eval_freq

        # Inverser l'allocation : instance -> [agents]
        self.instance_to_agents = defaultdict(list)
        for agent_id, instances in allocation.items():
            for inst_id in instances:
                self.instance_to_agents[inst_id].append(agent_id)

        # État interne
        self.current_agent_per_instance = {
            inst_id: agents_list[0]
            for inst_id, agents_list in self.instance_to_agents.items()
        }
        self.step_count_per_instance = defaultdict(int)

        logger.info(f"MultiAgentScheduler créé (mode={mode})")
        logger.info(f"   Allocation inversée : {dict(self.instance_to_agents)}")

        # Stats pour mode weighted
        self.agent_scores = defaultdict(list)  # {agent_id: [scores]}
        self.agent_total_steps = defaultdict(int)
        self.last_weighted_eval = 0

        logger.info(f"MultiAgentScheduler créé (mode={mode})")
        logger.info(f"Allocation inversée : {dict(self.instance_to_agents)}")
        if mode == 'weighted':
            logger.info(f"Réévaluation tous les {weighted_eval_freq} episodes")

    def get_action(
            self,
            instance_id: int,
            observation: np.ndarray
    ) -> Tuple[int, int]:
        """
        Obtient l'action à exécuter sur une instance

        Args:
            instance_id: ID de l'instance
            observation: Observation actuelle

        Returns:
            tuple: (action, agent_id_used)
        """
        agents_on_instance = self.instance_to_agents[instance_id]

        if len(agents_on_instance) == 1:
            # Cas simple : 1 seul agent sur cette instance
            agent_id = agents_on_instance[0]
            action, _ = self.agents[agent_id].predict(observation, deterministic=False)
            return int(action), agent_id

        # Plusieurs agents : appliquer le mode de scheduling
        if self.mode == 'independent':
            return self._independent_mode(instance_id, observation, agents_on_instance)

        elif self.mode == 'round_robin':
            return self._round_robin_mode(instance_id, observation, agents_on_instance)

        elif self.mode == 'majority_vote':
            return self._majority_vote_mode(instance_id, observation, agents_on_instance)

        else:
            raise ValueError(f"Mode inconnu : {self.mode}")

    def _independent_mode(
            self,
            instance_id: int,
            observation: np.ndarray,
            agents_on_instance: List[int]
    ) -> Tuple[int, int]:
        """
        Mode independent : Tour par tour, 1 step
        """
        current_agent_id = self.current_agent_per_instance[instance_id]

        # Prédire action
        action, _ = self.agents[current_agent_id].predict(observation, deterministic=False)

        # Passer au prochain agent
        current_index = agents_on_instance.index(current_agent_id)
        next_index = (current_index + 1) % len(agents_on_instance)
        self.current_agent_per_instance[instance_id] = agents_on_instance[next_index]

        return int(action), current_agent_id

    def _round_robin_mode(
            self,
            instance_id: int,
            observation: np.ndarray,
            agents_on_instance: List[int]
    ) -> Tuple[int, int]:
        """
        Mode round_robin : Tour par tour par blocs
        """
        current_agent_id = self.current_agent_per_instance[instance_id]
        step_count = self.step_count_per_instance[instance_id]

        # Prédire action
        action, _ = self.agents[current_agent_id].predict(observation, deterministic=False)

        # Incrémenter compteur
        self.step_count_per_instance[instance_id] += 1

        # Changer d'agent si bloc terminé
        if (step_count + 1) % self.block_size == 0:
            current_index = agents_on_instance.index(current_agent_id)
            next_index = (current_index + 1) % len(agents_on_instance)
            self.current_agent_per_instance[instance_id] = agents_on_instance[next_index]
            logger.debug(f"Instance {instance_id} : Passage à l'agent {agents_on_instance[next_index]}")

        return int(action), current_agent_id

    def _majority_vote_mode(
            self,
            instance_id: int,
            observation: np.ndarray,
            agents_on_instance: List[int]
    ) -> Tuple[int, int]:
        """Mode majority_vote : Vote démocratique"""
        # Tous les agents prédisent
        votes = []
        for agent_id in agents_on_instance:
            action, _ = self.agents[agent_id].predict(observation, deterministic=False)
            votes.append(action)

        # Majorité simple
        action_counts = defaultdict(int)
        for action in votes:
            action_counts[action] += 1

        # Action majoritaire
        majority_action = max(action_counts.items(), key=lambda x: x[1])[0]

        # Retourner action + agent "virtuel" (tous apprennent)
        return majority_action, -1  # -1 = tous les agents

    def update_agent_score(self, agent_id: int, episode_reward: float):
        """
        Met à jour le score d'un agent (pour mode weighted)

        Args:
            agent_id: ID de l'agent
            episode_reward: Reward de l'épisode terminé
        """
        self.agent_scores[agent_id].append(episode_reward)

        # Garder seulement derniers 100 scores
        if len(self.agent_scores[agent_id]) > 100:
            self.agent_scores[agent_id] = self.agent_scores[agent_id][-100:]

    def rebalance_weighted_allocation(self, scenario: str) -> bool:
        """
        Rééquilibre l'allocation selon performances (mode weighted)

        Returns:
            True si rééquilibrage effectué
        """
        if self.mode != 'weighted':
            return False

            # Vérifier si assez d'épisodes
        total_episodes = sum(len(scores) for scores in self.agent_scores.values())

        if total_episodes - self.last_weighted_eval < self.weighted_eval_freq:
            return False

        logger.info("")
        logger.info("=" * 70)
        logger.info("RÉÉQUILIBRAGE ALLOCATION (MODE WEIGHTED)")
        logger.info("=" * 70)

        # Calculer scores moyens
        agent_avg_scores = {}
        for agent_id, scores in self.agent_scores.items():
            if scores:
                agent_avg_scores[agent_id] = np.mean(scores[-50:])
            else:
                agent_avg_scores[agent_id] = 0.0

        # Afficher scores
        logger.info("Scores moyens (50 derniers épisodes) :")
        for agent_id in sorted(agent_avg_scores.keys()):
            logger.info(f"  Agent {agent_id} : {agent_avg_scores[agent_id]:.2f}")

        # Trier agents par performance
        sorted_agents = sorted(
            agent_avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if scenario == "AGENT_MULTIPLE_INSTANCES":
            # Répartir instances selon performances
            total_instances = sum(len(insts) for insts in self.allocation.values())

            # Calculer poids normalisés
            total_score = sum(score for _, score in sorted_agents if score > 0)

            if total_score == 0:
                logger.warning("Tous les scores sont nuls - pas de rééquilibrage")
                return False

            new_allocation = {}
            instances_assigned = 0

            for agent_id, score in sorted_agents[:-1]:
                weight = score / total_score
                num_instances = max(1, int(weight * total_instances))

                new_allocation[agent_id] = list(range(
                    instances_assigned,
                    instances_assigned + num_instances
                ))
                instances_assigned += num_instances

            # Dernier agent prend le reste
            last_agent_id = sorted_agents[-1][0]
            new_allocation[last_agent_id] = list(range(instances_assigned, total_instances))

            # Appliquer nouvelle allocation
            self.allocation = new_allocation

            # Mettre à jour instance_to_agents
            self.instance_to_agents = defaultdict(list)
            for agent_id, instances in self.allocation.items():
                for inst_id in instances:
                    self.instance_to_agents[inst_id].append(agent_id)

            logger.info("\nNouvelle allocation :")
            for agent_id in sorted(new_allocation.keys()):
                instances = new_allocation[agent_id]
                logger.info(f"  Agent {agent_id} : {len(instances)} instances {instances}")

        elif scenario == "INSTANCE_SHARING":
            # Meilleurs agents obtiennent instances exclusives
            num_instances = len(self.instance_to_agents)

            # Calculer ratio élite
            elite_count = max(1, int(len(sorted_agents) * 0.3))
            elite_agents = [agent_id for agent_id, _ in sorted_agents[:elite_count]]
            weak_agents = [agent_id for agent_id, _ in sorted_agents[elite_count:]]

            new_allocation = {}

            # Élites : instances exclusives
            instances_per_elite = max(1, num_instances // elite_count)
            instance_idx = 0

            for agent_id in elite_agents:
                new_allocation[agent_id] = [instance_idx]
                instance_idx = (instance_idx + 1) % num_instances

            # Faibles : partagent les instances restantes
            for agent_id in weak_agents:
                # Assigner à l'instance la moins chargée
                instance_loads = {}
                for inst_id in range(num_instances):
                    instance_loads[inst_id] = sum(
                        1 for alloc in new_allocation.values() if inst_id in alloc
                    )

                least_loaded = min(instance_loads.items(), key=lambda x: x[1])[0]
                new_allocation[agent_id] = [least_loaded]

            # Appliquer
            self.allocation = new_allocation

            # Mettre à jour instance_to_agents
            self.instance_to_agents = defaultdict(list)
            for agent_id, instances in self.allocation.items():
                for inst_id in instances:
                    self.instance_to_agents[inst_id].append(agent_id)

            # Réinitialiser current_agent_per_instance
            self.current_agent_per_instance = {
                inst_id: agents_list[0]
                for inst_id, agents_list in self.instance_to_agents.items()
            }

            logger.info("\nNouvelle allocation :")
            logger.info(f"Élites ({elite_count}) : instances dédiées")
            logger.info(f"Faibles ({len(weak_agents)}) : instances partagées")

            for inst_id, agents in self.instance_to_agents.items():
                logger.info(f"  Instance {inst_id} : Agents {agents}")

        logger.info("=" * 70)
        logger.info("")

        self.last_weighted_eval = total_episodes
        return True
