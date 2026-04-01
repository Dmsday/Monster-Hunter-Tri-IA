"""
Scheduler pour gérer plusieurs agents PPO partageant des instances Dolphin.
Implémente les modes : independent, round_robin, majority_vote, weighted.
"""

from typing import List, Dict, Tuple
from collections import defaultdict, Counter

import numpy as np
from stable_baselines3 import PPO

from info.module_logger import get_module_logger
from core.controller.action_heads import NUM_HEADS

logger = get_module_logger('multi_agent_scheduler')


class MultiAgentScheduler:
    """
    Gère l'ordonnancement des actions de plusieurs agents sur instances partagées.

    Modes supportés :
    - independent  : Tour par tour, 1 step à la fois (chaque agent joue alternativement)
    - round_robin  : Tour par tour par blocs de N steps (moins de changements de contexte)
    - majority_vote: Tous les agents prédisent, l'action majoritaire est exécutée
    - weighted     : Allocation adaptative selon performances (rééquilibrage périodique)

    Note: Le mode 'genetic' n'utilise PAS ce scheduler.
          GeneticTrainer gère lui-même la sélection/mutation des agents.
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
            agents: Liste des agents PPO (indexés 0..N-1)
            allocation: Dict {agent_id: [instance_ids]} qui définit quelle instance appartient à quel agent
            mode: Stratégie de scheduling ('independent', 'round_robin', 'majority_vote', 'weighted')
            block_size: Nombre de steps par bloc pour le mode round_robin
            weighted_eval_freq: Nombre d'épisodes entre deux réévaluations pour le mode weighted
        """
        self.agents = agents
        self.allocation = allocation
        self.mode = mode
        self.block_size = block_size
        self.weighted_eval_freq = weighted_eval_freq

        # Construire la map inverse : instance_id → [agent_ids] (utilisée à chaque step)
        # Exemple : {0: [0, 1], 1: [2]} si agents 0 et 1 partagent l'instance 0
        self.instance_to_agents = self._build_instance_to_agents(allocation)

        # Agent actif par instance (pour les modes qui tournent en round)
        self.current_agent_per_instance = {
            inst_id: agents_list[0]
            for inst_id, agents_list in self.instance_to_agents.items()
        }

        # Compteur de steps par instance (pour le mode round_robin)
        self.step_count_per_instance = defaultdict(int)

        # Stats pour le mode weighted : historique des rewards par agent
        self.agent_scores: Dict[int, List[float]] = defaultdict(list)   # {agent_id: [reward, ...]}
        self.agent_total_steps: Dict[int, int] = defaultdict(int)
        self.last_weighted_eval = 0   # Nb d'épisodes au dernier rééquilibrage

        logger.info(f"MultiAgentScheduler created (mode={mode})")
        logger.info(f"   Inverted allocation: {dict(self.instance_to_agents)}")
        if mode == 'weighted':
            logger.info(f"   Re-evaluation every {weighted_eval_freq} episodes")

    # ================================================================
    # HELPERS INTERNES
    # ================================================================

    @staticmethod
    def _build_instance_to_agents(allocation: Dict[int, List[int]]) -> defaultdict:
        """
        Construit la map inverse allocation : instance_id → [agent_ids].

        Extrait en méthode séparée car appelé à la fois dans __init__
        et dans rebalance_weighted_allocation() après chaque rééquilibrage.

        Args:
            allocation: Dict {agent_id: [instance_ids]}

        Returns:
            defaultdict {instance_id: [agent_ids]}
        """
        instance_map = defaultdict(list)
        for agent_id, instances in allocation.items():
            for inst_id in instances:
                instance_map[inst_id].append(agent_id)
        return instance_map

    # ================================================================
    # ACTION PRINCIPALE
    # ================================================================

    def get_action(
            self,
            instance_id: int,
            observation: np.ndarray
    )  -> Tuple[np.ndarray, int]:
        """
        Retourne l'action à exécuter et l'agent qui l'a choisie.

        Args:
            instance_id: Index de l'instance Dolphin (0, 1, 2, ...)
            observation: Observation actuelle de cet environnement

        Returns:
            (action, agent_id_used)
            - action: int, index de l'action (0-18 pour MH Tri)
            - agent_id_used: int, ID de l'agent qui a décidé (-1 = tous, pour majority_vote)

        Note:
            Le mode 'genetic' n'utilise PAS cette méthode (GeneticTrainer gère directement).
        """
        agents_on_instance = self.instance_to_agents[instance_id]

        # Cas simple : 1 seul agent sur cette instance → pas besoin de scheduler
        if len(agents_on_instance) == 1:
            agent_id = agents_on_instance[0]
            action, _ = self.agents[agent_id].predict(observation, deterministic=False)
            return np.asarray(action), agent_id

        # Plusieurs agents : router vers le mode configuré
        if self.mode == 'independent':
            return self._independent_mode(instance_id, observation, agents_on_instance)
        elif self.mode == 'round_robin':
            return self._round_robin_mode(instance_id, observation, agents_on_instance)
        elif self.mode == 'majority_vote':
            return self._majority_vote_mode(instance_id, observation, agents_on_instance)
        else:
            raise ValueError(f"Mode inconnu : '{self.mode}'. Valides: independent, round_robin, majority_vote, weighted")

    # ================================================================
    # MODES DE SCHEDULING
    # ================================================================

    def _independent_mode(
            self,
            instance_id: int,
            observation: np.ndarray,
            agents_on_instance: List[int]
    )  -> Tuple[np.ndarray, int]:
        """
        Mode independent : chaque agent joue à tour de rôle, 1 step chacun.

        Exemple avec agents [0, 1] sur instance 0 :
            Step 1 → Agent 0 joue, Agent 1 attend
            Step 2 → Agent 1 joue, Agent 0 attend
            Step 3 → Agent 0 joue, ...
        """
        current_agent_id = self.current_agent_per_instance[instance_id]
        action, _ = self.agents[current_agent_id].predict(observation, deterministic=False)

        # Avancer le pointeur circulaire vers le prochain agent
        current_index = agents_on_instance.index(current_agent_id)
        next_index = (current_index + 1) % len(agents_on_instance)
        self.current_agent_per_instance[instance_id] = agents_on_instance[next_index]

        return np.asarray(action), current_agent_id

    def _round_robin_mode(
            self,
            instance_id: int,
            observation: np.ndarray,
            agents_on_instance: List[int]
    )  -> Tuple[np.ndarray, int]:
        """
        Mode round_robin : par blocs de N steps (moins de switchs = plus stable).

        Exemple avec block_size=100, agents [0, 1] :
            Steps 1-100   → Agent 0 joue
            Steps 101-200 → Agent 1 joue
            Steps 201-300 → Agent 0 joue, ...
        """
        current_agent_id = self.current_agent_per_instance[instance_id]
        action, _ = self.agents[current_agent_id].predict(observation, deterministic=False)

        self.step_count_per_instance[instance_id] += 1

        # Changer d'agent uniquement quand le bloc est épuisé
        if self.step_count_per_instance[instance_id] % self.block_size == 0:
            current_index = agents_on_instance.index(current_agent_id)
            next_agent = agents_on_instance[(current_index + 1) % len(agents_on_instance)]
            self.current_agent_per_instance[instance_id] = next_agent
            logger.debug(f"Instance {instance_id}: switching to agent {next_agent}")

        return np.asarray(action), current_agent_id

    def _majority_vote_mode(
            self,
            instance_id: int,               # noqa: ARG002 — requis par l'interface get_action
            observation: np.ndarray,
            agents_on_instance: List[int]
    )  -> Tuple[np.ndarray, int]:
        """
        Mode majority_vote : tous les agents prédisent, l'action la plus fréquente gagne.

        Avantage : réduit la variance, l'ensemble se comporte mieux qu'un seul agent.
        Inconvénient : N fois plus de calculs GPU par step (N = nb agents sur l'instance).

        Returns:
            (action, -1) : -1 = tous les agents contribuent (pas un seul élu)
        """
        # Collecter les votes de chaque agent
        all_actions = [
            self.agents[agent_id].predict(observation, deterministic=False)[0]
            for agent_id in agents_on_instance
        ]
        # Vote independently per head
        majority = np.zeros(NUM_HEADS, dtype=np.int32)
        for h in range(NUM_HEADS):
            head_votes = [int(a[h]) for a in all_actions]
            majority[h] = Counter(head_votes).most_common(1)[0][0]
        return majority, -1

    # ================================================================
    # GESTION DES SCORES (mode weighted)
    # ================================================================

    def update_agent_score(self, agent_id: int, episode_reward: float):
        """
        Enregistre le reward d'un épisode terminé pour un agent.

        Utilisé par le mode weighted pour décider de l'allocation d'instances.
        On garde les 100 derniers épisodes pour un score glissant.

        Args:
            agent_id: ID de l'agent dont l'épisode vient de se terminer
            episode_reward: Reward total de l'épisode (float)
        """
        self.agent_scores[agent_id].append(episode_reward)

        # Fenêtre glissante : garder seulement les 100 derniers épisodes
        if len(self.agent_scores[agent_id]) > 100:
            self.agent_scores[agent_id] = self.agent_scores[agent_id][-100:]

    # ================================================================
    # RÉÉQUILIBRAGE (mode weighted)
    # ================================================================

    def rebalance_weighted_allocation(self, scenario: str) -> bool:
        """
        Rééquilibre l'allocation d'instances selon les performances des agents.

        Appelé périodiquement (tous les weighted_eval_freq épisodes).
        Les agents performants obtiennent plus d'instances, les faibles en perdent.

        Args:
            scenario: 'AGENT_MULTIPLE_INSTANCES' ou 'INSTANCE_SHARING'
                - AGENT_MULTIPLE_INSTANCES : chaque agent a ses propres instances
                - INSTANCE_SHARING : plusieurs agents partagent les mêmes instances

        Returns:
            True si le rééquilibrage a été effectué, False si pas encore nécessaire.
        """
        if self.mode != 'weighted':
            return False

        # Vérifier si assez d'épisodes se sont passés depuis le dernier rééquilibrage
        total_episodes = sum(len(scores) for scores in self.agent_scores.values())
        if total_episodes - self.last_weighted_eval < self.weighted_eval_freq:
            return False

        logger.info("")
        logger.info("=" * 70)
        logger.info("RÉÉQUILIBRAGE ALLOCATION (MODE WEIGHTED)")
        logger.info("=" * 70)

        # Calculer le score moyen sur les 50 derniers épisodes de chaque agent
        agent_avg_scores = {
            agent_id: float(np.mean(scores[-50:])) if scores else 0.0
            for agent_id, scores in self.agent_scores.items()
        }

        logger.info("Scores moyens (50 derniers épisodes) :")
        for agent_id in sorted(agent_avg_scores):
            logger.info(f"  Agent {agent_id} : {agent_avg_scores[agent_id]:.2f}")

        # Trier du meilleur au moins bon
        sorted_agents = sorted(agent_avg_scores.items(), key=lambda x: x[1], reverse=True)

        if scenario == "AGENT_MULTIPLE_INSTANCES":
            self._rebalance_agent_multiple_instances(sorted_agents)
        elif scenario == "INSTANCE_SHARING":
            self._rebalance_instance_sharing(sorted_agents)

        logger.info("=" * 70)
        logger.info("")

        self.last_weighted_eval = total_episodes
        return True

    def _rebalance_agent_multiple_instances(self, sorted_agents: List[Tuple[int, float]]):
        """
        Scénario AGENT_MULTIPLE_INSTANCES :
        Redistribue les instances proportionnellement aux scores (plus tu performes, plus tu joues).

        Args:
            sorted_agents: [(agent_id, score), ...] trié du meilleur au moins bon
        """
        total_instances = sum(len(insts) for insts in self.allocation.values())
        total_score = sum(score for _, score in sorted_agents if score > 0)

        if total_score == 0:
            logger.warning("Tous les scores sont nuls - rééquilibrage annulé")
            return

        new_allocation: Dict[int, List[int]] = {}
        instances_assigned = 0

        # Distribuer proportionnellement (sauf au dernier qui prend le reste)
        for agent_id, score in sorted_agents[:-1]:
            weight = score / total_score
            num_instances = max(1, int(weight * total_instances))
            new_allocation[agent_id] = list(range(instances_assigned, instances_assigned + num_instances))
            instances_assigned += num_instances

        # Dernier agent : toutes les instances restantes
        last_agent_id = sorted_agents[-1][0]
        new_allocation[last_agent_id] = list(range(instances_assigned, total_instances))

        self._apply_new_allocation(new_allocation)

        logger.info("Nouvelle allocation (proportionnelle aux scores) :")
        for agent_id in sorted(new_allocation):
            logger.info(f"  Agent {agent_id} : {len(new_allocation[agent_id])} instances {new_allocation[agent_id]}")

    def _rebalance_instance_sharing(self, sorted_agents: List[Tuple[int, float]]):
        """
        Scénario INSTANCE_SHARING :
        Les agents élites (top 30%) obtiennent des instances dédiées.
        Les agents faibles partagent les instances les moins chargées.

        Args:
            sorted_agents: [(agent_id, score), ...] trié du meilleur au moins bon
        """
        num_instances = len(self.instance_to_agents)

        # Séparer élites (top 30%) et agents faibles
        elite_count = max(1, int(len(sorted_agents) * 0.3))
        elite_agents = [agent_id for agent_id, _ in sorted_agents[:elite_count]]
        weak_agents  = [agent_id for agent_id, _ in sorted_agents[elite_count:]]

        new_allocation: Dict[int, List[int]] = {}

        # Élites : 1 instance dédiée chacun (round-robin sur les instances disponibles)
        instance_idx = 0
        for agent_id in elite_agents:
            new_allocation[agent_id] = [instance_idx]
            instance_idx = (instance_idx + 1) % num_instances

        # Faibles : chacun rejoint l'instance la moins chargée
        for agent_id in weak_agents:
            # Compter la charge de chaque instance (nb d'agents déjà assignés)
            instance_loads = {
                inst_id: sum(1 for alloc in new_allocation.values() if inst_id in alloc)
                for inst_id in range(num_instances)
            }
            least_loaded_instance = min(instance_loads, key=lambda iid: instance_loads[iid])
            new_allocation[agent_id] = [least_loaded_instance]

        self._apply_new_allocation(new_allocation)

        logger.info("Nouvelle allocation (élites dédiées / faibles partagés) :")
        logger.info(f"  Élites ({elite_count}) : instances dédiées → {elite_agents}")
        logger.info(f"  Faibles ({len(weak_agents)}) : instances partagées → {weak_agents}")
        for inst_id, agents in self.instance_to_agents.items():
            logger.info(f"  Instance {inst_id} : Agents {agents}")

    def _apply_new_allocation(self, new_allocation: Dict[int, List[int]]):
        """
        Applique une nouvelle allocation et met à jour toutes les structures internes.

        Centralise la mise à jour de self.allocation, instance_to_agents et
        current_agent_per_instance pour éviter de répéter ce code dans chaque scénario.

        Args:
            new_allocation: Dict {agent_id: [instance_ids]} calculé par un scénario de rééquilibrage
        """
        self.allocation = new_allocation

        # Reconstruire la map inverse instance→agents
        self.instance_to_agents = self._build_instance_to_agents(new_allocation)

        # Réinitialiser l'agent actif par instance (premier de la liste)
        self.current_agent_per_instance = {
            inst_id: agents_list[0]
            for inst_id, agents_list in self.instance_to_agents.items()
        }