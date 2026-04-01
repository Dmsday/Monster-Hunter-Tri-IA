"""
Système de contexte d'agent pour le routage automatique des logs en multi-agent.

Permet de détecter automatiquement quel agent est actif dans le thread courant,
afin que les logs soient préfixés "[Agent X]" sans l'écrire manuellement partout.

Usage basique :
    # À la création de l'environnement (ou au début d'un step/reset)
    AgentContext.set_current_agent(agent_id=0)

    # Tous les logs suivants de CE thread seront tagués automatiquement
    logger.info("Observation calculée")  # → "Agent 0 | Observation calculée"

    # Réinitialiser le contexte
    AgentContext.clear()

Usage avec context manager (recommandé pour du code scopé) :
    with agent_context(agent_id=2):
        logger.info("Training step")  # → "Agent 2 | Training step"
    # contexte restauré automatiquement à la sortie du bloc
"""

import threading
from typing import Optional


class AgentContext:
    """
    Contexte thread-local pour tracker l'agent courant.

    Utilise threading.local() : chaque thread Python a sa propre valeur d'agent_id.
    Ainsi, agent 0 et agent 1 peuvent tourner en parallèle sans s'écraser.

    Pas de variable globale fallback : elle causerait des corruptions croisées en multi-thread.
    """

    # threading.local() : chaque thread lit/écrit sa propre copie de cet objet
    _context = threading.local()

    @classmethod
    def set_current_agent(cls, agent_id: int):
        """
        Définit l'agent courant pour le thread appelant.

        Args:
            agent_id: ID de l'agent (0, 1, 2, ...)

        Exemple:
            AgentContext.set_current_agent(0)
            logger.info("Training")  # tagué automatiquement comme Agent 0
        """
        cls._context.agent_id = agent_id

    @classmethod
    def get_current_agent(cls) -> Optional[int]:
        """
        Retourne l'ID de l'agent courant pour ce thread.

        Returns:
            L'agent_id (int) si défini dans ce thread, None sinon.

        Note:
            hasattr() gère déjà AttributeError en interne, pas besoin de try/except.
            On garde quand même la vérification isinstance pour s'assurer d'un int valide.
        """
        if hasattr(cls._context, 'agent_id'):
            agent_id = cls._context.agent_id
            if isinstance(agent_id, int):
                return agent_id
        return None

    @classmethod
    def clear(cls):
        """Efface le contexte agent pour le thread courant."""
        if hasattr(cls._context, 'agent_id'):
            delattr(cls._context, 'agent_id')

    @classmethod
    def is_set(cls) -> bool:
        """Vérifie si un contexte agent est actif dans ce thread."""
        return cls.get_current_agent() is not None


# ============================================================
# CONTEXT MANAGER POUR SCOPING PROPRE
# ============================================================

class agent_context:
    """
    Context manager pour définir un contexte agent sur un bloc de code.

    Restaure automatiquement le contexte précédent à la sortie du bloc,
    ce qui permet d'imbriquer des contextes sans perdre l'état parent.

    Usage:
        with agent_context(agent_id=2):
            logger.info("Training step")  # → "Agent 2 | Training step"
        # ici le contexte est restauré à ce qu'il était avant
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.previous_agent: Optional[int] = None

    def __enter__(self):
        # Sauvegarder l'agent précédent pour le restaurer à la sortie
        self.previous_agent = AgentContext.get_current_agent()
        AgentContext.set_current_agent(self.agent_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurer le contexte précédent (ou effacer s'il n'y en avait pas)
        if self.previous_agent is not None:
            AgentContext.set_current_agent(self.previous_agent)
        else:
            AgentContext.clear()
        # Retourner False = ne pas supprimer les exceptions éventuelles
        return False

class EnvContext:
    """
    Thread-local context tracking the current environment instance id.
    Independent from AgentContext: an agent can run multiple envs.
    """
    _context = threading.local()

    @classmethod
    def set_current_env(cls, env_id: int) -> None:
        cls._context.env_id = env_id

    @classmethod
    def get_current_env(cls) -> Optional[int]:
        if hasattr(cls._context, 'env_id'):
            v = cls._context.env_id
            if isinstance(v, int):
                return v
        return None

    @classmethod
    def clear(cls) -> None:
        if hasattr(cls._context, 'env_id'):
            delattr(cls._context, 'env_id')