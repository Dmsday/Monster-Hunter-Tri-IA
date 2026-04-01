"""
Logger simple pour les modules
Complément à advanced_logging.py qui gère les données d'entraînement

Ce logger gère les logs généraux (info, debug, warning, error)
advanced_logging.py gère les données d'entraînement (steps, épisodes, métriques)
"""

# ============================================================
# IMPORTS — tous en haut du fichier
# ============================================================
import logging
import sys
import time                                  # Utilisé par log_execution_time
import functools                             # Utilisé par log_every_n_calls (@wraps)
from typing import Optional
from contextlib import contextmanager        # Utilisé par log_execution_time


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

# Niveau global (modifiable depuis train.py via set_global_log_level)
_GLOBAL_LOG_LEVEL = logging.WARNING  # stdout only — files always get DEBUG

# Registry des file handlers créés par TrainingLogger.
# Quand un nouveau mh_* logger est créé, tous ces handlers lui sont attachés.
# Quand un handler est ajouté, tous les mh_* existants le reçoivent aussi.
_GLOBAL_FILE_HANDLERS: list = []


def register_file_handler(handler: logging.Handler) -> None:
    """
    Enregistre un file handler (créé par TrainingLogger) pour qu'il soit
    attaché à tous les loggers mh_* existants ET futurs.

    Appelé uniquement par advanced_logging.TrainingLogger._setup_loggers().
    Le handler n'a PAS de AgentContextFilter : tous les agents reçoivent
    tous les logs (contrainte DummyVecEnv thread unique).
    """
    _GLOBAL_FILE_HANDLERS.append(handler)

    # Attacher immédiatement à tous les mh_* loggers déjà créés
    for logger_name in list(logging.root.manager.loggerDict.keys()):
        if logger_name.startswith('mh_'):
            existing = logging.getLogger(logger_name)
            if handler not in existing.handlers:
                existing.addHandler(handler)
                existing.propagate = False  # évite les doublons sur stdout


def set_global_log_level(level: str):
    """
    Change le niveau de log GLOBAL pour tous les modules.

    Met à jour dynamiquement le niveau des loggers mh_* ET de leur handler
    stdout. Les file handlers (debug.log, console.log) gardent leur propre
    niveau — debug.log reste à DEBUG même si le niveau global est ERROR.

    Args:
        level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    """
    global _GLOBAL_LOG_LEVEL
    _GLOBAL_LOG_LEVEL = getattr(logging, level.upper())

    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith('mh_'):
            existing_logger = logging.getLogger(logger_name)
            # Logger level is always DEBUG so file handlers receive everything.
            # Only the stdout StreamHandler is filtered by --log-level.
            existing_logger.setLevel(logging.DEBUG)
            for handler in existing_logger.handlers:
                if (isinstance(handler, logging.StreamHandler)
                        and not isinstance(handler, logging.FileHandler)):
                    handler.setLevel(_GLOBAL_LOG_LEVEL)


# ============================================================
# FORMATTER AVEC CONTEXTE AGENT
# ============================================================

class AgentAwareFormatter(logging.Formatter):
    """
    Formatter qui ajoute automatiquement le préfixe [Agent X] basé sur le contexte thread-local.

    En multi-agent, chaque thread a son propre agent_id dans AgentContext.
    Ce formatter lit ce contexte à chaque log pour préfixer automatiquement,
    sans avoir à passer l'agent_id manuellement à chaque logger.info().

    Résultat :
        "14:32:01 | INFO     | mh_env              | Agent 2 | Reset episode #5..."
    """

    def format(self, record):
        from info.agent_context import AgentContext, EnvContext
        agent_id = AgentContext.get_current_agent()
        env_id = EnvContext.get_current_env()

        parts = []
        if agent_id is not None and isinstance(agent_id, int) and agent_id >= 0:
            parts.append(f"Agent {agent_id}")
        if env_id is not None and isinstance(env_id, int) and env_id >= 0:
            # Only show env tag when it differs from agent_id (avoids redundancy
            # in ONE_TO_ONE mode where agent_id == env_id already)
            parts.append(f"Env {env_id}")

        record.agent_prefix = " | ".join(parts) + " | " if parts else ""
        return super().format(record)


# ============================================================
# CRÉATION DE LOGGER PAR MODULE
# ============================================================

def get_module_logger(module_name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Crée (ou récupère s'il existe déjà) un logger pour un module.

    Tous les loggers créés ici :
    - Utilisent AgentAwareFormatter (préfixe [Agent X] automatique en multi-agent)
    - Se connectent au handler de advanced_logging si celui-ci est actif
    - Partagent le niveau global modifiable via set_global_log_level()

    Args:
        module_name: Nom court du module (ex: 'mh_env', 'frame_capture')
        level: Niveau spécifique pour ce module (si None, utilise le global)

    Returns:
        logging.Logger configuré et prêt à l'emploi

    Exemple:
        from info.module_logger import get_module_logger
        logger = get_module_logger('mh_env')

        logger.info("Environnement initialisé")
        logger.warning("Frame identique détectée")
        logger.error("Erreur critique")
    """
    # Nommer nos loggers avec un préfixe pour les distinguer des libs externes
    # Exemple : 'mh_env', 'mh_frame_capture', 'mh_reward_calculator'
    logger_name = f'mh_{module_name}'
    logger = logging.getLogger(logger_name)

    # Si le logger existe déjà (appelé plusieurs fois avec le même module), le retourner tel quel
    # Évite d'ajouter des handlers en double
    if logger.handlers:
        return logger

    # Logger always at DEBUG — file handlers must receive everything.
    # Stdout handler is separately gated by _GLOBAL_LOG_LEVEL.
    logger.setLevel(logging.DEBUG)

    # Créer le formatter agent-aware (ajoute "Agent X | " quand contexte défini)
    agent_formatter = AgentAwareFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(agent_prefix)s%(message)s',
        datefmt='%H:%M:%S'
    )

    # Handler console (stdout) — toujours présent
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(agent_formatter)
    # Stdout respects --log-level
    console_handler.setLevel(
        getattr(logging, level.upper()) if level else _GLOBAL_LOG_LEVEL
    )
    logger.addHandler(console_handler)

    # Handler advanced_logging (optionnel) — actif seulement si TrainingLogger est initialisé
    # Ce handler écrit dans console.log (fichier disque structuré par expérience)
    # Il est créé par advanced_logging.py sous le nom 'advanced_console_capture'
    advanced_console_logger = logging.getLogger('advanced_console_capture')
    if advanced_console_logger.handlers:
        for handler in advanced_console_logger.handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)

        # propagate=False : empêche la remontée au root logger qui causerait des doublons
        logger.propagate = False
    else:
        # advanced_logging pas encore initialisé : propager au root pour ne rien perdre
        # (ce flag sera corrigé si TrainingLogger est créé après)
        logger.propagate = True

    # Appliquer le formatter agent-aware à tous les handlers (y compris ceux d'advanced_logging)
    for handler in logger.handlers:
        if not isinstance(handler.formatter, AgentAwareFormatter):
            handler.setFormatter(agent_formatter)

    # Attacher les file handlers déjà enregistrés par TrainingLogger (si déjà créés)
    for file_handler in _GLOBAL_FILE_HANDLERS:
        if file_handler not in logger.handlers:
            logger.addHandler(file_handler)
            logger.propagate = False  # évite les doublons sur stdout

    return logger


# ============================================================
# DÉCORATEUR POUR LOGS PÉRIODIQUES
# ============================================================

def log_every_n_calls(n: int = 1000, level: str = 'DEBUG'):
    """
    Décorateur pour logger seulement tous les N appels d'une fonction.

    Utile pour les fonctions très fréquentes (_get_observation, step, etc.)
    où on veut un log de monitoring sans spammer la console.

    Args:
        n: Fréquence de log (ex: 1000 = log tous les 1000 appels)
        level: Niveau de log ('DEBUG', 'INFO', etc.)

    Exemple:
        @log_every_n_calls(1000, 'INFO')
        def _get_observation(self):
            ...
        # Produira : "_get_observation() called 1,000 times"
        #            "_get_observation() called 2,000 times" etc.
    """
    def decorator(func):
        func.call_count = 0
        func._logger = get_module_logger(func.__module__)

        @functools.wraps(func)  # Préserve __name__, __doc__, __module__ de la fonction originale
        def wrapper(*args, **kwargs):
            func.call_count += 1
            if func.call_count % n == 0:
                log_method = getattr(func._logger, level.lower())
                log_method(f"{func.__name__}() called {func.call_count:,} times")
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================
# CONTEXT MANAGER POUR TIMING
# ============================================================

@contextmanager
def log_execution_time(operation_name: str, logger: logging.Logger, level: str = 'DEBUG'):
    """
    Context manager pour mesurer et logger le temps d'exécution d'un bloc.

    Exemple:
        with log_execution_time('Capture frame', logger):
            frame = self.capture_frame()
        # → Log: "Capture frame: 5.2ms"
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_method = getattr(logger, level.lower())
        log_method(f"{operation_name}: {elapsed_ms:.1f}ms")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("Test module_logger\n")

    env_logger = get_module_logger('mh_env')
    capture_logger = get_module_logger('frame_capture')

    env_logger.debug("Debug message (pas affiché par défaut)")
    env_logger.info("Info message (pas affiché par défaut)")
    env_logger.warning("Warning message (affiché)")
    env_logger.error("Error message (affiché)")

    print("\n--- Changer niveau global à DEBUG ---\n")
    set_global_log_level('DEBUG')

    env_logger.debug("Debug maintenant visible")
    capture_logger.info("Info maintenant visible")

    print("\n--- Test timing ---\n")
    with log_execution_time('Opération test', env_logger, 'INFO'):
        time.sleep(0.1)

    print("\nTest terminé")