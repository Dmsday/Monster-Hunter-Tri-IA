"""
Logger simple pour les modules
Complément à advanced_logging.py qui gère les données d'entraînement

Ce logger gère les logs généraux (info, debug, warning, error)
advanced_logging.py gère les données d'entraînement (steps, épisodes, métriques)
"""

import logging
import sys
from typing import Optional

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

# Niveau global (modifiable depuis train.py)
_GLOBAL_LOG_LEVEL = logging.WARNING  # Par défaut : seulement warnings/errors


def set_global_log_level(level: str):
    """
    Change le niveau de log GLOBAL pour tous les modules

    Args:
        level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    """
    global _GLOBAL_LOG_LEVEL
    _GLOBAL_LOG_LEVEL = getattr(logging, level.upper())

    # Mettre à jour tous les loggers existants
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith('mhlog_'):  # Seulement nos modules
            logger = logging.getLogger(logger_name)
            logger.setLevel(_GLOBAL_LOG_LEVEL)

            # Mettre à jour aussi tous les handlers de ce logger
            for handler in logger.handlers:
                handler.setLevel(_GLOBAL_LOG_LEVEL)

# ============================================================
# CRÉATION DE LOGGER PAR MODULE
# ============================================================

def get_module_logger(module_name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Crée un logger pour un module

    Args:
        module_name: Nom du module (ex: 'mh_env', 'frame_capture')
        level: Niveau spécifique (si None, utilise global)

    Returns:
        logging.Logger configuré

    Exemple:
        # Dans mh_env.py
        from utils.module_logger import get_module_logger
        logger = get_module_logger('mh_env')

        logger.info("Environnement initialisé")
        logger.warning("Frame identique détectée")
        logger.error("Erreur critique")
    """
    # Préfixe pour identifier nos loggers
    logger_name = f'mh_{module_name}'

    # Récupérer ou créer logger
    logger = logging.getLogger(logger_name)

    # Si déjà configuré, retourner
    if logger.handlers:
        return logger

    # Définir niveau
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    else:
        logger.setLevel(_GLOBAL_LOG_LEVEL)

    # Format des messages
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logger.level)
    logger.addHandler(console_handler)

    # Vérifier si advanced_logging existe et ajouter son handler
    # Cela permet de capturer les logs module dans advanced_logging
    advanced_console_logger = logging.getLogger('advanced_console_capture')
    if advanced_console_logger.handlers:
        # Si advanced_logging est actif, ajouter ses handlers
        # (notamment celui qui écrit dans console.log)
        for handler in advanced_console_logger.handlers:
            # Éviter les doublons si le handler est déjà présent
            if handler not in logger.handlers:
                logger.addHandler(handler)

        # IMPORTANT : Mettre propagate à False pour éviter la duplication vers le root logger (qui pourrait causer des doublons)
        logger.propagate = False
    else:
        # Si advanced_logging n'existe pas encore, propager au root
        # (cas où module_logger est utilisé avant TrainingLogger)
        logger.propagate = True

    return logger


# ============================================================
# DÉCORATEUR POUR LOGS PÉRIODIQUES
# ============================================================

def log_every_n_calls(n: int = 1000, level: str = 'DEBUG'):
    """
    Décorateur pour logger seulement tous les N appels

    Args:
        n: Fréquence de log (ex: tous les 1000 appels)
        level: Niveau de log ('DEBUG', 'INFO', etc.)

    Exemple:
        @log_every_n_calls(1000, 'INFO')
        def _get_observation(self):
            # ... code ...
            pass

        # Loggera "Appelé 1000 fois, 2000 fois, etc."
    """

    def decorator(func):
        func.call_count = 0
        func._logger = get_module_logger(func.__module__)

        def wrapper(*args, **kwargs):
            func.call_count += 1

            # Logger tous les N appels
            if func.call_count % n == 0:
                log_method = getattr(func.logger, level.lower())
                log_method(
                    f"{func.__name__}() appelé {func.call_count:,} fois"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================
# CONTEXT MANAGER POUR TIMING
# ============================================================

import time
from contextlib import contextmanager


@contextmanager
def log_execution_time(operation_name: str, logger: logging.Logger, level: str = 'DEBUG'):
    """
    Context manager pour logger le temps d'exécution

    Exemple:
        with log_execution_time('Capture frame', logger):
            frame = self.capture_frame()

        # Log: "Capture frame: 5.2ms"
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
    print("🧪 Test module_logger\n")

    # Créer loggers pour différents modules
    env_logger = get_module_logger('mh_env')
    capture_logger = get_module_logger('frame_capture')

    # Test différents niveaux
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

    print("\n✅ Test terminé")