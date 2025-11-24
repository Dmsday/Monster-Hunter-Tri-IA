"""
Système de logging avancé pour Monster Hunter RL
Créé plusieurs fichiers de logs structurés :
- errors.log : Toutes les erreurs avec traceback
- console.log : Copie de tout ce qui s'affiche en console
- training_data.jsonl : Données d'entraînement (rewards, états, actions)
- session_summary.json : Résumé de la session
- debug.log : Informations de debug détaillées
"""
import sys
import json
import logging
import traceback
import numpy as np

from datetime import datetime
from typing import Dict, Any
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from utils.module_logger import get_module_logger
module_logger = get_module_logger('advanced_logging')


class TrainingLogger:
    """
    Système de logging multi-fichiers pour l'entraînement
    """

    def __init__(
            self,
            experiment_name: str,
            base_dir: str = "./logs",
            console_log_level: str = "WARNING"
    ):
        """
        Args:
            experiment_name : Nom de l'expérience
            base_dir : Dossier racine des logs
            console_log_level : Niveau pour logs console ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.experiment_name = experiment_name
        self.session_start = datetime.now()
        self.console_log_level = console_log_level

        # Créer dossier de logs pour cette session
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(base_dir) / experiment_name / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)

        module_logger.info(f"📁 Dossier de logs : {self.log_dir}")

        # Compteurs
        self.step_count = 0
        self.episode_count = 0
        self.error_count = 0
        self.warning_count = 0

        # Stockage des données d'épisodes pour statistiques
        # Permet de calculer moyenne, écart-type, min, max
        self.episode_rewards = []      # Liste des rewards totales par épisode
        self.episode_lengths = []      # Liste des longueurs d'épisodes
        self.episode_hits = []         # Liste du nombre de hits par épisode
        self.episode_deaths = []       # Liste du nombre de morts par épisode
        self.episode_zones = []        # Liste du nombre de zones découvertes

        # Session data
        self.session_data = {
            'experiment_name': experiment_name,
            'start_time': self.session_start.isoformat(),
            'end_time': None,
            'duration_seconds': 0.0,
            'total_steps': 0,
            'total_episodes': 0,
            'errors': 0,
            'warnings': 0,
            'config': {},
            'episode_statistics': {},
        }

        # Setup des différents loggers
        self._setup_loggers()

        # Fichier training data (JSONL)
        self.training_data_file = open(
            self.log_dir / "training_data.jsonl",
            'w',
            encoding='utf-8'
        )

        module_logger.info("Système de logging initialisé")
        module_logger.info(f"📄 Fichiers créés :")
        module_logger.info(f"   - errors.log (erreurs avec traceback)")
        module_logger.info(f"   - console.log (sortie console)")
        module_logger.info(f"   - training_data.jsonl (données step-by-step)")
        module_logger.info(f"   - debug.log (debug détaillé)")
        module_logger.info(f"   - session_summary.json (résumé)")

    def _setup_loggers(self):
        """
        Configure les différents loggers
        """

        # 1. ERROR LOGGER
        self.error_logger = logging.getLogger(f'{self.experiment_name}_errors')
        self.error_logger.setLevel(logging.ERROR)

        error_handler = logging.FileHandler(
            self.log_dir / "errors.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s | STEP:%(step)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)

        # 2. CONSOLE LOGGER
        self.console_logger = logging.getLogger(f'{self.experiment_name}_console')
        # Utiliser le niveau configuré
        console_level = getattr(logging, self.console_log_level.upper())
        self.console_logger.setLevel(console_level)

        console_handler = logging.FileHandler(
            self.log_dir / "console.log",
            encoding='utf-8',
        )
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        console_handler.setFormatter(console_formatter)
        self.console_logger.addHandler(console_handler)

        # Créer un logger global pour capturer les messages de module_logger
        # Ce logger servira de pont entre module_logger et console.log
        self.global_capture_logger = logging.getLogger('advanced_console_capture')
        self.global_capture_logger.setLevel(console_level)
        # Partager le même handler que console_logger
        self.global_capture_logger.addHandler(console_handler)
        self.global_capture_logger.propagate = False

        # Notifier que le système de capture est prêt
        module_logger.info(f"📝 Système de capture console.log activé (niveau: {self.console_log_level})")

        # 3. DEBUG LOGGER
        self.debug_logger = logging.getLogger(f'{self.experiment_name}_debug')
        self.debug_logger.setLevel(logging.DEBUG)

        debug_handler = logging.FileHandler(
            self.log_dir / "debug.log",
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            '%(asctime)s | STEP:%(step)s EP:%(episode)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        debug_handler.setFormatter(debug_formatter)
        self.debug_logger.addHandler(debug_handler)

        # Rediriger stdout vers console.log pour capturer les print()
        # Les print() seront traités comme logger.info()
        self._redirect_stdout()
        module_logger.info("Redirection stdout → console.log activée (print() = INFO)")

    def _redirect_stdout(self):
        """
        Redirige stdout pour capturer tout ce qui s'affiche

        Les print() sont traités comme logger.info()
        et respectent donc le niveau de log configuré
        """

        class TeeOutput:
            def __init__(self, redirect_logger, original_stream, min_level):
                self.redirect_logger = redirect_logger
                self.original = original_stream
                self.min_level = min_level  # Niveau minimum pour enregistrer

            def write(self, message):
                # Toujours ecrire dans le terminal (pour éviter blocages)
                self.original.write(message)

                # Enregistrer dans console.log seulement si niveau le permet
                if message.strip() and self.min_level <= logging.INFO:
                    self.redirect_logger.info(message.rstrip())

            def flush(self):
                self.original.flush()

        # Récupérer le niveau configuré
        console_level = getattr(logging, self.console_log_level.upper())
        sys.stdout = TeeOutput(self.console_logger, sys.stdout, console_level)
        # Ajouter aussi la redirection de stderr pour capturer les warnings
        sys.stderr = TeeOutput(self.console_logger, sys.stderr, console_level)

    def log_error(
            self,
            error: Exception,
            context: str = "",
    ):
        """
        Log une erreur avec traceback complet

        Args:
            error: Exception capturée
            context: Contexte de l'erreur
        """
        self.error_count += 1

        error_msg = f"{context}" if context else ""
        error_msg += f"Exception: {type(error).__name__}: {str(error)}"
        error_msg += "Traceback:"
        error_msg += "".join(traceback.format_exception(type(error), error, error.__traceback__))

        self.error_logger.error(
            error_msg,
            extra={'step': self.step_count}
        )

        # Debug log aussi
        self.debug_logger.error(
            f"ERROR in {context}: {error}",
            extra={'step': self.step_count, 'episode': self.episode_count}
        )

    def log_warning(self, message: str):
        """
        Log un avertissement
        """
        self.warning_count += 1
        self.debug_logger.warning(
            message,
            extra={'step': self.step_count, 'episode': self.episode_count}
        )

    def log_info(self, message: str):
        """
        Log une information (si niveau INFO/DEBUG activé)
        """
        self.console_logger.info(message)
        self.debug_logger.info(
            message,
            extra={'step': self.step_count, 'episode': self.episode_count}
        )

    def log_debug(self, message: str):
        """
        Log debug détaillé (si niveau DEBUG activé)
        """
        self.debug_logger.debug(
            message,
            extra={'step': self.step_count, 'episode': self.episode_count}
        )

    def should_log_periodic(
            self,
            interval: int = 1000,
    ) -> bool:
        """
        Helper pour logs périodiques

        Args:
            interval: Intervalle de steps

        Returns:
            True si on doit logger ce step

        Exemple:
            if training_logger.should_log_periodic(1000):
                logger.info(f"Step {self.total_steps}: ...")
        """
        return self.step_count % interval == 0

    def log_step_data(self, data: Dict[str, Any]):
        """
        Log les données d'un step (JSONL format)

        Args:
            data: Dictionnaire contenant :
                - step, episode, reward, action, hp, stamina, zone, etc.
        """
        self.step_count += 1

        # Ajouter métadonnées temporelles
        entry = {
            'timestamp': datetime.now().isoformat(),
            'real_time_elapsed': (datetime.now() - self.session_start).total_seconds(),
            'step': self.step_count,
            'episode': self.episode_count,
            **data
        }

        # Écrire en JSONL (une ligne par step)
        self.training_data_file.write(json.dumps(entry))
        self.training_data_file.flush()  # Force write immédiat

        # Debug log (seulement info importante)
        if self.step_count % 1 == 0:  # Tous les steps
            self.debug_logger.debug(
                f"Step summary - Reward: {data.get('reward', 0):.2f}, "
                f"HP: {data.get('hp', 0)}, Zone: {data.get('zone', 0)}",
                extra={'step': self.step_count, 'episode': self.episode_count}
            )

    def log_episode_end(self, episode_data: Dict[str, Any]):
        """
        Log la fin d'un épisode

        Args:
            episode_data: Données de l'épisode (reward totale, longueur, etc.)
        """
        self.episode_count += 1

        # Stocker les données pour calculer les stats plus tard
        self.episode_rewards.append(episode_data.get('total_reward', 0.0))
        self.episode_lengths.append(episode_data.get('length', 0))
        self.episode_hits.append(episode_data.get('hits', 0))
        self.episode_deaths.append(episode_data.get('deaths', 0))
        self.episode_zones.append(episode_data.get('zones_discovered', 0))

        self.debug_logger.info(
            f"Episode END - Total reward: {episode_data.get('total_reward', 0):.2f}, "
            f"Length: {episode_data.get('length', 0)}, "
            f"Hits: {episode_data.get('hits', 0)}, "
            f"Deaths: {episode_data.get('deaths', 0)}",
            extra={'step': self.step_count, 'episode': self.episode_count}
        )

        # Écrire aussi en JSONL avec marker
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'EPISODE_END',
            'episode': self.episode_count,
            **episode_data
        }
        self.training_data_file.write(json.dumps(entry))
        self.training_data_file.flush()

    def log_config(self, config: Dict[str, Any]):
        """
        Log la configuration de l'entraînement

        Args:
            config: Hyperparamètres, args, etc.
        """
        self.session_data['config'] = config

        self.debug_logger.info(
            f"Training config: {json.dumps(config, indent=2)}",
            extra={'step': 0, 'episode': 0}
        )

    def log_checkpoint(self, checkpoint_path: str, timesteps: int):
        """Log la sauvegarde d'un checkpoint"""
        self.debug_logger.info(
            f"Checkpoint saved: {checkpoint_path} (timesteps: {timesteps:,})",
            extra={'step': self.step_count, 'episode': self.episode_count}
        )

    def save_session_summary(self):
        """
        Sauvegarde le résumé de la session
        """
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['duration_seconds'] = (
                datetime.now() - self.session_start
        ).total_seconds()
        self.session_data['total_steps'] = self.step_count
        self.session_data['total_episodes'] = self.episode_count
        self.session_data['errors'] = self.error_count
        self.session_data['warnings'] = self.warning_count

        # Calcul des statistiques d'épisodes - Fournit des métriques utiles pour analyser la performance
        if self.episode_rewards:  # Si au moins 1 épisode terminé
            episode_stats = {
                'total_episodes': len(self.episode_rewards),

                # REWARDS
                'reward': {
                    'mean': float(np.mean(self.episode_rewards)),
                    'std': float(np.std(self.episode_rewards)),
                    'min': float(np.min(self.episode_rewards)),
                    'max': float(np.max(self.episode_rewards)),
                    'median': float(np.median(self.episode_rewards)),
                    'total': float(np.sum(self.episode_rewards)),
                    # Quartiles (25%, 75%)
                    'q25': float(np.percentile(self.episode_rewards, 25)),
                    'q75': float(np.percentile(self.episode_rewards, 75)),
                },

                # LONGUEURS D'ÉPISODES
                'length': {
                    'mean': float(np.mean(self.episode_lengths)),
                    'std': float(np.std(self.episode_lengths)),
                    'min': int(np.min(self.episode_lengths)),
                    'max': int(np.max(self.episode_lengths)),
                    'median': float(np.median(self.episode_lengths)),
                    'total_steps': int(np.sum(self.episode_lengths)),
                },

                # HITS (coups portés)
                'hits': {
                    'mean': float(np.mean(self.episode_hits)),
                    'std': float(np.std(self.episode_hits)),
                    'min': int(np.min(self.episode_hits)),
                    'max': int(np.max(self.episode_hits)),
                    'total': int(np.sum(self.episode_hits)),
                },

                # DEATHS (morts)
                'deaths': {
                    'mean': float(np.mean(self.episode_deaths)),
                    'std': float(np.std(self.episode_deaths)),
                    'min': int(np.min(self.episode_deaths)),
                    'max': int(np.max(self.episode_deaths)),
                    'total': int(np.sum(self.episode_deaths)),
                },

                # ZONES DÉCOUVERTES
                'zones_discovered': {
                    'mean': float(np.mean(self.episode_zones)),
                    'std': float(np.std(self.episode_zones)),
                    'min': int(np.min(self.episode_zones)),
                    'max': int(np.max(self.episode_zones)),
                },

                # PROGRESSION : Comparaison premiers vs derniers épisodes
                # RAISON : Permet de voir si l'IA s'améliore
                'progression': self._calculate_progression_stats()
            }

            self.session_data['episode_statistics'] = episode_stats
        else:
            self.session_data['episode_statistics'] = {
                'total_episodes': 0,
                'note': 'Aucun episode termine durant cette session'
            }

        module_logger.debug(f"Données session préparées:")
        module_logger.debug(f"Steps: {self.step_count}")
        module_logger.debug(f"Episodes: {self.episode_count}")
        module_logger.debug(f"Errors: {self.error_count}")

        summary_path = self.log_dir / "session_summary.json"
        module_logger.debug(f"Écriture dans: {summary_path}")

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
            module_logger.debug(f"Fichier écrit avec succès")
            module_logger.debug(f"📊 Résumé de session sauvegardé : {summary_path}")

            # Afficher config environnement dans console si disponible
            env_config = self.session_data.get('config', {}).get('environment')
            if env_config:
                module_logger.info(f"🎮 Configuration environnement :")
                module_logger.info(f"   Mode : {env_config.get('mode', 'unknown')}")
                module_logger.info(f"   Vision : {'true' if env_config.get('use_vision') else 'false'}")
                module_logger.info(f"   Mémoire : {'true' if env_config.get('use_memory') else 'false'}")
                if env_config.get('multi_instance'):
                    module_logger.info(f"   Multi-instance : {env_config.get('num_instances')} instances")

        except Exception as save_session_summary_error:
            module_logger.debug(f"Erreur écriture: {save_session_summary_error}")
            traceback.print_exc()

    def _calculate_progression_stats(self) -> Dict[str, Any]:
        """
        Compare les premiers 20% et derniers 20% des épisodes
        pour mesurer la progression

        Returns:
            Dict avec comparaison début vs fin
        """

        if len(self.episode_rewards) < 5:
            # Pas assez d'épisodes pour calculer progression
            return {
                'note': 'Pas assez d\'épisodes pour calculer la progression (minimum 5)'
            }

        # Découper en 20% premiers et 20% derniers
        n_episodes = len(self.episode_rewards)
        split_size = max(1, n_episodes // 5)  # 20% mais minimum 1 épisode

        first_rewards = self.episode_rewards[:split_size]
        last_rewards = self.episode_rewards[-split_size:]

        first_lengths = self.episode_lengths[:split_size]
        last_lengths = self.episode_lengths[-split_size:]

        first_hits = self.episode_hits[:split_size]
        last_hits = self.episode_hits[-split_size:]

        # Cast explicite en float() pour éviter les warnings de typage
        # RAISON : np.mean() retourne numpy.floating qui n'est pas exactement float
        mean_first_rewards = float(np.mean(first_rewards))
        mean_last_rewards = float(np.mean(last_rewards))
        mean_first_lengths = float(np.mean(first_lengths))
        mean_last_lengths = float(np.mean(last_lengths))
        mean_first_hits = float(np.mean(first_hits))
        mean_last_hits = float(np.mean(last_hits))

        # Calculer les différences
        reward_improvement = mean_last_rewards - mean_first_rewards
        length_change = mean_last_lengths - mean_first_lengths
        hits_improvement = mean_last_hits - mean_first_hits

        # Calculer le pourcentage d'amélioration reward
        # Protection contre division par zéro
        if mean_first_rewards != 0:
            improvement_percent = (reward_improvement / abs(mean_first_rewards)) * 100
        else:
            improvement_percent = 0.0

        return {
            'episodes_compared': {
                'first': split_size,
                'last': split_size,
                'total': n_episodes
            },
            'reward': {
                'first_episodes_mean': mean_first_rewards,
                'last_episodes_mean': mean_last_rewards,
                'improvement': reward_improvement,
                'improvement_percent': improvement_percent
            },
            'length': {
                'first_episodes_mean': mean_first_lengths,
                'last_episodes_mean': mean_last_lengths,
                'change': length_change
            },
            'hits': {
                'first_episodes_mean': mean_first_hits,
                'last_episodes_mean': mean_last_hits,
                'improvement': hits_improvement
            }
        }

    def close(self):
        """
        Ferme tous les fichiers et loggers
        """
        module_logger.info("📝 Fermeture du système de logging...")

        # Sauvegarder résumé
        self.save_session_summary()

        # Fermer fichiers
        self.training_data_file.close()

        # Fermer handlers
        for close_logger in [self.error_logger, self.console_logger, self.debug_logger]:
            for handler in close_logger.handlers[:]:
                handler.close()
                close_logger.removeHandler(handler)

        # Nettoyer aussi le logger global de capture
        if hasattr(self, 'global_capture_logger'):
            for handler in self.global_capture_logger.handlers[:]:
                handler.close()
                self.global_capture_logger.removeHandler(handler)

        # Restaurer stdout
        sys.stdout = sys.__stdout__

        module_logger.info(f"Logs sauvegardés dans : {self.log_dir}")
        module_logger.info(f"Steps: {self.step_count:,}, Episodes: {self.episode_count}")
        module_logger.info(f"Erreurs: {self.error_count}, Warnings: {self.warning_count}")


# ============================================================
# INTÉGRATION AVEC LE CALLBACK
# ============================================================

class LoggingCallback(BaseCallback):
    """
    Callback pour intégrer le TrainingLogger avec SB3
    """

    def __init__(self, training_logger  : TrainingLogger, verbose=0):
        super().__init__(verbose)
        self.training_logger   = training_logger
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_data = {}

    def _on_step(self) -> bool:
        """
        Appelé à chaque step
        """

        # Extraire infos
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
        else:
            info = {}

        # Données du step
        step_data = {
            'reward': float(self.locals.get('rewards', [0])[0]),
            'action': int(self.locals.get('actions', [0])[0]),
            'hp': info.get('hp', 0) or 0,
            'stamina': info.get('stamina', 0) or 0,
            'zone': info.get('current_zone', 0) or 0,
            'player_x': info.get('player_x', 0.0) or 0.0,
            'player_y': info.get('player_y', 0.0) or 0.0,
            'player_z': info.get('player_z', 0.0) or 0.0,
            'hit_count': info.get('hit_count', 0) or 0,
            'death_count': info.get('death_count', 0) or 0,
            'total_cubes': info.get('total_cubes', 0) or 0,
            'zones_discovered': info.get('zones_discovered', 0) or 0,
            'in_combat': info.get('in_combat', False),
            'monsters_present': info.get('monsters_present', False),
            'reward_breakdown': info.get('reward_breakdown', {}),
        }

        # Logger le step
        self.training_logger.log_step_data(step_data)

        # Accumuler pour épisode
        self.episode_reward += step_data['reward']
        self.episode_length += 1
        self.episode_data = step_data.copy()

        # Fin d'épisode ?
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            episode_summary = {
                'total_reward': self.episode_reward,
                'length': self.episode_length,
                'final_hp': self.episode_data.get('hp', 0),
                'hits': self.episode_data.get('hit_count', 0),
                'deaths': self.episode_data.get('death_count', 0),
                'zones_discovered': self.episode_data.get('zones_discovered', 0),
                'total_cubes': self.episode_data.get('total_cubes', 0),
            }

            self.training_logger.log_episode_end(episode_summary)

            # Reset
            self.episode_reward = 0.0
            self.episode_length = 0
            self.episode_data = {}

        return True

# ============================================================
# UTILITAIRE : ANALYSER LES LOGS
# ============================================================

"""
Pour analyser une session après l'entraînement :

python -c "
from utils.advanced_logging import LogAnalyzer
LogAnalyzer.analyze_session('./logs/mh_20250105_143015/20250105_143022')
"
"""

class LogAnalyzer:
    """
    Utilitaire pour analyser les fichiers de logs
    """

    @staticmethod
    def load_training_data(jsonl_path: str) -> list:
        """Charge les données training_data.jsonl"""
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    @staticmethod
    def analyze_session(log_dir: str):
        """
        Analyse complète d'une session

        Args:
            log_dir: Chemin vers le dossier de logs
        """
        log_path = Path(log_dir)

        module_logger.info(f"📊 ANALYSE SESSION : {log_path.name}")
        module_logger.info("=" * 70)

        # 1. Résumé
        summary_file = log_path / "session_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            module_logger.info(f"⏱️  Durée : {summary['duration_seconds']:.0f}s "f"({summary['duration_seconds'] / 60:.1f} min)")
            module_logger.info(f"📈 Steps : {summary['total_steps']:,}")
            module_logger.info(f"🎮 Episodes : {summary['total_episodes']:,}")
            module_logger.info(f"❌ Erreurs : {summary['errors']}")
            module_logger.info(f"⚠️  Warnings : {summary['warnings']}")

        # 2. Erreurs
        errors_file = log_path / "errors.log"
        if errors_file.exists():
            with open(errors_file, 'r', encoding='utf-8') as f:
                errors = f.readlines()

            if errors:
                module_logger.error(f"❌ ERREURS ({len(errors)} lignes) :")
                module_logger.error("Voir errors.log pour détails")

        # 3. Training data stats
        training_file = log_path / "training_data.jsonl"
        if training_file.exists():
            data = LogAnalyzer.load_training_data(str(training_file))

            # Filtrer épisodes
            episodes = [d for d in data if d.get('type') == 'EPISODE_END']
            steps = [d for d in data if 'reward' in d and d.get('type') != 'EPISODE_END']

            if episodes:
                avg_reward = sum(j['total_reward'] for j in episodes) / len(episodes)
                avg_length = sum(k['length'] for k in episodes) / len(episodes)

                module_logger.info(f"📊 STATISTIQUES ÉPISODES :")
                module_logger.info(f"Reward moyenne : {avg_reward:.2f}")
                module_logger.info(f"Longueur moyenne : {avg_length:.0f} steps")

            if steps:
                total_reward = sum(s['reward'] for s in steps)
                module_logger.info(f"💰 REWARD TOTALE : {total_reward:.2f}")

        module_logger.info("" + "=" * 70)


if __name__ == "__main__":
    # Test du système
    print("🧪 Test du système de logging...\n")

    logger = TrainingLogger("test_experiment")

    # Simuler quelques steps
    for i in range(5):
        logger.log_step_data({
            'reward': 0.5,
            'action': 1,
            'hp': 100 - i * 10,
            'zone': 0
        })

    # Simuler erreur
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error(e, "Test context")

    # Fin épisode
    logger.log_episode_end({
        'total_reward': 2.5,
        'length': 5,
        'hits': 3
    })

    logger.close()

    print("\n✅ Test terminé - Vérifier le dossier logs/")