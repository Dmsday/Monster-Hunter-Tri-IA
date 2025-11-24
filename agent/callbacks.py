"""
Callbacks minimaux pour Stable-Baselines3
Version nettoyée : Seulement ce qui est réellement utilisé dans train.py

NOTE: train.py utilise directement CheckpointCallback de SB3, pas celui-ci.
      Ce fichier est donc OPTIONNEL et peut être supprimé.
"""

import os
import numpy as np
import time
from stable_baselines3.common.callbacks import BaseCallback

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from utils.module_logger import get_module_logger
logger = get_module_logger('callbacks')

class TrainingProgressCallback(BaseCallback):
    """
    Callback simple pour afficher la progression de l'entraînement
    Alternative à la barre de progression SB3 avec infos custom
    """

    def __init__(
            self,
            total_timesteps: int,
            print_freq: int = 1000,
            verbose: int = 1
    ):
        """
        Args:
            total_timesteps: Nombre total de timesteps prévus
            print_freq: Fréquence d'affichage (en steps)
            verbose: Niveau de verbosité
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.start_time = None
        self.last_print_step = 0

        # Stats
        self.episode_count = 0
        self.recent_rewards = []
        self.recent_lengths = []

    def _on_training_start(self):
        """Appelé au début de l'entraînement"""
        self.start_time = time.time()

        if self.verbose > 0:
            logger.info("" + "=" * 70)
            logger.info("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
            logger.info("=" * 70)
            logger.info(f"Timesteps prévus: {self.total_timesteps:,}")
            logger.info(f"Fréquence d'affichage: tous les {self.print_freq} steps")
            logger.info("=" * 70 + "")

    def _on_step(self) -> bool:
        """Appelé à chaque step"""

        # Collecter infos des épisodes terminés
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_count += 1
                    self.recent_rewards.append(info['episode']['r'])
                    self.recent_lengths.append(info['episode']['l'])

                    # Garder seulement les 100 derniers
                    if len(self.recent_rewards) > 100:
                        self.recent_rewards.pop(0)
                        self.recent_lengths.pop(0)

        # Afficher périodiquement
        if self.num_timesteps - self.last_print_step >= self.print_freq:
            self._print_progress()
            self.last_print_step = self.num_timesteps

        return True

    def _print_progress(self):
        """Affiche la progression actuelle"""
        if not self.start_time:
            return

        # Calculs
        elapsed = time.time() - self.start_time
        progress = self.num_timesteps / self.total_timesteps * 100

        # ETA
        if self.num_timesteps > 0:
            time_per_step = elapsed / self.num_timesteps
            remaining_steps = self.total_timesteps - self.num_timesteps
            eta_seconds = remaining_steps * time_per_step
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        else:
            eta_str = "??:??:??"

        # Stats récentes
        mean_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        mean_length = np.mean(self.recent_lengths) if self.recent_lengths else 0.0

        # Format temps écoulé
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        # Affichage
        logger.info(f"[{self.num_timesteps:8,}/{self.total_timesteps:8,}] "
              f"({progress:5.1f}%) | "
              f"⏱️  {elapsed_str} | "
              f"ETA: {eta_str}")

        if self.episode_count > 0:
            logger.info(f"   📊 Episodes: {self.episode_count:4d} | "
                  f"Reward moy: {mean_reward:+7.2f} | "
                  f"Length moy: {mean_length:6.1f}")

    def _on_training_end(self):
        """Appelé à la fin de l'entraînement"""
        if self.verbose > 0 and self.start_time:
            elapsed = time.time() - self.start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

            logger.info("" + "=" * 70)
            logger.info("ENTRAÎNEMENT TERMINÉ")
            logger.info("=" * 70)
            logger.info(f"Durée totale: {elapsed_str}")
            logger.info(f"Steps effectués: {self.num_timesteps:,}")
            logger.info(f"Episodes complétés: {self.episode_count}")

            if self.recent_rewards:
                final_mean = np.mean(self.recent_rewards)
                final_max = np.max(self.recent_rewards)
                final_min = np.min(self.recent_rewards)
                logger.info(f"Rewards finales (derniers 100 épisodes):")
                logger.info(f"   Moyenne: {final_mean:+.2f}")
                logger.info(f"   Max: {final_max:+.2f}")
                logger.info(f"   Min: {final_min:+.2f}")

            logger.info("=" * 70 + "")


class BestModelSaver(BaseCallback):
    """
    Sauvegarde automatique du meilleur modèle basé sur la reward moyenne
    Plus simple que BestModelCallback (pas d'évaluation séparée)
    """

    def __init__(
            self,
            save_path: str,
            name_prefix: str = "best_model",
            check_freq: int = 10000,
            window_size: int = 10,
            verbose: int = 1
    ):
        """
        Args:
            save_path: Dossier de sauvegarde
            name_prefix: Préfixe du nom de fichier
            check_freq: Fréquence de vérification (en steps)
            window_size: Nombre d'épisodes pour calculer la moyenne
            verbose: Verbosité
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.check_freq = check_freq
        self.window_size = window_size

        self.best_mean_reward = -np.inf
        self.last_check_step = 0
        self.recent_rewards = []

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Collecter rewards
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.recent_rewards.append(info['episode']['r'])

                    # Limiter la taille
                    if len(self.recent_rewards) > self.window_size:
                        self.recent_rewards.pop(0)

        # Vérifier périodiquement
        if self.num_timesteps - self.last_check_step >= self.check_freq:
            self._check_and_save()
            self.last_check_step = self.num_timesteps

        return True

    def _check_and_save(self):
        """Vérifie et sauvegarde si meilleur modèle"""
        if len(self.recent_rewards) < self.window_size:
            return  # Pas assez de données

        mean_reward = np.mean(self.recent_rewards)

        # Nouveau record ?
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

            # Sauvegarder
            model_path = os.path.join(self.save_path, f"{self.name_prefix}")
            self.model.save(model_path)

            if self.verbose > 0:
                logger.info(f"🏆 NOUVEAU MEILLEUR MODÈLE!")
                logger.info(f"   Reward moyenne: {mean_reward:+.2f}")
                logger.info(f"   Sauvegardé: {model_path}.zip")


class EpisodeStatsLogger(BaseCallback):
    """
    Log les statistiques détaillées de chaque épisode dans un fichier JSON
    Utile pour analyse post-training
    """

    def __init__(
            self,
            save_path: str,
            verbose: int = 0
    ):
        """
        Args:
            save_path: Chemin du fichier JSON de sortie
            verbose: Verbosité
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_stats = []

        # Créer le dossier parent
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _on_step(self) -> bool:
        # Collecter stats des épisodes terminés
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_data = {
                        'timestep': self.num_timesteps,
                        'reward': float(info['episode']['r']),
                        'length': int(info['episode']['l']),
                        'time_seconds': float(info['episode'].get('t', 0))
                    }

                    # Ajouter stats custom Monster Hunter si disponibles
                    custom_keys = [
                        'hp', 'stamina', 'hit_count', 'death_count',
                        'current_zone', 'damage_dealt', 'damage_taken',
                        'total_distance', 'zones_discovered'
                    ]

                    for key in custom_keys:
                        if key in info:
                            episode_data[key] = info[key]

                    self.episode_stats.append(episode_data)

        return True

    def _on_training_end(self):
        """Sauvegarde toutes les stats à la fin"""
        if not self.episode_stats:
            return

        import json

        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.episode_stats, f, indent=2)

        if self.verbose > 0:
            logger.info(f"\n📊 {len(self.episode_stats)} épisodes sauvegardés dans: {self.save_path}")


# ============================================================
# FONCTION HELPER POUR CRÉER UN SET DE CALLBACKS STANDARD
# ============================================================

def create_standard_callbacks(
        save_path: str,
        total_timesteps: int,
        save_freq: int = 10000,
        verbose: int = 1
) -> list:
    """
    Crée un ensemble de callbacks standard pour l'entraînement

    Args:
        save_path: Dossier de sauvegarde
        total_timesteps: Nombre total de timesteps
        save_freq: Fréquence de sauvegarde du meilleur modèle
        verbose: Verbosité

    Returns:
        Liste de callbacks prêts à l'emploi
    """
    callbacks = [
        # Progression
        TrainingProgressCallback(
            total_timesteps=total_timesteps,
            print_freq=1000,
            verbose=verbose
        ),

        # Meilleur modèle
        BestModelSaver(
            save_path=save_path,
            name_prefix="best_model",
            check_freq=save_freq,
            window_size=10,
            verbose=verbose
        ),

        # Stats épisodes
        EpisodeStatsLogger(
            save_path=os.path.join(save_path, "episode_stats.json"),
            verbose=verbose
        )
    ]

    return callbacks


# ============================================================
# EXEMPLE D'UTILISATION
# ============================================================

if __name__ == "__main__":
    """
    Test rapide des callbacks

    NOTE: Ces callbacks sont OPTIONNELS.
    train.py utilise directement CheckpointCallback de SB3.

    Pour utiliser ces callbacks dans train.py:

    1. Importer:
       from callbacks import create_standard_callbacks

    2. Créer les callbacks:
       my_callbacks = create_standard_callbacks(
           save_path=models_dir,
           total_timesteps=args.timesteps,
           verbose=1
       )

    3. Ajouter à la liste des callbacks:
       callbacks = [checkpoint_callback] + my_callbacks

    4. Passer à learn():
       agent.learn(
           total_timesteps=args.timesteps,
           callback=callbacks
       )
    """

    print("=" * 70)
    print("📦 CALLBACKS MINIMAUX")
    print("=" * 70)
    print("\n✅ Callbacks disponibles:\n")

    print("1. TrainingProgressCallback")
    print("   → Affiche la progression avec ETA et stats")
    print("   → Alternative à la barre de progression SB3")

    print("\n2. BestModelSaver")
    print("   → Sauvegarde automatique du meilleur modèle")
    print("   → Basé sur reward moyenne glissante")

    print("\n3. EpisodeStatsLogger")
    print("   → Enregistre toutes les stats dans episode_stats.json")
    print("   → Utile pour analyse post-training")

    print("\n4. create_standard_callbacks()")
    print("   → Fonction helper qui combine les 3 callbacks")

    print("\n" + "=" * 70)
    print("💡 UTILISATION")
    print("=" * 70)
    print("""
# Dans train.py, remplacer:
callbacks = [checkpoint_callback]

# Par:
from callbacks import create_standard_callbacks

my_callbacks = create_standard_callbacks(
    save_path=models_dir,
    total_timesteps=args.timesteps
)

callbacks = [checkpoint_callback] + my_callbacks

agent.learn(total_timesteps=args.timesteps, callback=callbacks)
""")

    print("=" * 70)
    print("⚠️  NOTE IMPORTANTE")
    print("=" * 70)
    print("""
Ce fichier est OPTIONNEL car train.py fonctionne déjà avec
les callbacks natifs de Stable-Baselines3.
""")

    # Test action space
    print("" + "=" * 70)
    print("🎮 VÉRIFICATION ACTION SPACE")
    print("=" * 70)
    print("\nSi tu vois des erreurs 'action out of bounds', vérifie :")
    print("1. mh_env.py : action_space = spaces.Discrete(25)")
    print("2. controller.py : Actions 0-24 implémentées")
    print("3. reward_calculator.py : Actions 23-24 gérées (optionnel)")