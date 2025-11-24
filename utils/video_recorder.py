"""
Enregistrement vidéo des épisodes
Capture les frames et crée des vidéos MP4 pour analyse
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import List


class VideoRecorder:
    """
    Enregistre les frames de jeu en vidéo
    Utile pour visualiser les performances de l'IA
    """

    def __init__(
            self,
            output_dir: str = "./videos/",
            fps: int = 30,
            codec: str = 'mp4v',
            quality: int = 5,  # 0-10, 10 = meilleure qualité
            record_every_n_episodes: int = 10
    ):
        """
        Args:
            output_dir: Dossier de sortie des vidéos
            fps: FPS de la vidéo
            codec: Codec vidéo ('mp4v', 'XVID', 'H264')
            quality: Qualité de compression (0-10)
            record_every_n_episodes: Enregistrer 1 épisode tous les N
        """
        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.record_every_n_episodes = record_every_n_episodes

        # État de l'enregistrement
        self.is_recording = False
        self.current_video_writer = None
        self.current_video_path = None
        self.frame_buffer = []
        self.episode_count = 0

        # Créer dossier de sortie
        os.makedirs(output_dir, exist_ok=True)

        print(f"🎥 Video Recorder initialisé")
        print(f"   Sortie: {output_dir}")
        print(f"   FPS: {fps}, Codec: {codec}")

    def should_record_episode(self, episode: int) -> bool:
        """
        Détermine si cet épisode doit être enregistré

        Args:
            episode: Numéro de l'épisode

        Returns:
            True si on doit enregistrer
        """
        # Toujours enregistrer le premier épisode
        if episode == 1:
            return True

        # Enregistrer 1 épisode tous les N
        return episode % self.record_every_n_episodes == 0

    def start_recording(self, episode: int, experiment_name: str = "mh"):
        """
        Commence l'enregistrement d'un épisode

        Args:
            episode: Numéro de l'épisode
            experiment_name: Nom de l'expérience
        """
        if self.is_recording:
            self.stop_recording()

        # Générer nom de fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_ep{episode:04d}_{timestamp}.mp4"
        self.current_video_path = os.path.join(self.output_dir, filename)

        self.is_recording = True
        self.frame_buffer = []

        print(f"🔴 Enregistrement démarré: {filename}")

    def add_frame(self, frame: np.ndarray, info: dict = None):
        """
        Ajoute une frame à l'enregistrement

        Args:
            frame: Image RGB (H, W, 3)
            info: Infos à afficher sur la frame (optionnel)
        """
        if not self.is_recording:
            return

        # Copier pour ne pas modifier l'original
        frame = frame.copy()

        # Ajouter infos textuelles si fournies
        if info:
            frame = self._add_info_overlay(frame, info)

        # Ajouter au buffer
        self.frame_buffer.append(frame)

    def _add_info_overlay(self, frame: np.ndarray, info: dict) -> np.ndarray:
        """
        Ajoute un overlay texte sur la frame

        Args:
            frame: Image RGB
            info: Dict avec les infos à afficher

        Returns:
            Frame avec overlay
        """
        # Convertir RGB -> BGR pour OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Position du texte
        y = 30
        line_height = 25

        # Style du texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # Blanc
        thickness = 2

        # Fond noir pour meilleure lisibilité
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (5, 5), (300, y + len(info) * line_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

        # Afficher chaque info
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(
                frame_bgr,
                text,
                (10, y),
                font,
                font_scale,
                color,
                thickness
            )
            y += line_height

        # Reconvertir BGR -> RGB
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def stop_recording(self, save: bool = True):
        """
        Arrête l'enregistrement et sauvegarde la vidéo

        Args:
            save: Si False, abandonne la vidéo
        """
        if not self.is_recording:
            return

        self.is_recording = False

        if not save or len(self.frame_buffer) == 0:
            print(f"⏹️  Enregistrement abandonné (0 frames)")
            self.frame_buffer = []
            return

        # Sauvegarder la vidéo
        self._write_video()

        print(f"✅ Vidéo sauvegardée: {self.current_video_path}")
        print(f"   {len(self.frame_buffer)} frames, durée: {len(self.frame_buffer) / self.fps:.1f}s")

        # Nettoyer
        self.frame_buffer = []
        self.current_video_writer = None

    def _write_video(self):
        """Écrit le buffer de frames dans un fichier vidéo"""
        if len(self.frame_buffer) == 0:
            return

        # Obtenir dimensions de la première frame
        height, width = self.frame_buffer[0].shape[:2]

        # Créer VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        video_writer = cv2.VideoWriter(
            self.current_video_path,
            fourcc,
            self.fps,
            (width, height)
        )

        if not video_writer.isOpened():
            print(f"❌ Impossible de créer la vidéo: {self.current_video_path}")
            return

        # Écrire toutes les frames
        for frame in self.frame_buffer:
            # OpenCV attend BGR, pas RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        # Libérer
        video_writer.release()

    def record_episode(
            self,
            env,
            agent,
            episode: int,
            max_steps: int = 1000,
            deterministic: bool = True
    ):
        """
        Enregistre un épisode complet

        Args:
            env: Environnement Gym
            agent: Agent PPO
            episode: Numéro de l'épisode
            max_steps: Nombre max de steps
            deterministic: Actions déterministes
        """
        self.start_recording(episode)

        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done and step < max_steps:
            # Prédire action
            action, _ = agent.predict(obs, deterministic=deterministic)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step += 1

            # Capturer frame
            frame = env.render()  # Doit retourner RGB array

            if frame is not None:
                # Infos à afficher
                display_info = {
                    'Step': step,
                    'Reward': f"{episode_reward:.1f}",
                    'HP': info.get('hp', 'N/A'),
                    'Stamina': info.get('stamina', 'N/A')
                }

                self.add_frame(frame, display_info)

        self.stop_recording()

        return episode_reward, step

    def create_comparison_video(
            self,
            video_paths: List[str],
            output_path: str,
            labels: List[str] = None
    ):
        """
        Crée une vidéo côte-à-côte pour comparer plusieurs runs

        Args:
            video_paths: Liste des chemins vidéo
            output_path: Chemin de sortie
            labels: Labels pour chaque vidéo
        """
        print("🎬 Création vidéo de comparaison...")

        # Ouvrir toutes les vidéos
        captures = [cv2.VideoCapture(path) for path in video_paths]

        if not all(cap.isOpened() for cap in captures):
            print("❌ Impossible d'ouvrir une ou plusieurs vidéos")
            return

        # Obtenir dimensions
        widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in captures]
        heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in captures]

        # Vidéo de sortie (côte à côte)
        out_width = sum(widths)
        out_height = max(heights)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (out_width, out_height))

        frame_count = 0

        while True:
            frames = []
            all_valid = True

            # Lire une frame de chaque vidéo
            for cap in captures:
                ret, frame = cap.read()
                if not ret:
                    all_valid = False
                    break
                frames.append(frame)

            if not all_valid:
                break

            # Concatener horizontalement
            combined = np.hstack(frames)

            # Ajouter labels si fournis
            if labels:
                for i, label in enumerate(labels):
                    x_offset = sum(widths[:i]) + 10
                    cv2.putText(
                        combined,
                        label,
                        (x_offset, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

            out.write(combined)
            frame_count += 1

        # Libérer
        for cap in captures:
            cap.release()
        out.release()

        print(f"✅ Vidéo de comparaison créée: {output_path}")
        print(f"   {frame_count} frames")

    def cleanup_old_videos(self, keep_last_n: int = 20):
        """
        Supprime les vieilles vidéos pour économiser l'espace

        Args:
            keep_last_n: Nombre de vidéos récentes à garder
        """
        # Lister toutes les vidéos
        videos = [
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.endswith('.mp4')
        ]

        # Trier par date de modification
        videos.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Supprimer les anciennes
        for video in videos[keep_last_n:]:
            os.remove(video)
            print(f"🗑️  Supprimé: {os.path.basename(video)}")

        if len(videos) > keep_last_n:
            print(f"✅ Nettoyage: {len(videos) - keep_last_n} vidéos supprimées")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Test du Video Recorder\n")

    recorder = VideoRecorder(
        output_dir="./test_videos/",
        fps=30
    )

    # Simuler enregistrement
    print("📹 Simulation d'enregistrement...\n")

    recorder.start_recording(episode=1, experiment_name="test")

    # Créer frames de test
    for i in range(90):  # 3 secondes à 30 FPS
        # Frame aléatoire colorée
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Infos de test
        info = {
            'Frame': i,
            'HP': 100 - i,
            'Stamina': 80
        }

        recorder.add_frame(frame, info)

    recorder.stop_recording()

    print("\n✅ Test réussi!")
    print(f"📂 Vérifie la vidéo dans: {recorder.output_dir}")