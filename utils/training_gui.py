"""
═══════════════════════════════════════════════════════════════════════════════
                MONSTER HUNTER TRI – TRAINING GUI OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

MODULE: training_gui.py
VERSION: 3.0
AUTHOR: DR.
DATE: 2025-11-05

DESCRIPTION :
    Multi-window GUI to monitor AI training in Monster Hunter Tri (Dolphin).
    Live display of:
        • Core stats (HP, stamina, reward, zone…)
        • Episode graphs (reward, length, hits)
        • 3D exploration map with markers
        • Player data (position, inventory, orientation)
        • Reward breakdown
        • Zone statistics (monsters, HP, exploration)

WINDOW STRUCTURE :
    Main window: Stats panel + 3 matplotlib plots
        Additional windows (collapsible):
            • Player window (XYZ, orientation, inventory)
            • Rewards window (categories, top gains/losses)
            • Map window (3D map + tooltips)

KEY FEATURES:
    • Runs in a separate, non-blocking thread
    • Auto-save window positions/sizes (gui_config.json)
    • Interactive 3D map (colored cubes, markers, auto-zoom)
    • Rolling averages for rewards (10s / 300 steps)
    • Zone monster detection + exploration tracking

DEPENDENCIES:
    tkinter, matplotlib, numpy, threading, json, deque

TYPE CHECKER FALSE POSITIVES (SHORT EXPLANATION):
    Several `# type: ignore` markers are intentional due to known stub issues:

    1) tkinter.pack(side=…, fill=…)
       → tk.LEFT, tk.RIGHT, etc. *are strings*, but type checkers still warn.

    2) tkinter.after(func)
       → after() accepts callables without args, but stubs expect *args.

    Both cases run correctly; ignores suppress cosmetic warnings only.

THREAD SAFETY:
    • GUI runs in daemon thread
    • Updates done via after() (Tk-safe)
    • Config save protected against TclError
    • winfo_exists() checks before updating widgets

PERFORMANCE:
    • Graphs updated every 500 ms
    • 3D map reduces cube count (radius 2500)
    • Bounded history (100 episodes, 300 rewards)
    • Uses non-interactive Matplotlib backend ("Agg")

GENERATED FILES:
    • gui_config.json — window layout persistence

═══════════════════════════════════════════════════════════════════════════════
"""
# ============================================================================
# IMPORTS SYSTÈME ET UTILITAIRES
# ============================================================================
import os          # Gestion fichiers (config JSON)
import time        # Gestion du temps (timestamps, délais)
import json        # Sauvegarde/chargement config GUI
import threading   # Thread séparé pour GUI non-bloquante
import numpy as np # Calculs mathématiques (moyennes, distances 3D)

# ============================================================================
# INTERFACE GRAPHIQUE TKINTER
# ============================================================================
import tkinter as tk  # Bibliothèque GUI principale

# ============================================================================
# MATPLOTLIB (GRAPHIQUES ET CARTE 3D)
# ============================================================================
# Backend matplotlib thread-safe
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif (thread-safe, évite conflits)

from typing import Dict, Any                                     # Type hints
from collections import deque                                    # Historique à taille fixe
from matplotlib.figure import Figure                             # Création de figures matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection          # Cubes 3D pour carte exploration
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Intégration matplotlib dans Tkinter

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from utils.module_logger import get_module_logger
logger = get_module_logger('training_gui')


class TrainingGUI:
    """
    Interface graphique
    """

    def __init__(self, title="Monster Hunter IA - Entraînement"):
        self.title = title
        self.window = None
        # New config location
        config_dir = os.path.join(".", "config", "user")
        os.makedirs(config_dir, exist_ok=True)
        self.config_file = os.path.join(config_dir, "gui_config.json")

        # Migrate old config if exists in project root
        old_config = "gui_config.json"
        if os.path.exists(old_config) and not os.path.exists(self.config_file):
            import shutil
            logger.warning(f"GUI config detected in root project")
            shutil.move(old_config, self.config_file)
            logger.warning(f"Migrated GUI config to {self.config_file}")

        self.config = self._load_config()
        self.running = False
        self.stop_requested = False
        self._closing = False

        # Données
        self.current_stats = {
            'episode': 0,
            'step': 0,
            'total_steps': 0,
            'reward': 0.0,
            'hp': 100.0,
            'stamina': 100.0,
            'hits': 0,
            'deaths': 0,
            'zone': 0,
            'action': 0,
            'player_x': 0.0,
            'player_y': 0.0,
            'player_z': 0.0,
            'orientation': 0.0,
            'money': 0,
            'distance': 0.0,
            'game_menu_open_count': 0,
            'sharpness': 150,
            'quest_time': 90000,
            'episode_reward': 0.0,  # Reward de l'épisode en cours
            'inventory': [],

            # Initialisation des breakdowns
            'reward_breakdown': {
                'survival': 0.0,
                'combat': 0.0,
                'health': 0.0,
                'exploration': 0.0,
                'penalties': 0.0,
                'zone_change': 0.0,
                'defensive_actions': 0.0,
                'oxygen': 0.0,
                'monster_zone': 0.0,
                'death': 0.0,
                'damage_taken': 0.0,
                'monster_hit': 0.0,
                'hit': 0.0,
                'camp_penalty': 0.0,
                'menu_penalty': 0.0,
                'other': 0.0
            },
            'reward_breakdown_detailed': {}
        }

        # Cache pour affichage rewards (garde dernière valeur non-nulle)
        self.reward_display_cache = {}
        self.reward_display_timestamps = {}
        self.reward_cache_duration = 2.0  # Durée en secondes

        # Historique
        self.history_size = 100
        self.episode_history = deque(maxlen=self.history_size)
        self.reward_history = deque(maxlen=self.history_size)
        self.length_history = deque(maxlen=self.history_size)
        self.hits_history = deque(maxlen=self.history_size)
        self.reward_breakdown_history = deque(maxlen=300)
        self.reward_breakdown_detailed_history = deque(maxlen=300)

        # Widgets
        self.stat_labels = {}
        self.fig = None
        self.canvas = None
        self.axes = []

        # Fenêtres dépliables
        self.player_window = None
        self.rewards_window = None
        self.map_window = None
        self.zone_stats_window = None

        # Attributs carte 3D (initialisés ici pour éviter AttributeError)
        self.map_cube_data = []
        self.map_annotation = None
        self.map_ax = None
        self.map_canvas = None
        self.map_zone_filter = None
        self.map_stats_label = None
        self.map_auto_refresh = True

        # Thread
        self.update_thread = None
        self.start_time = time.time()

    def start(self):
        """Démarre l'interface dans un thread séparé"""
        if self.running:
            return

        self.running = True
        self.stop_requested = False
        self.start_time = time.time()

        self.update_thread = threading.Thread(target=self._run_gui, daemon=True)
        self.update_thread.start()

        logger.info("Interface graphique démarrée")

    def _load_config(self) -> dict:
        """Charge la config depuis le fichier"""
        default_config = {
            'main_window': {'geometry': '1400x900+100+100'},
            'player_window': {
                'open': False,
                'geometry': '500x700+1500+100'
            },
            'rewards_window': {
                'open': False,
                'geometry': '600x700+1500+100'
            },
            'map_window': {
                'open': False,
                'geometry': '800x700+1500+100'
            },
            'zone_stats_window': {
                'open': False,
                'geometry': '600x500+1500+100'
            }
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge avec default
                    for key in default_config:
                        if key in loaded:
                            default_config[key].update(loaded[key])
                    return default_config
            except Exception as e:
                logger.error(f"Erreur lecture config: {e}")

        return default_config

    def _save_config(self):
        """
        Save config (thread-safe with shutdown protection)
        """
        # Don't save during shutdown - window may be partially destroyed
        if self._closing:
            logger.debug("Skipping config save during shutdown")
            return

        try:
            # Get current geometries ONLY if window valid
            try:
                if self.window and self.window.winfo_exists():
                    self.config['main_window']['geometry'] = self.window.geometry()
            except tk.TclError:
                pass

            try:
                if self.player_window and self.player_window.winfo_exists():
                    self.config['player_window']['open'] = True
                    self.config['player_window']['geometry'] = self.player_window.geometry()
                else:
                    self.config['player_window']['open'] = False
            except tk.TclError:
                self.config['player_window']['open'] = False

            try:
                if self.rewards_window and self.rewards_window.winfo_exists():
                    self.config['rewards_window']['open'] = True
                    self.config['rewards_window']['geometry'] = self.rewards_window.geometry()
                else:
                    self.config['rewards_window']['open'] = False
            except tk.TclError:
                self.config['rewards_window']['open'] = False

            try:
                if self.map_window and self.map_window.winfo_exists():
                    self.config['map_window']['open'] = True
                    self.config['map_window']['geometry'] = self.map_window.geometry()
                else:
                    self.config['map_window']['open'] = False
            except tk.TclError:
                self.config['map_window']['open'] = False

            # Sauvegarder le fichier JSON
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)

        except (tk.TclError, RuntimeError) as save_config_tkinter_error:
            # Erreur Tkinter attendue lors de la fermeture (widgets détruits)
            logger.error(f"Erreur Tkinter lors sauvegarde config (attendu): {save_config_tkinter_error}")
        except (IOError, OSError, PermissionError) as save_config_file_error:
            # Erreurs fichier (disque plein, permissions, etc.)
            logger.error(f"Erreur I/O sauvegarde config: {save_config_file_error}")
        except (TypeError, ValueError) as save_config_data_error:
            # Erreurs de données (JSON invalide, types incompatibles)
            logger.error(f"Erreur données sauvegarde config: {save_config_data_error}")

    def _run_gui(self):
        """Exécute la boucle GUI"""
        self.window = tk.Tk()
        self.window.title(self.title)

        # Appliquer géométrie sauvegardée
        saved_geometry = self.config['main_window'].get('geometry', '1400x900')
        self.window.geometry(saved_geometry)

        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_widgets()

        # Rouvrir les fenêtres ouvertes
        if self.config['player_window'].get('open', False):
            self.window.after(500, self._open_player_window) # type: ignore

        if self.config['rewards_window'].get('open', False):
            self.window.after(600, self._open_rewards_window) # type: ignore

        if self.config['map_window'].get('open', False):
            self.window.after(700, self._open_map_window) # type: ignore

        self._schedule_update()
        self.window.mainloop()
        self.running = False

    def _create_widgets(self):
        """
        Crée tous les widgets
        """
        # Header
        header_frame = tk.Frame(self.window, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP) # type: ignore

        tk.Label(
            header_frame,
            text="🎮 MONSTER HUNTER TRI - ENTRAÎNEMENT IA v3",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#2c3e50"
        ).pack(pady=15)

        # Main container
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) # type: ignore

        # Left panel
        left_panel = tk.Frame(main_frame, width=350, bg="#ecf0f1")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10)) # type: ignore

        # Right panel
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True) # type: ignore

        self._create_stats_panel(left_panel)
        self._create_plots(right_panel)

    def _create_stats_panel(self, parent):
        """Crée le panel de stats"""
        tk.Label(
            parent,
            text="📊 STATISTIQUES",
            font=("Arial", 14, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(fill=tk.X) # type: ignore

        stats_container = tk.Frame(parent, bg="#ecf0f1")
        stats_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) # type: ignore

        # Stats de base - Affichage fenetre principal
        stats_config = [
            ("episode", "Episode", "#3498db"),
            ("step", "Step (épisode)", "#3498db"),
            ("total_steps", "Steps total", "#95a5a6"),
            ("reward", "Reward (step)", "#2ecc71"),
            ("episode_reward", "Reward (épisode)", "#27ae60"),
            ("hp", "HP", "#e74c3c"),
            ("stamina", "Stamina", "#f39c12"),
            ("hits", "Hits", "#9b59b6"),
            ("deaths", "Deaths", "#95a5a6"),
            ("zone", "Zone", "#1abc9c"),
            ("action", "Action", "#34495e"),
        ]

        # Boucle avec stockage des labels
        for key, label_text, color in stats_config:
            frame = tk.Frame(stats_container, bg="white", relief=tk.RAISED, bd=2) # type: ignore
            frame.pack(fill=tk.X, pady=5) # type: ignore

            tk.Label(
                frame,
                text=label_text + ":",
                font=("Arial", 10, "bold"),
                bg="white",
                anchor="w"
            ).pack(side=tk.LEFT, padx=10, pady=6)  # type: ignore

            value_label = tk.Label(
                frame,
                text="0",
                font=("Arial", 10),
                bg="white",
                fg=color,
                anchor="e"
            )
            value_label.pack(side=tk.RIGHT, padx=10, pady=6)  # type: ignore

            # Stocker le label pour mise à jour ultérieure
            self.stat_labels[key] = value_label

        # AJOUTER indicateur d'état menu (après la boucle)
        menu_status_frame = tk.Frame(stats_container, bg="white", relief=tk.RAISED, bd=2)  # type: ignore
        menu_status_frame.pack(fill=tk.X, pady=5)  # type: ignore

        tk.Label(
            menu_status_frame,
            text="Menu ouvert:",
            font=("Arial", 10, "bold"),
            bg="white",
            anchor="w"
        ).pack(side=tk.LEFT, padx=10, pady=6)  # type: ignore

        self.menu_status_indicator = tk.Label(
            menu_status_frame,
            text="NON",
            font=("Arial", 11),
            bg="white",
            fg="#95a5a6",
            anchor="e"
        )
        self.menu_status_indicator.pack(side=tk.RIGHT, padx=10, pady=6)  # type: ignore

        # Temps écoulé
        self.time_label = tk.Label(
            stats_container,
            text="⏱️ Temps: 00:00:00",
            font=("Arial", 11),
            bg="#ecf0f1"
        )
        self.time_label.pack(pady=10)

        # Boutons dépliables
        buttons_frame = tk.Frame(stats_container, bg="#ecf0f1")
        buttons_frame.pack(fill=tk.X, pady=10) # type: ignore

        tk.Button(
            buttons_frame,
            text="📈 Stats Étendues (Player)",
            font=("Arial", 10, "bold"),
            bg="#3498db",
            fg="white",
            command=self._open_player_window,
            cursor="hand2"
        ).pack(fill=tk.X, pady=3) # type: ignore

        tk.Button(
            buttons_frame,
            text="Reward Breakdown",
            font=("Arial", 10, "bold"),
            bg="#2ecc71",
            fg="white",
            command=self._open_rewards_window,
            cursor="hand2"
        ).pack(fill=tk.X, pady=3) # type: ignore

        tk.Button(
            buttons_frame,
            text="Statistiques Zone",
            font=("Arial", 10, "bold"),
            bg="#e67e22",
            fg="white",
            command=self._open_zone_stats_window,
            cursor="hand2"
        ).pack(fill=tk.X, pady=3) # type: ignore

        tk.Button(
            buttons_frame,
            text="⏹️ ARRÊTER L'ENTRAÎNEMENT",
            font=("Arial", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            command=self._on_stop_clicked,
            cursor="hand2",
            height=2
        ).pack(fill=tk.X, pady=(10, 3)) # type: ignore

        tk.Button(
            buttons_frame,
            text="Carte d'Exploration 3D",
            font=("Arial", 10, "bold"),
            bg="#1abc9c",
            fg="white",
            command=self._open_map_window,
            cursor="hand2"
        ).pack(fill=tk.X, pady=3) # type: ignore

    def _open_zone_stats_window(self):
        """
        Ouvre la fenêtre des statistiques de zone
        """
        if self.zone_stats_window and self.zone_stats_window.winfo_exists():
            self.zone_stats_window.lift()
            return

        self.zone_stats_window = tk.Toplevel(self.window)
        self.zone_stats_window.title("📊 Statistiques Zone")

        # Appliquer géométrie sauvegardée
        saved_geometry = self.config.get('zone_stats_window', {}).get('geometry', '600x500')
        self.zone_stats_window.geometry(saved_geometry)

        tk.Label(
            self.zone_stats_window,
            text="📊 STATISTIQUES PAR ZONE",
            font=("Arial", 14, "bold"),
            bg="#e67e22",
            fg="white"
        ).pack(fill=tk.X) # type: ignore

        container = tk.Frame(self.zone_stats_window, bg="#ecf0f1")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) # type: ignore

        # Zone actuelle
        current_frame = tk.Frame(container, bg="#34495e", relief=tk.RAISED, bd=2) # type: ignore
        current_frame.pack(fill=tk.X, pady=10) # type: ignore

        tk.Label(
            current_frame,
            text="ZONE ACTUELLE:",
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="white",
            anchor="w"
        ).pack(side=tk.LEFT, padx=10, pady=10) # type: ignore

        self.zone_current_label = tk.Label(
            current_frame,
            text="Zone 0",
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="#f39c12",
            anchor="e"
        )
        self.zone_current_label.pack(side=tk.RIGHT, padx=10, pady=10) # type: ignore

        # Stats zone actuelle
        tk.Label(
            container,
            text="Statistiques zone actuelle:",
            font=("Arial", 11, "bold"),
            bg="#ecf0f1"
        ).pack(anchor="w", pady=(10, 5))

        self.zone_stats_labels = {}

        basic_zone_stats = [
            ('monsters_present', '👹 Monstres présents', '#c0392b'),
            ('monster_count', '🔢 Nombre de monstres', '#e74c3c'),
            ('in_monster_zone', '🗺️ Zone avec monstres', '#2ecc71'),
            ('in_combat', '⚔️ En combat actif', '#e67e22'),
            ('zone_reward_total', '💰 Reward totale zone', '#f39c12'),
        ]

        for key, label_text, color in basic_zone_stats:
            frame = tk.Frame(container, bg="white", relief=tk.RAISED, bd=1) # type: ignore
            frame.pack(fill=tk.X, pady=2) # type: ignore

            tk.Label(
                frame,
                text=label_text,
                font=("Arial", 9, "bold"),
                bg="white",
                fg=color,
                anchor="w"
            ).pack(side=tk.LEFT, padx=10, pady=5) # type: ignore

            value_label = tk.Label(
                frame,
                text="0",
                font=("Arial", 9),
                bg="white",
                fg=color,
                anchor="e"
            )
            value_label.pack(side=tk.RIGHT, padx=10, pady=5) # type: ignore

            self.zone_stats_labels[key] = value_label

        # MENU DÉROULANT pour HP monstres (détection automatique)
        hp_frame = tk.Frame(container, bg="#ecf0f1")
        hp_frame.pack(fill=tk.X, pady=10) # type: ignore

        tk.Label(
            hp_frame,
            text="💚 HP Monstres:",
            font=("Arial", 10, "bold"),
            bg="#ecf0f1"
        ).pack(anchor="w", padx=10)

        # Bouton expand/collapse
        self.hp_monsters_expanded = False

        self.hp_expand_button = tk.Button(
            hp_frame,
            text="▶ Afficher HP monstres",
            font=("Arial", 9),
            bg="white",
            command=self._toggle_hp_monsters
        )
        self.hp_expand_button.pack(anchor="w", padx=20, pady=5)

        # Frame avec SCROLL pour HP
        hp_scroll_frame = tk.Frame(hp_frame, bg="#ecf0f1")

        hp_canvas = tk.Canvas(hp_scroll_frame, bg="white", height=150)
        hp_scrollbar = tk.Scrollbar(hp_scroll_frame, orient="vertical", command=hp_canvas.yview)
        self.hp_monsters_inner_frame = tk.Frame(hp_canvas, bg="white")

        self.hp_monsters_inner_frame.bind(
            "<Configure>",
            lambda e: hp_canvas.configure(scrollregion=hp_canvas.bbox("all"))
        )

        hp_canvas.create_window((0, 0), window=self.hp_monsters_inner_frame, anchor="nw")
        hp_canvas.configure(yscrollcommand=hp_scrollbar.set)

        hp_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) # type: ignore
        hp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y) # type: ignore

        self.hp_monsters_frame = hp_scroll_frame
        self.hp_monsters_labels = {}

        # EXPLORATION TRACKER STATS (déplacées ici)
        tk.Label(
            container,
            text="📊 Statistiques Exploration:",
            font=("Arial", 11, "bold"),
            bg="#ecf0f1"
        ).pack(anchor="w", pady=(15, 5))

        exploration_items = [
            ('total_cubes', '🗺️ Cubes explorés', '#3498db'),
            ('zones_discovered', '🌍 Zones découvertes', '#1abc9c'),
            ('exploration_visits', '👣 Visites totales', '#9b59b6'),
            ('left_monster_zone_count', '🚪 Sorties zone monstre', '#e74c3c'),
        ]

        for key, label_text, color in exploration_items:
            frame = tk.Frame(container, bg="white", relief=tk.RAISED, bd=1) # type: ignore
            frame.pack(fill=tk.X, pady=2) # type: ignore

            tk.Label(
                frame,
                text=label_text,
                font=("Arial", 9, "bold"),
                bg="white",
                fg=color,
                anchor="w"
            ).pack(side=tk.LEFT, padx=10, pady=5) # type: ignore

            value_label = tk.Label(
                frame,
                text="0",
                font=("Arial", 9),
                bg="white",
                fg=color,
                anchor="e"
            )
            value_label.pack(side=tk.RIGHT, padx=10, pady=5) # type: ignore

            self.zone_stats_labels[key] = value_label

        self._update_zone_stats_window()

    def _update_hp_monsters_display(self):
        """
        Met à jour l'affichage dynamique des HP monstres (DÉTECTION AUTOMATIQUE)
        """
        # Vérifier que la frame existe et est valide
        if not hasattr(self, 'hp_monsters_inner_frame') or not self.hp_monsters_inner_frame.winfo_exists():
            return

        # Détruire PROPREMENT les anciens widgets
        for widget in self.hp_monsters_inner_frame.winfo_children():
            try:
                widget.destroy()
            except tk.TclError:
                pass

        # Vider le dict
        self.hp_monsters_labels.clear()

        # DÉTECTION AUTOMATIQUE de tous les smonster_hp* dans current_stats
        monster_hp_keys = [key for key in self.current_stats.keys() if
                           key.startswith('smonster') and key.endswith('_hp')]

        # Trier par numéro (smonster1_hp, smonster2_hp, etc.)
        monster_hp_keys.sort(key=lambda k: int(k.replace('smonster', '').replace('_hp', '')))

        if not monster_hp_keys:
            # Message si aucun monstre détecté
            tk.Label(
                self.hp_monsters_inner_frame,
                text="Aucun monstre détecté",
                font=("Arial", 9, "italic"),
                bg="white",
                fg="#95a5a6"
            ).pack(anchor="w", padx=10, pady=5)
            return

        # Créer les slots dynamiquement pour chaque monstre détecté
        for key in monster_hp_keys:
            # Extraire le numéro (ex: smonster3_hp -> 3)
            monster_num = int(key.replace('smonster', '').replace('_hp', ''))
            hp = self.current_stats.get(key, 0)

            # Créer frame pour ce monstre
            try:
                frame = tk.Frame(self.hp_monsters_inner_frame, bg="white", relief=tk.RAISED, bd=1) # type: ignore
                frame.pack(fill=tk.X, pady=1, padx=5) # type: ignore

                # Label nom
                name_label = tk.Label(
                    frame,
                    text=f"Monstre {monster_num}:",
                    font=("Arial", 8),
                    bg="white",
                    fg="#27ae60" if hp and hp > 0 else "#95a5a6",
                    anchor="w"
                )
                name_label.pack(side=tk.LEFT, padx=10, pady=3) # type: ignore

                # Label valeur
                value_label = tk.Label(
                    frame,
                    text="—" if not hp or hp <= 0 else f"{hp} HP",
                    font=("Arial", 8),
                    bg="white",
                    fg="#27ae60" if hp and hp > 0 else "#95a5a6",
                    anchor="e"
                )
                value_label.pack(side=tk.RIGHT, padx=10, pady=3) # type: ignore

                # Stocker référence
                self.hp_monsters_labels[key] = value_label

            except tk.TclError as e:
                logger.error(f"Erreur création widget HP monstre {monster_num}: {e}")
                continue

    def _toggle_hp_monsters(self):
        """Affiche/masque les HP de monstres"""
        self.hp_monsters_expanded = not self.hp_monsters_expanded

        if self.hp_monsters_expanded:
            # Afficher
            self.hp_expand_button.config(text="▼ Masquer HP monstres")
            self.hp_monsters_frame.pack(fill=tk.BOTH, expand=True, padx=20) # type: ignore

            # Créer/mettre à jour les labels dynamiquement
            self._update_hp_monsters_display()
        else:
            # Masquer
            self.hp_expand_button.config(text="▶ Afficher HP monstres")
            self.hp_monsters_frame.pack_forget()

    def _update_zone_stats_window(self):
        """
        Met à jour les stats de zone (SÉCURISÉ)
        """
        # Mettre à jour carte 3D seulement si ouverte
        if self.map_window and self.map_window.winfo_exists():
            try:
                self._update_map_window()
            except (AttributeError, RuntimeError) as update_map_error:
                logger.error(f"Erreur mise à jour carte 3D : {update_map_error}")
                # Recréer le canvas si nécessaire
                if hasattr(self, 'map_ax'):
                    self.map_ax = None

        # Vérifier que les labels existent
        if not hasattr(self, 'zone_stats_labels') or not self.zone_stats_labels:
            return

        current_zone = self.current_stats.get('zone', 0)

        # Mettre à jour label zone actuelle (sécurisé)
        if hasattr(self, 'zone_current_label') and self.zone_current_label.winfo_exists():
            try:
                self.zone_current_label.config(text=f"Zone {current_zone}")
            except tk.TclError:
                pass

        # Mettre à jour tous les labels
        for stat_key, stat_label in list(self.zone_stats_labels.items()):  # list() pour éviter RuntimeError
            # Vérifier que le label existe encore
            try:
                if not stat_label.winfo_exists():
                    continue
            except tk.TclError:
                continue

            value = self.current_stats.get(stat_key, 0)

            # Gérer None
            if value is None:
                if stat_key in ['in_monster_zone', 'monsters_present']:
                    try:
                        stat_label.config(text="NON")
                    except tk.TclError:
                        pass
                else:
                    try:
                        stat_label.config(text="0")
                    except tk.TclError:
                        pass
                continue

            # Formater selon le type
            try:
                if stat_key == 'in_monster_zone':
                    # Forcer conversion en bool pour éviter problèmes avec 0/1
                    bool_value = bool(value) if value is not None else False
                    stat_label.config(text="OUI" if bool_value else "NON")
                elif stat_key == 'monsters_present':
                    bool_value = bool(value) if value is not None else False
                    stat_label.config(text="OUI" if bool_value else "NON")
                elif stat_key == 'in_combat':
                    bool_value = bool(value) if value is not None else False
                    stat_label.config(text="OUI" if bool_value else "NON")
                elif 'hp' in stat_key:
                    if value and value > 0:
                        stat_label.config(text=f"{value} HP")
                    else:
                        stat_label.config(text="—")
                elif 'reward' in stat_key:
                    stat_label.config(text=f"{value:.2f}")
                else:
                    stat_label.config(text=str(value))
            except tk.TclError:
                pass  # Widget détruit entre-temps

        # Mettre à jour HP monstres si visible (sécurisé)
        if hasattr(self, 'hp_monsters_expanded') and self.hp_monsters_expanded:
            try:
                # Ne recréer que si nécessaire
                if not self.hp_monsters_labels:
                    self._update_hp_monsters_display()
                else:
                    # Juste mettre à jour les valeurs existantes
                    for monster_key, monster_label in list(self.hp_monsters_labels.items()):
                        try:
                            if not monster_label.winfo_exists():
                                continue
                        except tk.TclError:
                            continue

                        hp = self.current_stats.get(monster_key, 0)
                        try:
                            if not hp or hp <= 0:
                                monster_label.config(text="—", fg="#95a5a6")
                            else:
                                monster_label.config(text=f"{hp} HP", fg="#27ae60")
                        except tk.TclError:
                            pass
            except tk.TclError:
                pass  # Ignorer si widgets détruits

    def _open_map_window(self):
        """
        Ouvre la fenêtre de visualisation 3D de la carte
        """
        # Vérifier que la fenetre existe
        if self.map_window and self.map_window.winfo_exists():
            self.map_window.lift()
            return

        self.map_window = tk.Toplevel(self.window)
        self.map_window.title("🗺️ Carte d'Exploration 3D")
        self.map_window.geometry("800x700")

        tk.Label(
            self.map_window,
            text="🗺️ CARTE D'EXPLORATION (VUE 3D)",
            font=("Arial", 14, "bold"),
            bg="#1abc9c",
            fg="white"
        ).pack(fill=tk.X) # type: ignore
        # Creer figure matplotlib 3D avec verification
        try:
            fig = Figure(figsize=(8, 6), dpi=100)
            self.map_ax = fig.add_subplot(111, projection='3d')
            self.map_ax.set_title("Cubes explorés", fontsize=12, fontweight='bold')
            self.map_ax.set_xlabel("X (Est-Ouest)")
            self.map_ax.set_ylabel("Y (Haut-Bas)")
            self.map_ax.set_zlabel("Z (Nord-Sud)")

            # CREER CANVAS ET VERIFIER LA LIAISON
            self.map_canvas = FigureCanvasTkAgg(fig, self.map_window)

            # Verifier que la figure est bien liee
            if self.map_canvas.figure is None:
                logger.warning("Figure non liée au canvas - tentative de correction")
                self.map_canvas.figure = fig

            # Verifier que le DPI est defini
            if not hasattr(fig, 'dpi') or fig.dpi is None:
                logger.warning("DPI non défini - correction")
                fig.dpi = 100

            self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        except Exception as map_creation_error:
            logger.error(f"ERREUR création carte 3D: {map_creation_error}")
            import traceback
            traceback.print_exc()

            # Fermer la fenêtre car erreur critique
            if self.map_window:
                self.map_window.destroy()
                self.map_window = None
            return

        # Configurer les tooltips immédiatement
        self._setup_map_tooltips()
        self._tooltips_configured = True

        # Rafraîchissement automatique toutes les 15s
        self.map_auto_refresh = True
        self._schedule_map_refresh()

        # Contrôles
        controls_frame = tk.Frame(self.map_window, bg="#ecf0f1")
        controls_frame.pack(fill=tk.X, padx=10, pady=5) # type: ignore

        tk.Button(
            controls_frame,
            text="🗺️ Carte Zone",
            command=self._open_full_map_snapshot,
            bg="#3498db",
            fg="white"
        ).pack(side=tk.LEFT, padx=5) # type: ignore

        self.map_zone_filter = tk.IntVar(value=0)
        tk.Label(controls_frame, text="Zone:", bg="#ecf0f1").pack(side=tk.LEFT, padx=5) # type: ignore
        tk.Entry(
            controls_frame,
            textvariable=self.map_zone_filter,
            width=5
        ).pack(side=tk.LEFT) # type: ignore

        # Stats
        self.map_stats_label = tk.Label(
            self.map_window,
            text="Cubes: 0 | Zones: 0 | Visites: 0",
            font=("Arial", 10),
            bg="#ecf0f1"
        )
        self.map_stats_label.pack(fill=tk.X, padx=10, pady=5) # type: ignore

        self._update_map_window()

        # Légende des marqueurs en bas
        markers_legend_frame = tk.Frame(self.map_window, bg="#34495e", relief=tk.RAISED, bd=2) #type: ignore
        markers_legend_frame.pack(fill=tk.X, padx=10, pady=5) #type: ignore

        tk.Label(
            markers_legend_frame,
            text="🏷️ LÉGENDE MARQUEURS:",
            font=("Arial", 10, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=5) #type: ignore

        # Créer les labels pour chaque type de marqueur
        marker_labels = [
            ("💧 Eau", "#3498db"),
            ("👹 Monstre", "#e74c3c"),
            ("🚪 Transition", "#f39c12"),
            ("❌ Obstacle", "#95a5a6"),
        ]

        for text, color in marker_labels:
            tk.Label(
                markers_legend_frame,
                text=text,
                font=("Arial", 9),
                bg="#34495e",
                fg=color,
                padx=10
            ).pack(side=tk.LEFT) #type: ignore

        # Maintenant appeler _update_map_window()
        self._update_map_window()

    def _open_full_map_snapshot(self):
        """
        Ouvre une fenêtre séparée avec snapshot complet de la zone
        """
        current_zone = self.current_stats.get('zone', 0)

        # VÉRIFIER que map_zone_filter existe
        if hasattr(self, 'map_zone_filter') and self.map_zone_filter:
            filter_zone = self.map_zone_filter.get()
            if filter_zone != 0:
                current_zone = filter_zone

        # Créer nouvelle fenêtre
        snapshot_window = tk.Toplevel(self.window)
        snapshot_window.title(f"Carte Complète - Zone {current_zone}")
        snapshot_window.geometry("1000x800")

        # Créer figure matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Zone {current_zone} - Carte Complète", fontsize=14, fontweight='bold')

        # Récupérer TOUS les cubes de la zone (pas de filtre)
        cubes_data = self.current_stats.get('exploration_cubes', {})

        if current_zone not in cubes_data:
            tk.Label(
                snapshot_window,
                text=f"Aucune donnée pour la zone {current_zone}",
                font=("Arial", 12)
            ).pack(pady=20)
            return

        zone_cubes = cubes_data[current_zone]

        # Dessiner tous les cubes (COORDONNÉES ABSOLUES)
        for cube in zone_cubes:
            x, y, z = cube['center_x'], cube['center_y'], cube['center_z']
            size_x = cube.get('size_x', 200)
            size_y = cube.get('size_y', 200)
            size_z = cube.get('size_z', 200)
            visits = cube['visit_count']

            # Couleur selon visites
            if visits == 0:
                color, alpha = 'grey', 0.1
            elif visits == 1:
                color, alpha = '#0BA5FE', 0.25
            elif visits <= 4:
                color, alpha = '#6CDD40', 0.3
            elif visits <= 6:
                color, alpha = '#D2E637', 0.3
            elif visits <= 9:
                color, alpha = 'yellow', 0.35
            else:
                color, alpha = 'red', 0.4

            # UTILISER ax au lieu de self.map_ax
            self._draw_cube_3d_to_ax(ax, x, y, z, size_x, size_y, size_z, color, alpha, cube_data=cube)

        # Ajuster limites automatiquement
        if zone_cubes:
            xs = [c['center_x'] for c in zone_cubes]
            ys = [c['center_y'] for c in zone_cubes]
            zs = [c['center_z'] for c in zone_cubes]

            # Calculer les étendues
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z_min, z_max = min(zs), max(zs)

            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min

            # Trouver la plus grande étendue
            max_range = max(x_range, y_range, z_range)

            # Si trop petit, forcer un minimum
            if max_range < 1000:
                max_range = 1000

            # Ajouter marge de 20%
            margin = max_range * 0.2

            # Centres
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2

            # Définir limites uniformes (pour éviter écrasement)
            half_range = (max_range + margin) / 2

            ax.set_xlim(x_center - half_range, x_center + half_range)
            ax.set_ylim(z_center - half_range, z_center + half_range)
            ax.set_zlim(y_center - half_range, y_center + half_range)

            # Forcer aspect ratio égal
            ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel("X (Est-Ouest)")
        ax.set_ylabel("Z (Nord-Sud)")
        ax.set_zlabel("Y (Hauteur)")

        # Canvas
        canvas = FigureCanvasTkAgg(fig, snapshot_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        # Statistiques
        total = len(zone_cubes)
        total_visits = sum(c['visit_count'] for c in zone_cubes)

        stats_label = tk.Label(
            snapshot_window,
            text=f"Zone {current_zone} : {total} cubes | {total_visits} visites totales",
            font=("Arial", 11),
            bg="#ecf0f1"
        )
        stats_label.pack(fill=tk.X, pady=5) # type: ignore

    @staticmethod
    def _draw_cube_3d_to_ax(ax, cx: float, cy: float, cz: float,
                            size_x: float, size_y: float, size_z: float,
                            color: str, alpha: float, cube_data: dict = None):
        """
        Version de _draw_cube_3d qui accepte un ax spécifique
        (pour snapshot dans nouvelle fenêtre)
        """
        half_x = size_x / 2
        half_y = size_y / 2
        half_z = size_z / 2

        # 8 sommets
        vertices = [
            [cx - half_x, cy - half_y, cz - half_z],
            [cx + half_x, cy - half_y, cz - half_z],
            [cx + half_x, cy + half_y, cz - half_z],
            [cx - half_x, cy + half_y, cz - half_z],
            [cx - half_x, cy - half_y, cz + half_z],
            [cx + half_x, cy - half_y, cz + half_z],
            [cx + half_x, cy + half_y, cz + half_z],
            [cx - half_x, cy + half_y, cz + half_z],
        ]

        # 6 faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
        ]

        cube_collection = Poly3DCollection(
            faces,
            facecolors=color,
            linewidths=1,
            edgecolors='black',
            alpha=alpha,
            zorder=10
        )

        ax.add_collection3d(cube_collection)

        # Dessiner les obstacles si cube_data fourni
        if cube_data and 'blocked_directions' in cube_data:
            blocked = cube_data['blocked_directions']

            for direction, count in blocked.items():
                if count >= 5:
                    if direction == 'north':
                        mx, my, mz = cx, cz + half_z * 0.9, cy
                    elif direction == 'south':
                        mx, my, mz = cx, cz - half_z * 0.9, cy
                    elif direction == 'east':
                        mx, my, mz = cx + half_x * 0.9, cz, cy
                    elif direction == 'west':
                        mx, my, mz = cx - half_x * 0.9, cz, cy
                    elif direction == 'up':
                        mx, my, mz = cx, cz, cy + half_y * 0.9
                    elif direction == 'down':
                        mx, my, mz = cx, cz, cy - half_y * 0.9
                    else:
                        continue

                    ax.scatter(
                        mx, my, mz,
                        c='red',
                        marker='x',
                        s=150,
                        linewidths=3,
                        alpha=0.8,
                        zorder=101,
                        depthshade=False
                    )

    def _schedule_map_refresh(self):
        """
        Planifie le prochain rafraîchissement de la carte
        """
        if self.map_window and self.map_window.winfo_exists() and self.map_auto_refresh:
            try:  # Protection contre erreurs Tkinter
                # Vérifier que map_zone_filter existe
                if hasattr(self, 'map_zone_filter'):
                    self._update_map_window()
                self.map_window.after(1000, self._schedule_map_refresh) # refresh après 1 seconde
            except tk.TclError:
                # Widget détruit entre-temps
                pass

    def _update_map_window(self):
        """
        Met à jour la carte 3D (avec recréation automatique si nécessaire)
        """
        # Vérifier que la fenetre existe
        if not self.map_window or not self.map_window.winfo_exists():
            return

        # AJOUT : Réinitialiser les flags de labels à CHAQUE refresh
        self._water_label_added = False
        self._monster_label_added = False
        self._transition_label_added = False
        self._obstacle_label_added = False

        # VÉRIFICATION COMPLÈTE ET RECRÉATION SI NÉCESSAIRE
        needs_recreation = False

        # Vérifier les attributs essentiels
        if not hasattr(self, 'map_ax') or self.map_ax is None:
            needs_recreation = True
            logger.warning("map_ax manquant - recréation carte")
        elif not hasattr(self, 'map_canvas') or self.map_canvas is None:
            needs_recreation = True
            logger.warning("map_canvas manquant - recréation carte")
        elif not hasattr(self.map_canvas, 'figure') or self.map_canvas.figure is None:
            needs_recreation = True
            logger.warning("figure manquante - recréation carte")
        else:
            # Vérifier la figure du canvas
            try:
                fig = self.map_canvas.figure
                if fig is None:
                    needs_recreation = True
                    logger.warning("figure None - recréation carte")
                elif not hasattr(fig, 'dpi') or fig.dpi is None or fig.dpi == 0:
                    needs_recreation = True
                    logger.warning(f"DPI invalide ({getattr(fig, 'dpi', 'absent')}) - recréation carte")
            except (AttributeError, RuntimeError, TypeError) as canvas_verification_error:
                needs_recreation = True
                logger.error(f"Erreur vérification canvas: {canvas_verification_error} - recréation carte")

        # RECRÉER LE CANVAS SI NÉCESSAIRE
        if needs_recreation:
            try:
                # Détruire l'ancien canvas s'il existe
                if hasattr(self, 'map_canvas') and self.map_canvas is not None:
                    try:
                        self.map_canvas.get_tk_widget().destroy()
                    except(tk.TclError, AttributeError):
                        pass

                # Réinitialiser complètement avant recréation
                self.map_canvas = None
                self.map_ax = None

                # Créer nouvelle figure
                from matplotlib.figure import Figure
                fig = Figure(figsize=(8, 6), dpi=100)

                # Vérifier que la figure est valide AVANT de créer les axes
                if not hasattr(fig, 'dpi') or fig.dpi is None or fig.dpi <= 0:
                    fig.dpi = 100

                # Continuer de créer la figure
                self.map_ax = fig.add_subplot(111, projection='3d')
                self.map_ax.set_title("Cubes explorés", fontsize=12, fontweight='bold')
                self.map_ax.set_xlabel("X (Est-Ouest)")
                self.map_ax.set_ylabel("Y (Haut-Bas)")
                self.map_ax.set_zlabel("Z (Nord-Sud)")

                # Créer nouveau canvas
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                self.map_canvas = FigureCanvasTkAgg(fig, self.map_window)

                # Double vérification de la liaison
                if self.map_canvas.figure is None:
                    self.map_canvas.figure = fig

                # Vérifier que le canvas est bien lié à la fenêtre
                if not self.map_window or not self.map_window.winfo_exists():
                    logger.error("map_window invalide - impossible de recréer carte")
                    self.map_canvas = None
                    self.map_ax = None
                    return

                # Réinsérer dans la fenêtre
                try:
                    self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    logger.debug("Carte 3D recréée avec succès")
                except tk.TclError as pack_error:
                    logger.error(f"Erreur pack widget: {pack_error}")
                    self.map_canvas = None
                    self.map_ax = None
                    return

            except Exception as recreation_error:
                logger.error(f"Impossible de recréer la carte: {recreation_error}")
                self.map_canvas = None
                self.map_ax = None
                return

        # VÉRIFICATION FINALE AVANT DE CONTINUER
        if not hasattr(self, 'map_canvas') or self.map_canvas is None:
            logger.warning("Canvas toujours None après tentative de recréation")
            return

        if not hasattr(self.map_canvas, 'figure') or self.map_canvas.figure is None:
            logger.warning("Figure toujours None après tentative de recréation")
            return

        # Verifie que map_ax existe
        if not hasattr(self, 'map_ax') or self.map_ax is None:
            logger.warning("map_ax toujours None après tentative de recréation")
            return

        # Vérifier les axes 3D spécifiquement
        try:
            # Tester que map_ax est bien un axe 3D valide
            _ = self.map_ax.get_proj()  # Cette méthode n'existe que sur les axes 3D
        except (AttributeError, RuntimeError):
            logger.warning("map_ax invalide (pas un axe 3D) - abandon mise à jour")
            self.map_ax = None
            return

        # Verifie que map_zone_filter existe
        if not hasattr(self, 'map_zone_filter') or self.map_zone_filter is None:
            return

        # Verifie que map_stats_label existe
        if not hasattr(self, 'map_stats_label') or self.map_stats_label is None:
            return

        # Reset data cubes pour tooltips
        self.map_cube_data = []

        # Récupérer les données d'exploration depuis current_stats
        total_cubes = self.current_stats.get('total_cubes', 0)
        zones_discovered = self.current_stats.get('zones_discovered', 0)

        # Récupérer les cubes par zone
        cubes_data = self.current_stats.get('exploration_cubes', {})

        # Clear
        self.map_ax.clear()

        zone_filter = self.map_zone_filter.get()

        current_zone = self.current_stats.get('zone', 0)

        if zone_filter == 0:
            zone_filter = current_zone

        # Plot cubes
        cubes_in_view = 0

        # Reset labels flags
        self._water_label_added = False
        self._monster_label_added = False
        self._transition_label_added = False

        # Obtient les coordonnées du joueur
        player_x = self.current_stats.get('player_x', 0)
        player_y = self.current_stats.get('player_y', 0)
        player_z = self.current_stats.get('player_z', 0)

        # COMPTER TOUS LES CUBES DE LA ZONE (pour stats)
        total_cubes_in_zone = 0
        total_visits_in_zone = 0

        # FILTRER : Garder seulement les cubes proches du joueur
        if player_x and player_y and player_z:
            max_distance = 3250.0 # Rayon de d'observation des cubes (5 * 650, soit 5 cubes)
            filtered_cubes = []

            for zone_id, cubes in cubes_data.items():
                if zone_id != zone_filter:
                    continue

                # COMPTER TOUS les cubes de cette zone
                total_cubes_in_zone += len(cubes)
                total_visits_in_zone += sum(cube['visit_count'] for cube in cubes)

                for cube in cubes:
                    cx, cy, cz = cube['center_x'], cube['center_y'], cube['center_z']

                    # Distance 3D
                    distance = np.sqrt(
                        (cx - player_x) ** 2 +
                        (cy - player_y) ** 2 +
                        (cz - player_z) ** 2
                    )

                    if distance <= max_distance:
                        filtered_cubes.append((zone_id, cube, distance))

            # Trier par distance
            filtered_cubes.sort(key=lambda d_x: d_x[2])

            # Limiter à 40 cubes max
            filtered_cubes = filtered_cubes[:40]

            cubes_in_view = 0

            for zone_id, cube, distance in filtered_cubes:
                # COORDONNÉES RELATIVES AU JOUEUR
                x = cube['center_x'] - player_x
                y = cube['center_y'] - player_y
                z = cube['center_z'] - player_z

                size_x = cube.get('size_x', cube.get('size', 200.0))
                size_y = cube.get('size_y', cube.get('size', 200.0))
                size_z = cube.get('size_z', cube.get('size', 200.0))
                visits = cube['visit_count']

                # Couleur selon visites
                if visits == 0:
                    color = 'grey'
                    alpha = 0.1
                elif visits == 1:
                    color = '#0BA5FE'
                    alpha = 0.25
                elif visits <= 4:
                    color = '#6CDD40'
                    alpha = 0.3
                elif visits <= 6:
                    color = '#D2E637'
                    alpha = 0.3
                elif visits <= 9:
                    color = 'yellow'
                    alpha = 0.35
                else:
                    color = 'red'
                    alpha = 0.4

                # Dessiner cube
                self._draw_cube_3d(x, y, z, size_x, size_y, size_z, color, alpha, cube_data=cube)

                # Stocker pour tooltip
                self.map_cube_data.append({
                    'center': (x, y, z),
                    'size': (size_x, size_y, size_z),
                    'cube': cube
                })
                cubes_in_view += 1

                # Dessiner les marqueurs AU CENTRE du cube
                if 'markers' in cube and cube['markers']:
                    markers = cube['markers']
                    # Les coordonnées x, y, z sont déjà relatives au joueur
                    self._draw_cube_markers_from_system(markers, x, z, y)

            # ZOOM AUTO centré sur le JOUEUR
            if player_x and player_y and player_z:
                view_range = 1625.0  # Nouveau range (2.5 * 650)
                # Cela affiche environ 5 cubes de 650 dans chaque direction

                x_center = 0.0
                y_center = 0.0
                z_center = 0.0

                # Protection contre map_ax None
                try:
                    self.map_ax.set_xlim(x_center - view_range, x_center + view_range)
                    self.map_ax.set_ylim(z_center - view_range, z_center + view_range)
                    self.map_ax.set_zlim(y_center - view_range / 2, y_center + view_range / 2)
                except (AttributeError, RuntimeError) as set_lim_error:
                    logger.warning(f"Erreur définition limites axes: {set_lim_error}")
                    self.map_ax = None
                    return
            else:
                # Fallback si pas de position joueur
                if cubes_in_view > 0 and filtered_cubes:
                    cube_positions = [(cube['center_x'], cube['center_y'], cube['center_z'])
                                      for _, cube, _ in filtered_cubes]
                    xs = [pos[0] for pos in cube_positions]
                    ys = [pos[1] for pos in cube_positions]
                    zs = [pos[2] for pos in cube_positions]

                    margin = 500.0
                    self.map_ax.set_xlim(min(xs) - margin, max(xs) + margin)
                    self.map_ax.set_ylim(min(zs) - margin, max(zs) + margin)
                    self.map_ax.set_zlim(min(ys) - margin, max(ys) + margin)

        # Dessine le joueur
        if player_x and player_y and player_z:
            player_rel_x = 0.0
            player_rel_y = 0.0
            player_rel_z = 0.0

            orientation = self.current_stats.get('orientation', 0.0) or 0.0
            orientation_rad = np.radians(orientation)

            arrow_length = 300.0
            arrow_dx = arrow_length * np.sin(orientation_rad)
            arrow_dz = arrow_length * np.cos(orientation_rad)
            arrow_dy = 0

            self.map_ax.quiver(
                player_rel_x, player_rel_z, player_rel_y,
                arrow_dx, arrow_dz, arrow_dy,
                color='cyan',
                arrow_length_ratio=0.3,
                linewidth=4,
                alpha=0.9,
                label='Joueur',
                zorder=1000
            )

            self.map_ax.scatter(
                player_rel_x, player_rel_z, player_rel_y,
                c='cyan',
                marker='o',
                s=100,
                edgecolors='black',
                linewidths=2,
                zorder=1001,
                depthshade=False
            )

        # Activer tooltips
        if not hasattr(self, '_tooltips_configured') or not self._tooltips_configured:
            self._setup_map_tooltips()
            self._tooltips_configured = True

        # Axes de la carte
        self.map_ax.set_xlabel("X (Est-Ouest)")
        self.map_ax.set_ylabel("Z (Nord-Sud)")
        self.map_ax.set_zlabel("Y (Hauteur)")

        # Legende overlay de la carte 3D
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor='grey',
                  alpha=0.3,
                  label='0 visite (non exploré)'
                  ),

            Patch(facecolor='#0BA5FE',
                  alpha=0.3,
                  label='1 visite ou nouveau cube'
                  ),

            Patch(facecolor='#6CDD40',
                  alpha=0.3,
                  label='2-6 visites'
                  ),

            Patch(facecolor='yellow',
                  alpha=0.3,
                  label='6-10 visites'
                  ),

            Patch(facecolor='red',
                  alpha=0.3,
                  label='+10 visites'
                  ),

            Line2D([0],
                   [0],
                   marker='o',
                   color='w',
                   markerfacecolor='cyan',
                   markersize=10,
                   label='Joueur (centre)',
                   markeredgecolor='black',
                   markeredgewidth=2
                   ),

            Line2D([0],
                   [0],
                   marker='x',
                   color='red',
                   linewidth=0,
                   markersize=8,
                   label='Obstacle',
                   markeredgewidth=2
                   ),
        ]

        # Dessine la legende en overlay
        legend = self.map_ax.legend(handles=legend_elements,
                                    loc='upper right',  # Position de la fenetre
                                    bbox_to_anchor=(0.98, 0.98),  # Décalage depuis le coin
                                    framealpha=0.95,  # Opacité
                                    fontsize=8,
                                    shadow=True,  # Ombre pour mieux voir
                                    fancybox=True,  # Coins arrondis
                                    )
        legend.set_zorder(2000) # Dessinner l'overlay au-dessus du reste

        # Stats
        self.map_stats_label.config(
            text=f"Zone {zone_filter}: {total_cubes_in_zone} cubes ({cubes_in_view} affichés) | {total_visits_in_zone} visites | Global: {total_cubes} cubes, {zones_discovered} zones"
        )

        # VÉRIFICATION CRITIQUE AVANT DRAW (sans doublon)
        try:
            # Vérifications de base
            if not hasattr(self, 'map_canvas') or self.map_canvas is None:
                logger.warning("Canvas None avant draw - abandon")
                return

            if not hasattr(self.map_canvas, 'figure') or self.map_canvas.figure is None:
                logger.warning("Figure None avant draw - abandon")
                return

            # Vérifier et corriger DPI
            fig = self.map_canvas.figure
            if not hasattr(fig, 'dpi') or fig.dpi is None or fig.dpi <= 0:
                logger.warning(f"DPI invalide ({getattr(fig, 'dpi', 'absent')}) - correction à 100")
                try:
                    fig.dpi = 100
                except (AttributeError, RuntimeError):
                    logger.error("Impossible de corriger DPI - abandon draw")
                    return

            # Vérifier que le widget Tk existe avant draw
            try:
                widget = self.map_canvas.get_tk_widget()
                if not widget or not widget.winfo_exists():
                    logger.warning("Widget Tk détruit - abandon draw")
                    self.map_canvas = None
                    self.map_ax = None
                    return
            except (tk.TclError, AttributeError):
                logger.warning("Widget Tk inaccessible - abandon draw")
                self.map_canvas = None
                self.map_ax = None
                return

            # UN SEUL DRAW
            self.map_canvas.draw()

        except AttributeError as update_map_window_attr_error:
            # Attribut manquant (figure, dpi, etc.)
            logger.warning(f"Attribut manquant lors du draw: {update_map_window_attr_error}")
            return # simplement retourner et attendre prochain cycle

        except RuntimeError as update_map_window_runtime_error:
            # Erreur runtime matplotlib/tkinter (thread, canvas détruit, etc.)
            logger.error(f"Erreur runtime draw: {update_map_window_runtime_error}")
            # Mettre à None si erreur critique
            if "destroyed" in str(update_map_window_runtime_error).lower():
                self.map_ax = None
                self.map_canvas = None
            return

        except TypeError as update_map_window_type_error:
            # Erreur de type (DPI invalide, etc.)
            logger.error(f"Erreur de type draw: {update_map_window_type_error}")
            return

    def _draw_cube_markers_from_system(self, markers: dict, cx: float, cy: float, cz: float):
        """
        Dessine les marqueurs depuis le marker_system comme des points au centre du cube

        IMPORTANT : cx, cy, cz sont déjà en COORDONNÉES RELATIVES au joueur
                    (calculées dans _update_map_window avec - player_x/y/z)

        AXES MATPLOTLIB 3D:
            - X axis = X du jeu (Est-Ouest)
            - Y axis = Z du jeu (Nord-Sud)
            - Z axis = Y du jeu (Hauteur)
        """
        from environment.cube_markers import MarkerType

        # Marqueur EAU (bleu, point au centre)
        if MarkerType.WATER in markers:
            self.map_ax.scatter(
                cx,  # X du jeu → X matplotlib
                cz,  # Z du jeu → Y matplotlib
                cy,  # Y du jeu → Z matplotlib
                c='blue',
                marker='o',
                s=150,
                alpha=0.9,
                edgecolors='white',
                linewidths=2,
                label='Eau' if not self._water_label_added else None,
                zorder=150,  # Augmenté pour être au-dessus des cubes (10-101)
                depthshade=False
            )
            self._water_label_added = True

        # Marqueur MONSTRE (rouge, X au centre)
        if MarkerType.MONSTER_LOCATION in markers:
            marker = markers[MarkerType.MONSTER_LOCATION]
            marker_size = 150 + (marker.strength * 200)
            alpha = max(0.3, marker.strength * 0.9)

            self.map_ax.scatter(
                cx,  # X du jeu → X matplotlib
                cz,  # Z du jeu → Y matplotlib
                cy,  # Y du jeu → Z matplotlib
                c='red',
                marker='X',
                s=marker_size,
                alpha=alpha,
                edgecolors='black',
                linewidths=2,
                label='Monstre' if not self._monster_label_added else None,
                zorder=150,  # Au-dessus des cubes
                depthshade=False
            )
            self._monster_label_added = True

        # Marqueur TRANSITION (jaune, triangle au centre)
        if MarkerType.ZONE_TRANSITION in markers:
            self.map_ax.scatter(
                cx,  # X du jeu → X matplotlib
                cz,  # Z du jeu → Y matplotlib
                cy,  # Y du jeu → Z matplotlib
                c='yellow',
                marker='^',
                s=200,
                alpha=0.9,
                edgecolors='orange',
                linewidths=2,
                label='Transition' if not self._transition_label_added else None,
                zorder=150,  # Au-dessus des cubes
                depthshade=False
            )
            self._transition_label_added = True

    def _draw_cube_3d(self, cx: float, cy: float, cz: float,
                  size_x: float, size_y: float, size_z: float,
                  color: str, alpha: float, cube_data: dict = None):
        """
        Dessine un parallélépipède 3D (cube ou étendu)

        Args:
            cx, cy, cz: Centre
            size_x, size_y, size_z: Dimensions sur chaque axe
            color: Couleur
            alpha: Transparence
            cube_data: Données du cube (pour obstacles)
        """
        half_x = size_x / 2
        half_y = size_y / 2
        half_z = size_z / 2

        # 8 sommets du parallélépipède
        vertices = [
            [cx - half_x, cy - half_y, cz - half_z],  # 0
            [cx + half_x, cy - half_y, cz - half_z],  # 1
            [cx + half_x, cy + half_y, cz - half_z],  # 2
            [cx - half_x, cy + half_y, cz - half_z],  # 3
            [cx - half_x, cy - half_y, cz + half_z],  # 4
            [cx + half_x, cy - half_y, cz + half_z],  # 5
            [cx + half_x, cy + half_y, cz + half_z],  # 6
            [cx - half_x, cy + half_y, cz + half_z],  # 7
        ]

        # 6 faces du parallélépipède
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bas
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Haut
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Face avant
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Face arrière
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Côté gauche
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Côté droit
        ]

        # Créer collection de polygones
        cube_collection = Poly3DCollection(
            faces,
            facecolors=color,
            linewidths=1,
            edgecolors='black',
            alpha=alpha,
            zorder=10,
        )

        self.map_ax.add_collection3d(cube_collection)

        # Marqueurs d'obstacles si cube fourni
        if cube_data and 'blocked_directions' in cube_data:
            blocked = cube_data['blocked_directions']

            # Dessiner X rouge pour obstacles DÉTECTÉS (count >= 5)
            for direction, count in blocked.items():
                if count >= 5:  # Seuil de confirmation d'obstacle
                    # Position du marqueur selon direction
                    if direction == 'north':  # +Z
                        mx, my, mz = cx, cz + half_z * 0.9, cy
                    elif direction == 'south':  # -Z
                        mx, my, mz = cx, cz - half_z * 0.9, cy
                    elif direction == 'east':  # +X
                        mx, my, mz = cx + half_x * 0.9, cz, cy
                    elif direction == 'west':  # -X
                        mx, my, mz = cx - half_x * 0.9, cz, cy
                    elif direction == 'up':  # +Y
                        mx, my, mz = cx, cz, cy + half_y * 0.9
                    elif direction == 'down':  # -Y
                        mx, my, mz = cx, cz, cy - half_y * 0.9
                    else:
                        continue

                    # Dessiner X rouge avec label unique
                    self.map_ax.scatter(
                        mx, my, mz,
                        c='red',
                        marker='x',
                        s=150,
                        linewidths=3,
                        alpha=0.8,
                        label='Obstacle' if not hasattr(self, '_obstacle_label_added') else None,
                        zorder=101,
                        depthshade=False
                    )
                    self._obstacle_label_added = True

    def _setup_map_tooltips(self):
        """
        Configure les tooltips interactifs sur la carte
        """
        # Recréer l'annotation à chaque fois (car clear() la détruit)
        self.map_annotation = self.map_ax.text2D(
            0.02, 0.98, "",
            transform=self.map_ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5",
                      fc="#ffffcc",
                      alpha=0.95,
                      edgecolor='black',
                      linewidth=2,
                      ),
            verticalalignment='top',
            fontsize=9,
            visible=False,
            family='monospace',
            zorder=1000 # Definie ordre d'affichage
        )

        # Variable pour mémoriser le cube actuellement survolé
        # Initialiser UNIQUEMENT si pas encore défini
        if not hasattr(self, '_current_hovered_cube'):
            self._current_hovered_cube = None
        # Sinon garder la valeur existante pour éviter le clignotement

        def on_hover(event: Any) -> None:
            """
            Callback survol souris - VERSION CORRIGÉE FINALE
            """
            # Vérifications de base
            if not hasattr(self, 'map_ax') or event.inaxes != self.map_ax:
                return

            if not hasattr(self, 'map_cube_data') or not self.map_cube_data:
                return

            if event.xdata is None or event.ydata is None:
                return

            # Projection 3D -> 2D : matplotlib utilise une transformation complexe
            # On doit convertir les coordonnées 3D des cubes en coordonnées 2D d'affichage

            closest_cube = None
            min_dist_screen = float('inf')

            try:
                # Récupérer la transformation 3D->2D
                proj = self.map_ax.get_proj()

                for data in self.map_cube_data:
                    cx, cy, cz = data['center']

                    # Convertir position 3D en position 2D écran
                    # Dans matplotlib 3D : x=X, y=Z, z=Y
                    point_3d = np.array([cx, cz, cy, 1.0])
                    point_2d_homogeneous = proj.dot(point_3d)

                    # Normaliser coordonnées homogènes
                    if point_2d_homogeneous[3] != 0:
                        point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[3]
                    else:
                        continue

                    # Transformer en coordonnées data
                    try:
                        # Inverse transform pour obtenir data coords
                        inv = self.map_ax.transData.inverted()
                        display_coords = self.map_ax.transData.transform([[point_2d[0], point_2d[1]]])
                        data_coords = inv.transform(display_coords)[0]

                        screen_x, screen_y = data_coords[0], data_coords[1]
                    except (AttributeError, ValueError, IndexError) as transform_error:
                        # Fallback: utiliser directement X et Z
                        logger.error(f"Transform échouée pour cube {data.get('cube', {}).get('zone_id', '?')}: {transform_error}")
                        screen_x, screen_y = cx, cz

                    # Distance 2D entre souris et cube projeté
                    dx = event.xdata - screen_x
                    dy = event.ydata - screen_y
                    dist_screen = np.sqrt(dx ** 2 + dy ** 2)

                    # Rayon de détection adaptatif
                    cube_dict = data['cube']
                    size_x = cube_dict.get('size_x', 200)
                    size_z = cube_dict.get('size_z', 200)
                    avg_size = (size_x + size_z) / 2.0
                    detection_radius = avg_size * 1.2  # Augmenté à 120%

                    # Garder le plus proche
                    if dist_screen < detection_radius and dist_screen < min_dist_screen:
                        min_dist_screen = dist_screen
                        closest_cube = data

            except Exception as proj_error:
                # Fallback simple si projection échoue
                logger.error(f"Projection 3D->2D échouée: {proj_error}, utilisation méthode simple")

                for data in self.map_cube_data:
                    cx, cy, cz = data['center']

                    # Distance simple X-Z
                    dx = event.xdata - cx
                    dz = event.ydata - cz
                    dist = np.sqrt(dx ** 2 + dz ** 2)

                    cube_dict = data['cube']
                    size_x = cube_dict.get('size_x', 200)
                    size_z = cube_dict.get('size_z', 200)
                    avg_size = (size_x + size_z) / 2.0
                    detection_radius = avg_size * 1.2

                    if dist < detection_radius and dist < min_dist_screen:
                        min_dist_screen = dist
                        closest_cube = data

            # AFFICHAGE (avec correction anti clignotement)
            if closest_cube:
                # Vérifier si c'est un nouveau cube ou si le texte a changé
                need_update = False

                if closest_cube != self._current_hovered_cube:
                    # Nouveau cube détecté
                    need_update = True
                    self._current_hovered_cube = closest_cube

                if need_update:
                    cube = closest_cube['cube']

                    # Position absolue
                    abs_x = cube['center_x']
                    abs_y = cube['center_y']
                    abs_z = cube['center_z']
                    cube_id = f"({abs_x:.0f}, {abs_y:.0f}, {abs_z:.0f})"

                    # Visites
                    visit_count = cube.get('visit_count', 0)
                    effective_visits = cube.get('effective_visit_count', visit_count)
                    total_visits = cube.get('total_visits', 0)

                    # Marqueurs
                    markers = cube.get('markers', {})
                    marker_names = []
                    if markers:
                        from environment.cube_markers import MarkerType
                        marker_map = {
                            MarkerType.WATER: "[EAU]",
                            MarkerType.MONSTER_LOCATION: "[MONSTRE]",
                            MarkerType.ZONE_TRANSITION: "[TRANSITION]",
                            MarkerType.DANGER: "[DANGER]",
                        }
                        for marker_type in markers.keys():
                            name = marker_map.get(marker_type, str(marker_type))
                            marker_names.append(name)

                    # Texte
                    lines = [
                        "╔═══════════════════════════",
                        f"║ CUBE - Zone {cube['zone_id']}",
                        f"║ ID: {cube_id}",
                        "╠═══════════════════════════",
                        "║ Position (jeu):",
                        f"║   X: {abs_x:>8.1f}",
                        f"║   Y: {abs_y:>8.1f}",
                        f"║   Z: {abs_z:>8.1f}",
                        "╠═══════════════════════════",
                        "║ Visites:",
                        f"║   Episode (reel):  {visit_count:>4d}",
                        f"║   Effectif (IA):   {effective_visits:>4d}",
                    ]

                    if effective_visits < visit_count:
                        decay = visit_count - effective_visits
                        lines.append(f"║   [v] Reduit de:     {decay:>4d}")

                    lines.append(f"║   Total (global):  {total_visits:>4d}")
                    lines.append("╠═══════════════════════════")

                    if marker_names:
                        lines.append("║ Marqueurs:")
                        for name in marker_names:
                            lines.append(f"║   • {name}")
                        lines.append("╠═══════════════════════════")

                    size_x = cube.get('size_x', 200)
                    size_y = cube.get('size_y', 200)
                    size_z = cube.get('size_z', 200)
                    lines.append("║ Dimensions:")
                    lines.append(f"║   {size_x:.0f} x {size_y:.0f} x {size_z:.0f}")

                    blocked = cube.get('blocked_directions', {})
                    obstacles = [d for d, c in blocked.items() if c >= 5]
                    if obstacles:
                        lines.append("╠═══════════════════════════")
                        lines.append(f"║ Obstacles: {', '.join(obstacles)}")

                    lines.append("╚═══════════════════════════")

                    tooltip_text = '\n'.join(lines)

                    if hasattr(self, 'map_annotation') and self.map_annotation:
                        # Mettre à jour le texte
                        self.map_annotation.set_text(tooltip_text)

                        # Afficher si caché
                        if not self.map_annotation.get_visible():
                            self.map_annotation.set_visible(True)

                        # UN SEUL draw_idle(), UNIQUEMENT si changement
                        try:
                            self.map_canvas.draw_idle()
                        except (RuntimeError, AttributeError) as draw_error1:
                            logger.error(f"draw_idle() échoué: {draw_error1}")
            else:
                # Aucun cube: cacher UNIQUEMENT si visible
                if hasattr(self, '_current_hovered_cube') and self._current_hovered_cube is not None:
                    self._current_hovered_cube = None
                    if hasattr(self,
                               'map_annotation') and self.map_annotation and self.map_annotation.get_visible():
                        self.map_annotation.set_visible(False)
                        try:
                            self.map_canvas.draw_idle()
                        except (RuntimeError, AttributeError) as draw_error2:
                            logger.error(f"draw_idle() échoué: {draw_error2}")

        # Connecter événement (motion_notify_event = mouvement souris)
        self.map_canvas.mpl_connect("motion_notify_event", on_hover)

    def _create_plots(self, parent):
        """Crée les graphiques"""
        self.fig = Figure(figsize=(10, 8), dpi=100)

        self.axes = [
            self.fig.add_subplot(3, 1, 1),
            self.fig.add_subplot(3, 1, 2),
            self.fig.add_subplot(3, 1, 3)
        ]

        self.axes[0].set_title("Reward par Episode", fontsize=12, fontweight='bold')
        self.axes[1].set_title("Longueur Episode (steps)", fontsize=12, fontweight='bold')
        self.axes[2].set_title("Hits par Episode", fontsize=12, fontweight='bold')

        for ax in self.axes:
            ax.set_xlabel("Episode")
            ax.grid(True, alpha=0.3)

        self.axes[0].set_ylabel("Reward")
        self.axes[1].set_ylabel("Steps")
        self.axes[2].set_ylabel("Hits")

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _schedule_update(self):
        """
        Schedule next update (stops immediately if closing)
        """
        # Check _closing flag FIRST to stop recursion during shutdown
        if self._closing or not self.running:
            logger.debug("GUI update loop stopped (closing or not running)")
            return

        # Check window exists before any operations
        if not self.window or not self.window.winfo_exists():
            logger.debug("GUI window destroyed, stopping updates")
            self.running = False
            return

        try:
            self._update_display()

            # Update zone stats if open
            if self.zone_stats_window and self.zone_stats_window.winfo_exists():
                if self.current_stats.get('total_steps', 0) % 5 == 0:
                    self._update_zone_stats_window()

            # Schedule next update ONLY if still running and not closing
            if self.running and not self._closing and self.window and self.window.winfo_exists():
                self.window.after(300, self._schedule_update) # refresh every 300ms

        except tk.TclError as schedule_error:
            # Window destroyed during update - stop cleanly
            logger.debug(f"TclError during update (window closed): {schedule_error}")
            self.running = False

    def _update_display(self):
        """
        Updates display
        """
        # Stop immediately if closing
        if self._closing or not self.running:
            return

        # Verify window exists
        if not self.window or not self.window.winfo_exists():
            self.running = False
            return

        try:
            # Update menu indicator
            is_menu_open = self.current_stats.get('in_game_menu', False)
            if hasattr(self, 'menu_status_indicator'):
                if is_menu_open:
                    self.menu_status_indicator.config(text="OUI", fg="#e67e22")
                else:
                    self.menu_status_indicator.config(text="NON", fg="#95a5a6")

            # Boucle
            for key, label in self.stat_labels.items():
                value = self.current_stats.get(key, 0)

                # Gérer None explicitement
                if value is None:
                    if key in ['reward', 'episode_reward']:
                        label.config(text="+0.00")
                    elif key in ['hp', 'stamina']:
                        label.config(text="0.0%")
                    else:
                        label.config(text="0")
                    continue

                if key in ['reward', 'episode_reward']:
                    label.config(text=f"{value:+.2f}")
                elif key in ['hp', 'stamina']:
                    label.config(text=f"{value:.1f}%")
                elif key == 'total_steps':
                    label.config(text=f"{value:,}")
                elif key == 'game_menu_open_count':
                    count = self.current_stats.get('game_menu_open_count', 0)
                    label.config(text=str(count))
                else:
                    label.config(text=str(value)) # Afficher step_episode

            # Temps écoulé
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.time_label.config(text=f"⏱️ Temps: {hours:02d}:{minutes:02d}:{seconds:02d}")

            # Mise à jour des graphiques
            self._update_plots()

            # Mise à jour des fenêtres étendues si ouvertes
            if self.player_window and self.player_window.winfo_exists():
                self._update_player_window()

            if self.rewards_window and self.rewards_window.winfo_exists():
                self._update_rewards_window()

                # Mise à jour carte 3D si ouverte
                if self.map_window and self.map_window.winfo_exists():
                    self._update_map_window()

        except tk.TclError as display_error:
            # Window destroyed during update - stop cleanly
            logger.debug(f"Display update error (window closing): {display_error}")
            self.running = False

        except Exception as unexpected_display_error:
            # Unexpected error - log but don't crash
            logger.error(f"Unexpected error in display update: {unexpected_display_error}")

    def _update_plots(self):
        """
        Updates charts with smart scaling
        """
        if len(self.episode_history) < 2:
            return

        episodes = list(self.episode_history)

        for ax in self.axes:
            ax.clear()

        # Plot 1: Reward avec échelle intelligente
        if len(self.reward_history) > 0:
            rewards = list(self.reward_history)

            # Calcul d'échelle intelligente
            # Exclure les outliers extrêmes (au-delà de 3 écarts-types)
            if len(rewards) >= 10:
                rewards_array = np.array(rewards)
                mean_reward = float(np.mean(rewards_array))
                std_reward = float(np.std(rewards_array))

                # Définir les limites raisonnables
                lower_bound = mean_reward - 3 * std_reward
                upper_bound = mean_reward + 3 * std_reward

                # Garder une marge de 20%
                margin = (upper_bound - lower_bound) * 0.2
                ylim_min = lower_bound - margin
                ylim_max = upper_bound + margin

                self.axes[0].set_ylim(ylim_min, ylim_max)

            self.axes[0].plot(episodes, rewards, 'b-', linewidth=2, alpha=0.6)
            self.axes[0].fill_between(episodes, rewards, alpha=0.2)
            self.axes[0].set_title("Reward par Episode (échelle auto - exclut outliers)", fontsize=12,
                                   fontweight='bold')
            self.axes[0].set_ylabel("Reward")
            self.axes[0].grid(True, alpha=0.3)

            # Moyenne mobile
            if len(rewards) >= 10:
                window = min(10, len(rewards))
                moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
                self.axes[0].plot(
                    episodes[window - 1:],
                    moving_avg,
                    'r--',
                    linewidth=2,
                    label=f'Moy. {window} ep.',
                    alpha=0.8
                )
                self.axes[0].legend()

                # Ligne de tendance (régression linéaire)
                if len(rewards) >= 20:
                    x_trend = np.arange(len(rewards))
                    z = np.polyfit(x_trend, rewards, 1)
                    p = np.poly1d(z)
                    self.axes[0].plot(
                        episodes,
                        p(x_trend),
                        'g:',
                        linewidth=1.5,
                        label='Tendance',
                        alpha=0.6
                    )
                    self.axes[0].legend()

        # Plot 2: Length
        if len(self.length_history) > 0:
            lengths = list(self.length_history)
            self.axes[1].plot(episodes, lengths, 'g-', linewidth=2)
            self.axes[1].fill_between(episodes, lengths, alpha=0.3)
            self.axes[1].set_title("Longueur Episode (steps)", fontsize=12, fontweight='bold')
            self.axes[1].set_ylabel("Steps")
            self.axes[1].grid(True, alpha=0.3)
            self.axes[1].set_ylim(bottom=0)  # Minimum à 0

        # Plot 3: Hits
        if len(self.hits_history) > 0:
            hits = list(self.hits_history)
            self.axes[2].bar(episodes, hits, color='purple', alpha=0.7)
            self.axes[2].set_title("Hits par Episode", fontsize=12, fontweight='bold')
            self.axes[2].set_ylabel("Hits")
            self.axes[2].grid(True, alpha=0.3)
            self.axes[2].set_ylim(bottom=0) # Minimum à 0

        self.canvas.draw()

    def _open_player_window(self):
        """Ouvre la fenêtre Player avec inventaire"""
        if self.player_window and self.player_window.winfo_exists():
            self.player_window.lift()
            return

        self.player_window = tk.Toplevel(self.window)
        self.player_window.title("📈 Stats Étendues - Player")
        # Appliquer géométrie sauvegardée
        saved_geometry = self.config['player_window'].get('geometry', '500x700')
        self.player_window.geometry(saved_geometry)

        tk.Label(
            self.player_window,
            text="STATISTIQUES ÉTENDUES",
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white"
        ).pack(fill=tk.X) # type: ignore

        container = tk.Frame(self.player_window, bg="#ecf0f1")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) # type: ignore

        # Stats player
        self.player_labels = {}

        player_stats = [
            ("Position X", "player_x"),
            ("Position Y", "player_y"),
            ("Position Z", "player_z"),
            ("Orientation (°)", "orientation"),
            ("Zone ID", "zone"),
            ("Money (Zennys)", "money"),
            ("Distance parcourue", "distance"),
            ("Sharpness", "sharpness"),
            ("Quest Time (s)", "quest_time"),
            ("Menu compteur", "game_menu_open_count"),
            ("Temps total menu (s)", "game_menu_total_time"),
        ]

        for label_text, key in player_stats:
            frame = tk.Frame(container, bg="white", relief=tk.RAISED, bd=1) # type: ignore
            frame.pack(fill=tk.X, pady=3) # type: ignore

            tk.Label(
                frame,
                text=label_text + ":",
                font=("Arial", 10, "bold"),
                bg="white",
                anchor="w"
            ).pack(side=tk.LEFT, padx=10, pady=6) # type: ignore

            value_label = tk.Label(
                frame,
                text="0",
                font=("Arial", 10),
                bg="white",
                fg="#2c3e50",
                anchor="e"
            )
            value_label.pack(side=tk.RIGHT, padx=10, pady=6) # type: ignore

            self.player_labels[key] = value_label

        # Section inventaire
        tk.Label(
            container,
            text="🎒 INVENTAIRE (slots non-vides)",
            font=("Arial", 11, "bold"),
            bg="#3498db",
            fg="white"
        ).pack(fill=tk.X, pady=(10, 5)) # type: ignore

        # Frame scrollable pour l'inventaire
        inventory_canvas = tk.Canvas(container, bg="white", height=200)
        inventory_scrollbar = tk.Scrollbar(container, orient="vertical", command=inventory_canvas.yview)
        self.inventory_frame = tk.Frame(inventory_canvas, bg="white")

        self.inventory_frame.bind(
            "<Configure>",
            lambda e: inventory_canvas.configure(scrollregion=inventory_canvas.bbox("all"))
        )

        inventory_canvas.create_window((0, 0), window=self.inventory_frame, anchor="nw")
        inventory_canvas.configure(yscrollcommand=inventory_scrollbar.set)

        inventory_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5) # type: ignore
        inventory_scrollbar.pack(side=tk.RIGHT, fill=tk.Y) # type: ignore

        # Pas de labels pré-créés, on les crée dynamiquement
        self.inventory_labels = []  # Sera recréé à chaque update

        self._update_player_window()

    def _update_player_window(self):
        """Met à jour la fenêtre Player avec inventaire"""
        if not self.player_window or not self.player_window.winfo_exists():
            return

        # Stats classiques
        for key, label in self.player_labels.items():
            value = self.current_stats.get(key, 0)

            if key in ['player_x', 'player_y', 'player_z']:
                label.config(text=f"{value:.2f}")
            elif key == 'orientation':
                label.config(text=f"{value:.1f}°")
            elif key == 'distance':
                label.config(text=f"{value:.1f}m")
            elif key == 'money':
                label.config(text=f"{value:,}z")
            elif key == 'game_menu_open_count':
                label.config(text=f"{int(value)} fois")
            elif key == 'game_menu_total_time':
                label.config(text=f"{value:.1f}s")
            else:
                label.config(text=str(value))

        # Mise à jour inventaire
        inventory = self.current_stats.get('inventory', [])

        # Détruire anciens labels
        for widget in self.inventory_frame.winfo_children():
            widget.destroy()

        # Créer labels uniquement pour slots remplis
        if inventory:
            for item in inventory:
                slot_num = item.get('slot')
                item_name = item.get('name', f"Item {item.get('item_id')}")
                quantity = item.get('quantity', 0)

                label = tk.Label(
                    self.inventory_frame,
                    text=f"SLOT {slot_num}: {item_name} x{quantity}",
                    font=("Arial", 9, "bold"),
                    bg="white",
                    fg="#2c3e50",
                    anchor="w"
                )
                label.pack(anchor="w", padx=10, pady=2)
        else:
            # Message si vide
            tk.Label(
                self.inventory_frame,
                text="Inventaire vide ou non détecté",
                font=("Arial", 9, "italic"),
                bg="white",
                fg="#95a5a6"
            ).pack(anchor="w", padx=10, pady=5)

    def _open_rewards_window(self):
        """
        Ouvre la fenêtre Rewards
        """
        if self.rewards_window and self.rewards_window.winfo_exists():
            self.rewards_window.lift()
            return

        self.rewards_window = tk.Toplevel(self.window)
        self.rewards_window.title("💰 Reward Breakdown")
        self.rewards_window.geometry("600x700")

        tk.Label(
            self.rewards_window,
            text="💰 REWARD BREAKDOWN (Valeurs instantanées par seconde)",
            font=("Arial", 14, "bold"),
            bg="#2ecc71",
            fg="white"
        ).pack(fill=tk.X) # type: ignore

        container = tk.Frame(self.rewards_window, bg="#ecf0f1")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) # type: ignore

        self.rewards_labels = {}

        # Total épisode
        total_frame = tk.Frame(container, bg="#34495e", relief=tk.RAISED, bd=2) # type: ignore
        total_frame.pack(fill=tk.X, pady=10) # type: ignore

        tk.Label(
            total_frame,
            text="TOTAL ÉPISODE:",
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="white",
            anchor="w"
        ).pack(side=tk.LEFT, padx=10, pady=10) # type: ignore

        self.episode_reward_label = tk.Label(
            total_frame,
            text="0.00",
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="#2ecc71",
            anchor="e"
        )
        self.episode_reward_label.pack(side=tk.RIGHT, padx=10, pady=10) # type: ignore

        # Breakdown
        tk.Label(
            container,
            text="Breakdown par catégorie:",
            font=("Arial", 11, "bold"),
            bg="#ecf0f1"
        ).pack(anchor="w", pady=(10, 5))

        # Catégories avec expansion
        categories = [
            # 🟢 Survie (simple bonus constant)
            ('survival', '🟢 Survie', '#2ecc71', None),

            # ⚔️ Combat (hits, dégâts monstres, combos)
            ('combat', '⚔️ Combat', '#9b59b6', None),

            # 🎯 Hits (séparé de combat)
            ('hit', '🎯 Hits', '#e67e22', [
                'hit_success',  # Doublons avec combat (normal)
                'combo_bonus',  # hit.combo_bonus
            ]),

            # 💥 Dégâts Monstres
            ('monster_hit', '💥 Monstres', '#c0392b', [
                'monster_damage',  # Reward pour dégâts infligés
            ]),

            # ❤️ Santé (dégâts reçus, HP critique, récupération)
            ('health', '❤️ Santé', '#e74c3c', [
                'damage_penalty',  # health.damage_penalty
                'big_hit_penalty',  # health.big_hit_penalty
                'hit_flag_penalty',  # health.hit_flag_penalty
                'low_hp_penalty',  # health.low_hp_penalty
                'critical_hp_penalty',  # health.critical_hp_penalty
                'good_health_bonus',  # health.good_health_bonus
                'buffed_hp_bonus',  # health.buffed_hp_bonus
                'hp_recovery',  # health.hp_recovery
            ]),

            # 🗺️ Exploration (découverte, revisites)
            ('exploration', '🗺️ Exploration', '#3498db', [
                'new_cube_bonus',  # exploration.new_cube_bonus
                'new_zone_bonus',  # exploration.new_zone_bonus
                'revisit_penalty',  # exploration.revisit_penalty
            ]),

            # 👹 Zones Monstres (bonus présence, pénalité fuite)
            ('monster_zone', '👹 Zones Monstres', '#8e44ad', [
                'in_zone_bonus',  # monster_zone.in_zone_bonus
                'left_zone_penalty',  # monster_zone.left_zone_penalty
                'persistence_bonus',  # monster_zone.persistence_bonus
            ]),

            # 💨 Oxygène (pénalité progressive, récupération)
            ('oxygen', '💨 Oxygène', '#1abc9c', [
                'oxygen_progressive',  # oxygen.oxygen_progressive
                'oxygen_recovery',  # oxygen.oxygen_recovery
            ]),

            # 🗺️ Changement Zone (bonus transitions)
            ('zone_change', '🗺️ Changement Zone', '#16a085', [
                'zone_bonus',  # zone_change.zone_bonus
                'first_exit',  # zone_change.first_exit
                'exit_after_death',  # zone_change.exit_after_death
            ]),

            # 💀 Morts
            ('death', '💀 Morts', '#95a5a6', [
                'death_penalty',  # penalties.death_penalty
            ]),

            # 🛡️ Actions Défensives (bloc, esquive, attaque tentée)
            ('defensive_actions', '🛡️ Actions Défensives', '#16a085', [
                'block',  # defensive_actions.block
                'dodge',  # defensive_actions.dodge
                'attack_attempt',  # defensive_actions.attack_attempt (AJOUTÉ)
            ]),

            # 🚫 Pénalités Camp
            ('camp_penalty', '🏕️ Camp', '#e67e22', [
                'camp_base',  # penalties.camp_base
                'camp_periods',  # penalties.camp_periods
            ]),

            # 📱 Pénalités Menu
            ('menu_penalty', '📱 Menu', '#d35400', [
                'menu_initial',  # penalties.menu_initial
                'menu_recurring',  # penalties.menu_recurring
            ]),

            # 🔧 Autres pénalités
            ('penalties', '🚫 Pénalités Diverses', '#7f8c8d', [
                'idle',  # penalties.idle
                'stamina_low',  # penalties.stamina_low
            ]),

            # ➕ Autres (buffs stamina, etc.)
            ('other', '➕ Autres Bonus', '#34495e', [
                'buffed_stamina_bonus',  # other.buffed_stamina_bonus
            ]),

            # 🔪 Dégâts pris (catégorie séparée pour clarté)
            ('damage_taken', '🔪 Dégâts Subis', '#c0392b', [
                'damage_penalty',
                'big_hit_penalty',
                'hit_flag_penalty',
            ]),
        ]

        self.rewards_expand_buttons = {}
        self.rewards_detail_frames = {}

        for key, label_text, color, subcategories in categories:
            # Frame principal
            main_frame = tk.Frame(container, bg="white", relief=tk.RAISED, bd=1) # type: ignore
            main_frame.pack(fill=tk.X, pady=2) # type: ignore

            # Header (avec bouton expand si sous-catégories)
            header_frame = tk.Frame(main_frame, bg="white")
            header_frame.pack(fill=tk.X) # type: ignore

            if subcategories:
                # Bouton expand/collapse
                expand_btn = tk.Button(
                    header_frame,
                    text="▶",
                    font=("Arial", 8),
                    bg="white",
                    fg=color,
                    bd=0,
                    cursor="hand2",
                    command=lambda k=key: self._toggle_reward_category(k)
                )
                expand_btn.pack(side=tk.LEFT, padx=5) # type: ignore
                self.rewards_expand_buttons[key] = expand_btn

            tk.Label(
                header_frame,
                text=label_text,
                font=("Arial", 9, "bold"),
                bg="white",
                fg=color,
                anchor="w"
            ).pack(side=tk.LEFT, padx=10 if not subcategories else 5, pady=5) # type: ignore

            value_label = tk.Label(
                header_frame,
                text="+0.00",
                font=("Arial", 9),
                bg="white",
                fg=color,
                anchor="e"
            )
            value_label.pack(side=tk.RIGHT, padx=10, pady=5) # type: ignore

            self.rewards_labels[key] = value_label

            # Frame détails (caché par défaut)
            if subcategories:
                detail_frame = tk.Frame(main_frame, bg="#ecf0f1")
                self.rewards_detail_frames[key] = {
                    'frame': detail_frame,
                    'labels': {},
                    'visible': False
                }

                for subcat in subcategories:
                    subcat_frame = tk.Frame(detail_frame, bg="#ecf0f1")
                    subcat_frame.pack(fill=tk.X, padx=20, pady=1) # type: ignore

                    tk.Label(
                        subcat_frame,
                        text=f"  • {subcat}:",
                        font=("Arial", 8),
                        bg="#ecf0f1",
                        fg="#7f8c8d",
                        anchor="w"
                    ).pack(side=tk.LEFT, padx=5) # type: ignore

                    subcat_label = tk.Label(
                        subcat_frame,
                        text="+0.00",
                        font=("Arial", 8),
                        bg="#ecf0f1",
                        fg="#7f8c8d",
                        anchor="e"
                    )
                    subcat_label.pack(side=tk.RIGHT, padx=5) # type: ignore

                    self.rewards_detail_frames[key]['labels'][subcat] = subcat_label

        # Top 3
        top_frame = tk.Frame(container, bg="#ecf0f1")
        top_frame.pack(fill=tk.X, pady=10) # type: ignore

        # Top gains
        gains_frame = tk.Frame(top_frame, bg="white", relief=tk.RAISED, bd=2) # type: ignore
        gains_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5)) # type: ignore

        tk.Label(
            gains_frame,
            text="🏆 TOP 3 GAINS",
            font=("Arial", 10, "bold"),
            bg="#2ecc71",
            fg="white"
        ).pack(fill=tk.X) # type: ignore

        self.top_gains_labels = []
        for i in range(3):
            label = tk.Label(
                gains_frame,
                text=f"{i + 1}. ---",
                font=("Arial", 9),
                bg="white",
                fg="#2ecc71",
                anchor="w"
            )
            label.pack(anchor="w", padx=10, pady=3)
            self.top_gains_labels.append(label)

        # Top pertes
        losses_frame = tk.Frame(top_frame, bg="white", relief=tk.RAISED, bd=2) # type: ignore
        losses_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0)) # type: ignore

        tk.Label(
            losses_frame,
            text="TOP 3 PERTES",
            font=("Arial", 10, "bold"),
            bg="#e74c3c",
            fg="white"
        ).pack(fill=tk.X) # type: ignore

        self.top_losses_labels = []
        for i in range(3):
            label = tk.Label(
                losses_frame,
                text=f"{i + 1}. ---",
                font=("Arial", 9),
                bg="white",
                fg="#e74c3c",
                anchor="w"
            )
            label.pack(anchor="w", padx=10, pady=3)
            self.top_losses_labels.append(label)

        self._update_rewards_window()

    def _toggle_reward_category(self, category_key: str):
        """Développe/réduit une catégorie de reward"""
        if category_key not in self.rewards_detail_frames:
            return

        detail_info = self.rewards_detail_frames[category_key]
        frame = detail_info['frame']
        is_visible = detail_info['visible']

        if is_visible:
            # Réduire
            frame.pack_forget()
            detail_info['visible'] = False
            self.rewards_expand_buttons[category_key].config(text="▶")
        else:
            # Développer
            frame.pack(fill=tk.X, after=self.rewards_expand_buttons[category_key].master)
            detail_info['visible'] = True
            self.rewards_expand_buttons[category_key].config(text="▼")

    def _update_rewards_window(self):
        """
        Met à jour la fenêtre Rewards
        """
        if not self.rewards_window or not self.rewards_window.winfo_exists():
            return

        # DEBUG CRITIQUE : Afficher TOUT current_stats
        if not hasattr(self, '_debug_shown'):
            logger.debug(f"🔍 DEBUG current_stats au démarrage:")
            logger.debug(f"Clés disponibles: {list(self.current_stats.keys())}")
            logger.debug(f"reward_breakdown présent: {'reward_breakdown' in self.current_stats}")
            if 'reward_breakdown' in self.current_stats:
                logger.debug(f"reward_breakdown contenu: {self.current_stats['reward_breakdown']}")
            self._debug_shown = True

        # 1. REWARD TOTALE DE L'ÉPISODE
        episode_reward = self.current_stats.get('episode_reward', 0.0)
        self.episode_reward_label.config(text=f"{episode_reward:+.2f}")

        # 2. RÉCUPÉRER LES BREAKDOWNS AVEC FALLBACK ROBUSTE
        breakdown_main = self.current_stats.get('reward_breakdown', {})
        breakdown_detailed = self.current_stats.get('reward_breakdown_detailed', {})

        # DEBUG : afficher une fois si vide
        if not hasattr(self, '_breakdown_debug_shown'):
            if not breakdown_main:
                logger.debug("⚠️ reward_breakdown vide dans current_stats!")
                logger.debug(f"Clés disponibles: {list(self.current_stats.keys())}")
            self._breakdown_debug_shown = True

        # 3. METTRE À JOUR CATÉGORIES PRINCIPALES AVEC CACHE
        current_time = time.time()

        for key, label in self.rewards_labels.items():
            value = breakdown_main.get(key, 0.0)

            # Gérer None explicitement
            if value is None:
                value = 0.0

            # LOGIQUE DE CACHE:
            # Afficher directement si non-nul
            if abs(value) > 0.0001:  # Seuil très bas pour détecter même petites valeurs
                self.reward_display_cache[key] = value
                self.reward_display_timestamps[key] = current_time
                sign = "+" if value >= 0 else ""
                label.config(text=f"{sign}{value:.3f}")  # 3 décimales pour voir petites valeurs
            else:
                # Valeur nulle : vérifier cache
                if key in self.reward_display_cache:
                    age = current_time - self.reward_display_timestamps.get(key, 0)

                    if age < self.reward_cache_duration:
                        # Cache encore valide, afficher ancienne valeur
                        cached_value = self.reward_display_cache[key]
                        sign = "+" if cached_value >= 0 else ""
                        label.config(text=f"{sign}{cached_value:.3f}")
                    else:
                        # Cache expiré, afficher 0
                        label.config(text="+0.000")
                        del self.reward_display_cache[key]
                        del self.reward_display_timestamps[key]
                else:
                    # Pas de cache, afficher 0
                    label.config(text="+0.000")

        # 4. METTRE À JOUR SOUS-CATÉGORIES AVEC CACHE
        for category_key, detail_info in self.rewards_detail_frames.items():
            if not detail_info['visible']:
                continue

            for subcat_key, subcat_label in detail_info['labels'].items():
                # Construire clé complète
                full_key = f"{category_key}.{subcat_key}"

                # Récupérer valeur
                value = breakdown_detailed.get(full_key, 0.0)

                if value is None:
                    value = 0.0

                # MÊME LOGIQUE DE CACHE
                # Afficher directement si non-nul
                if abs(value) > 0.0001:
                    self.reward_display_cache[full_key] = value
                    self.reward_display_timestamps[full_key] = current_time
                    sign = "+" if value >= 0 else ""
                    subcat_label.config(text=f"{sign}{value:.3f}")
                else:
                    if full_key in self.reward_display_cache:
                        age = current_time - self.reward_display_timestamps.get(full_key, 0)

                        if age < self.reward_cache_duration:
                            cached_value = self.reward_display_cache[full_key]
                            sign = "+" if cached_value >= 0 else ""
                            subcat_label.config(text=f"{sign}{cached_value:.3f}")
                        else:
                            subcat_label.config(text="+0.000")
                            del self.reward_display_cache[full_key]
                            del self.reward_display_timestamps[full_key]
                    else:
                        subcat_label.config(text="+0.000")

        # 5. TOP 3 GAINS/PERTES (aussi en /seconde)
        sorted_gains = sorted(
            [(k, v) for k, v in breakdown_main.items() if v > 0.0001],  # Seuil ajusté
            key=lambda x: x[1],
            reverse=True
        )[:3]

        sorted_losses = sorted(
            [(k, v) for k, v in breakdown_main.items() if v < -0.0001],  # Seuil ajusté
            key=lambda x: x[1]
        )[:3]

        # Noms affichables
        category_names = {
            'survival': 'Survie',
            'combat': 'Combat',
            'hit': 'Hits',
            'monster_hit': 'Monstres',
            'exploration': 'Exploration',
            'monster_zone': 'Zones Monstres',
            'damage_taken': 'Dégâts Subis',
            'health': 'Santé',
            'oxygen': 'Oxygène',
            'menu_penalty': 'Menus',
            'camp_penalty': 'Camp',
            'zone_change': 'Zones',
            'death': 'Morts',
            'defensive_actions': 'Défense',
            'penalties': 'Pénalités',
            'other': 'Autres'
        }

        # Afficher Top 3 Gains
        for i, label in enumerate(self.top_gains_labels):
            if i < len(sorted_gains):
                key, value = sorted_gains[i]
                name = category_names.get(key, key)
                label.config(text=f"{i + 1}. {name}: +{value:.3f}")
            else:
                label.config(text=f"{i + 1}. ---")

        # Afficher Top 3 Pertes
        for i, label in enumerate(self.top_losses_labels):
            if i < len(sorted_losses):
                key, value = sorted_losses[i]
                name = category_names.get(key, key)
                label.config(text=f"{i + 1}. {name}: {value:.3f}")
            else:
                label.config(text=f"{i + 1}. ---")

    def _calculate_average_breakdown(self) -> dict:
        """Calcule la moyenne du breakdown"""
        if len(self.reward_breakdown_history) == 0:
            return {}

        avg_breakdown = {}
        all_keys = set()

        for bd in self.reward_breakdown_history:
            all_keys.update(bd.keys())

        for key in all_keys:
            values = [bd.get(key, 0.0) for bd in self.reward_breakdown_history]
            avg_breakdown[key] = np.mean(values)

        return avg_breakdown

    def update_stats(self, stats: Dict):
        """
        Met à jour les stats avec reward d'épisode
        """
        #1. METTRE À JOUR current_stats D'ABORD
        prev_zone = self.current_stats.get('zone', 0)

        # Accumuler la reward de l'épisode en cours
        if 'reward' in stats:
            self.current_stats['episode_reward'] = self.current_stats.get('episode_reward', 0.0) + stats['reward']

        #2. Mettre à jour current_stats AVANT toute autre opération
        self.current_stats.update(stats)

        # S'assurer que les breakdowns sont bien présents
        if 'reward_breakdown' in stats:
            self.current_stats['reward_breakdown'] = stats['reward_breakdown'].copy()

        if 'reward_breakdown_detailed' in stats:
            self.current_stats['reward_breakdown_detailed'] = stats['reward_breakdown_detailed'].copy()

        #3. PUIS tracker breakdown
        if 'reward_breakdown_detailed' in stats:
            self.reward_breakdown_detailed_history.append(stats['reward_breakdown_detailed'])

        if 'reward_breakdown' in stats:
            self.reward_breakdown_history.append(stats['reward_breakdown'])

        #4. FORCER mise à jour zone stats (avec données à jour)
        if self.zone_stats_window and self.zone_stats_window.winfo_exists():
            self._update_zone_stats_window()

        # 5. Détecter changement de zone pour carte
        new_zone = self.current_stats.get('zone', 0)
        if prev_zone != new_zone and self.map_window and self.map_window.winfo_exists():
            self._update_map_window()

    def add_episode_data(self, episode: int, reward: float, length: int, hits: int = 0):
        """
        Ajoute un épisode et reset la reward
        """
        try:
            episode = int(episode)
        except (ValueError, TypeError):
            episode = len(self.episode_history) + 1

        try:
            reward = float(reward)
        except (ValueError, TypeError):
            reward = 0.0

        try:
            length = int(length)
        except (ValueError, TypeError):
            length = 0

        try:
            hits = int(hits)
        except (ValueError, TypeError):
            hits = 0

        self.episode_history.append(episode)
        self.reward_history.append(reward)
        self.length_history.append(length)
        self.hits_history.append(hits)

        # Reset la reward de l'épisode après l'avoir ajoutée
        self.current_stats['episode_reward'] = 0.0

    def _on_stop_clicked(self):
        """Appelé quand le bouton stop est cliqué"""
        if not self.stop_requested:
            self.stop_requested = True
            logger.info("Arrêt demandé par l'utilisateur...")

    def _on_close(self):
        """
        Called when window is closed (prevents double execution)
        """
        # Prevent double close if already closing
        if self._closing:
            logger.debug("Close already in progress, ignoring duplicate call")
            return

        logger.info("GUI close requested by user")
        self._on_stop_clicked()
        self.close()  # Use unified close() method

    def close(self):
        """
        Closes GUI properly to avoid threading errors
        """
        # Step 1: Signal shutdown to all windows
        self._save_config()
        self.running = False

        # Step 2: Clean matplotlib canvases (prevents memory leaks)
        try:
            if hasattr(self, 'map_canvas') and self.map_canvas:
                self.map_canvas.get_tk_widget().destroy()
                self.map_canvas = None
        except (tk.TclError, AttributeError, RuntimeError) as close_map_canvas_error:
            logger.debug(f"Map canvas cleanup error (non-critical): {close_map_canvas_error}")
            self.map_canvas = None

        try:
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
        except (tk.TclError, AttributeError, RuntimeError) as cleaning_canvas_error:
            logger.debug(f"Canvas cleanup error (non-critical): {cleaning_canvas_error}")
            self.canvas = None

        # Step 3: Force immediate window destruction (no after() delay)
        try:
            if self.window and self.window.winfo_exists():
                self.window.quit()  # Stop mainloop immediately
                self.window.destroy()  # Destroy window immediately
                self.window = None
        except (tk.TclError, RuntimeError, AttributeError) as window_destroy_error:
            logger.debug(f"Window destruction error (non-critical): {window_destroy_error}")
            self.window = None

        # Step 4: Clean matplotlib canvases (prevents memory leaks)
        try:
            if hasattr(self, 'map_canvas') and self.map_canvas:
                self.map_canvas.get_tk_widget().destroy()
                self.map_canvas = None
        except (tk.TclError, AttributeError, RuntimeError):
            self.map_canvas = None

        try:
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
        except (tk.TclError, AttributeError, RuntimeError):
            self.canvas = None

        # Step 5: Close all child windows
        for child_window in [self.player_window, self.rewards_window,
                             self.map_window, self.zone_stats_window]:
            if child_window:
                try:
                    if child_window.winfo_exists():
                        child_window.destroy()
                except (tk.TclError, AttributeError, RuntimeError):
                    pass

        # Step 6: Destroy main window immediately
        if self.window:
            try:
                if self.window.winfo_exists():
                    self.window.quit()  # Stop mainloop
                    self.window.update()  # Process pending events
                    self.window.destroy()  # Destroy window
            except (tk.TclError, RuntimeError, AttributeError):
                pass
            finally:
                self.window = None

        # Step 7: Wait for GUI thread to finish (non-blocking)
        if self.update_thread and self.update_thread.is_alive():
            logger.debug("Waiting for GUI thread termination...")
            self.update_thread.join(timeout=1.0)

            if self.update_thread.is_alive():
                logger.debug("GUI thread still alive after 1s (will terminate as daemon)")

        logger.info("GUI closed successfully")

    def should_stop(self) -> bool:
        """Retourne True si l'utilisateur a demandé l'arrêt"""
        return self.stop_requested

    def wait_until_closed(self):
        """Attend que la fenêtre soit fermée"""
        while self.running:
            time.sleep(0.1)