# Monster Hunter Tri — Reinforcement Learning AI

*A Deep Reinforcement Learning agent trained to play Monster Hunter Tri on the Dolphin emulator.*  
*Un agent d'apprentissage par renforcement profond entraîné à jouer à Monster Hunter Tri sur l'émulateur Dolphin.*

---

## 🌍 Language / Langue

- [🇬🇧 English Version](#-english-version)
- [🇫🇷 Version Française](#-version-française)

---

# 🇬🇧 English Version

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training Modes](#training-modes)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project trains a **PPO (Proximal Policy Optimization)** agent to play **Monster Hunter Tri** (Wii) using the **Dolphin** emulator. The AI perceives the game through a combination of screen capture and direct RAM reading, and controls the character via a virtual Xbox 360 controller.

> ⚠️ **Alpha status.** The agent learns from scratch and the reward system is still being tuned. Results are a proof of concept. See [Training Results](#monitoring) for current expectations.

> ⚠️ **Note on code comments.** A portion of comments and log messages are in French and are being progressively translated. Contributions welcome!

---

## Architecture

### Perception — Hybrid Input

The agent processes two parallel streams of information:

| Stream | Details |
|---|---|
| **Vision (CNN)** | Game frames resized to 84×84 px, processed by a convolutional network (Nature) |
| **Memory vector** | ~70 features read directly from Dolphin's RAM via Dolphin Memory Engine: HP, stamina, position, orientation, inventory, zone, quest timer, monster HP… |

### Exploration

A **15×15×4 dynamic exploration map** tracks visited zones, zone transitions, detected monsters, and water areas. It provides a structured spatial signal to help the agent break out of local optima.

### Reward System

Rewards are decomposed into multiple categories computed at each step:

- **Survival** — staying alive, managing HP and stamina
- **Combat** — hitting monsters, reducing their HP, avoiding hits
- **Exploration** — discovering new areas, zone transitions
- **Penalties** — deaths, menu abuse, idle behavior

### Multi-Agent Support

Up to **32 PPO agents** can train simultaneously across up to **16 Dolphin instances**, with three instance-sharing scenarios:

| Scenario | Description |
|---|---|
| One-to-One | 1 agent ↔ 1 instance (simplest) |
| Multi-instance | 1 agent controls N instances (majority vote) |
| Instance sharing | N agents share 1 instance (round-robin / weighted) |

---

## Project Structure

```
monster_hunter_ai/
│
├── config/
│   └── memory_addresses.py          # Dolphin RAM addresses (DME)
│
├── core/
│   ├── dynamic_memory_reader.py     # Async RAM reading
│   ├── state_fusion.py              # Vision + memory fusion
│   ├── controller.py                # WiiController (vgamepad / pynput)
│   └── exploration_map_incremental.py
│
├── vision/
│   ├── frame_capture.py             # FrameCapture (GDI + DLL)
│   ├── preprocessing.py             # Crop, resize, normalize
│   └── feature_extractor.py        # CNN architectures
│
├── environment/
│   ├── mh_env.py                    # Main Gymnasium environment
│   ├── reward_calculator.py
│   ├── exploration_tracker.py
│   └── cube_markers.py
│
├── agent/
│   └── ppo_agent.py                 # PPO agent (Stable-Baselines3)
│
├── utils/
│   ├── multi_agent_scheduler.py
│   ├── multi_agent_trainer.py
│   ├── genetic_trainer.py
│   ├── hidhide_manager.py
│   ├── training_gui.py              # Real-time GUI
│   ├── advanced_logging.py
│   ├── module_logger.py
│   └── safe_float.py
│
├── train.py                         # Main entry point
├── check_setup.py                   # Setup diagnostic tool  ← run this first
├── launch_dolphin_instances.ps1     # PowerShell multi-instance launcher
└── requirements.txt
```

---

## Requirements

### Software

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.8+ | Tested on 3.10 |
| Dolphin Emulator | 2509+ | [dolphin-emu.org](https://dolphin-emu.org) |
| Dolphin Memory Engine | 1.3.0-preview2+ | |
| Monster Hunter Tri | NTSC-U or PAL | ISO / WBFS / RVZ |
| ViGEmBus | latest | Required for virtual controller |
| HidHide | latest | Optional — multi-instance controller isolation |

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 6+ cores (multi-instance) |
| RAM | 8 GB | 16 GB+ |
| GPU | — | NVIDIA GTX 1060+ (CUDA) |
| Storage | 10 GB free | — |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dmsday/Monster-Hunter-Tri-IA.git
cd Monster-Hunter-Tri-IA
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

**GPU support (CUDA 12.4):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install ViGEmBus

1. Download [ViGEmBus](https://github.com/nefarius/ViGEmBus/releases) → run `ViGEmBusSetup_x64.msi` as Administrator
2. **Restart Windows** (mandatory)

### 4. Install HidHide *(optional — multi-instance only)*

1. Download [HidHide](https://github.com/ViGEm/HidHide/releases) → install as Administrator
2. **Restart Windows**

### 5. Configure Dolphin

#### Enable portable mode

Navigate to your Dolphin folder and create an empty file named `portable.txt` at the root. This forces Dolphin to store its configuration in a local `User/` subfolder instead of `AppData`, which is required for multi-instance setups.

#### Controller setup

1. Launch Dolphin → **Options → Controller Settings**
2. **Wiimote 1 → Emulated Wiimote → Configure**
3. Set **Device** to `WGInput/0/Xbox 360 Controller for Windows`

> ⚠️ The virtual controller must already exist for Dolphin to see it in the list.  
> Run `python core/controller.py` first to create it, then reopen the controller config window.

#### Graphics settings

- **Backend:** `Direct3D 11` or `Vulkan` (not OpenGL)
- **Mode:** Windowed (not fullscreen)
- **Render to Main Window:** disabled

### 6. Verify your setup

```bash
python check_setup.py           # Static checks — no Dolphin needed
python check_setup.py --full    # Full checks — Dolphin must be running
```

---

## Configuration

### Single-instance (default)

No configuration needed beyond Dolphin setup above. Run directly:

```bash
python train.py --timesteps 100000 --name my_first_run
```

### Multi-instance

#### Prepare user folders

For each additional Dolphin instance, duplicate the `User/` folder:

```bat
cd "C:\Path\To\Dolphin"
xcopy User User1 /E /I
xcopy User User2 /E /I
```

#### Configure the PowerShell launcher

Copy `launch_dolphin_instances.ps1` to your Dolphin folder and edit the paths:

```powershell
$DolphinPath = "C:\Path\To\Dolphin\Dolphin.exe"
$GameISO    = "C:\Path\To\MHTri.rvz"
```

Test the script:

```powershell
powershell -ExecutionPolicy Bypass -File launch_dolphin_instances.ps1
```

#### Create a save state

Load Monster Hunter Tri, enter a quest, then create a save state in slot 5 (**Emulation → Save State → Slot 5**). The training script will reload from this state at the beginning of each episode.

---

## Usage

### Quick start

```bash
# Verify setup
python check_setup.py

# Basic training (single instance, 1M steps)
python train.py --timesteps 1000000 --name my_run
```

### All training options

```bash
python train.py --help
```

#### Common arguments

| Argument | Default | Description |
|---|---|---|
| `--timesteps N` | 100 000 | Total training steps |
| `--name TEXT` | auto | Experiment name |
| `--lr FLOAT` | 0.0001 | Learning rate |
| `--save-state N` | 5 | Save state slot to reload each episode (1–8) |
| `--resume PATH` | — | Resume from checkpoint `.zip` |
| `--grayscale` | off | Use grayscale frames (faster) |
| `--rtvision` | off | Show real-time vision window |
| `--rtminimap` | off | Show real-time exploration map |
| `--cpu` | off | Force CPU training |
| `--keyboard` | off | Use keyboard instead of virtual controller |

#### Multi-agent arguments

| Argument | Default | Description |
|---|---|---|
| `--num-agents N` | 1 | Number of PPO agents (1–32) |
| `--num-instances N` | 1 | Number of Dolphin instances (1–16) |
| `--multi-agent-mode MODE` | independent | `independent` / `round_robin` / `majority_vote` / `weighted` / `genetic` |
| `--allocation-mode MODE` | auto | `auto` / `manual` / `weighted` |
| `--steps-per-agent N` | 4096 | Steps collected per agent per update |

#### Genetic algorithm arguments

| Argument | Default | Description |
|---|---|---|
| `--genetic-generations N` | 10 | Number of generations |
| `--genetic-elite-ratio F` | 0.25 | Fraction of top agents preserved |
| `--genetic-mutation-rate F` | 0.3 | Weight mutation rate |

### Examples

```bash
# 8 agents on 4 instances, round-robin
python train.py \
  --num-agents 8 --num-instances 4 \
  --multi-agent-mode round_robin \
  --timesteps 2000000

# Genetic algorithm, 16 agents on 8 instances
python train.py \
  --num-agents 16 --num-instances 8 \
  --multi-agent-mode genetic \
  --genetic-generations 10 \
  --timesteps 5000000

# Resume from checkpoint with real-time vision
python train.py \
  --resume models/my_run/checkpoint_500000.zip \
  --timesteps 500000 \
  --rtvision --rtminimap
```

### Testing a trained agent

```bash
python test.py \
  --model-path models/my_run/final_model.zip \
  --n-episodes 10 \
  --deterministic
```

---

## Training Modes

| Mode | Description | Best for |
|---|---|---|
| `independent` | Each agent trains separately on its own instance | Baseline, single instance |
| `round_robin` | Agents take turns collecting steps | Balanced multi-agent |
| `majority_vote` | Multiple agents vote on each action | Robust collective behavior |
| `weighted` | Allocation adapts to each agent's performance | Automatic curriculum |
| `genetic` | Evolutionary selection across generations | Long-run optimization |

---

## Monitoring

### Real-time GUI

The training GUI shows (when active):
- Episode reward and length over time
- Player stats: HP, stamina, deaths
- Reward breakdown by category
- 3D exploration map
- Monster HP

Press the **Stop** button for a clean shutdown with model save.

### TensorBoard

```bash
tensorboard --logdir ./logs/
```

Tracks: policy loss, value loss, entropy, learning rate, episode stats.

### Expected learning phases

The agent is still in early development. As of v0.1-alpha, expected behavior:
- **0–100k steps:** random exploration, frequent deaths
- **100k–500k steps:** basic survival behavior emerging
- **500k+ steps:** early combat patterns, zone navigation

Results vary significantly depending on hardware, number of agents, and reward tuning.

---

## Troubleshooting

### Virtual controller not working

1. ViGEmBus installed and Windows restarted?
2. Run `python core/controller.py` to create the virtual controller
3. Reopen Dolphin controller settings — the Xbox 360 device should appear
4. **Options → Interface → uncheck** "Keyboard shortcuts require window focus"
5. Click the game window before starting training (single instance)

### Black or empty frames

- Keep the Dolphin window **visible** — do not minimize it
- Use **windowed mode**, not fullscreen
- Set graphics backend to `Direct3D 11` or `Vulkan`
- Disable **Render to Main Window**

### CUDA out of memory

```bash
python train.py --grayscale --cpu   # fallback to CPU
python train.py --grayscale         # reduce frame size
```

### Multi-instance: all agents capture the same window

The PowerShell launcher renames Dolphin windows to `MHTri-0`, `MHTri-1`, etc. If this fails:

1. Check PowerShell output for errors
2. Increase the `--dolphin-delay` parameter in the script
3. Verify that `portable.txt` exists in the Dolphin folder

---

## Contributing

Contributions are welcome. Priority areas:

- **Translation** — converting French comments and log messages to English
- **Memory addresses** — finding missing addresses (`LMONSTER1_HP`, `LMONSTER1_POS`, `SMONSTER_NUMBER`…)
- **Reward tuning** — improving the reward signal for faster learning
- **Testing** — unit tests, integration tests
- **Documentation** — tutorials, setup guides, video walkthroughs

```bash
git checkout -b feature/my-feature
git commit -am "Add my feature"
git push origin feature/my-feature
# Then open a Pull Request
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Capcom** — Monster Hunter Tri
- **Dolphin Emulator Team** — [dolphin-emu.org](https://dolphin-emu.org)
- **Stable-Baselines3** — PPO implementation
- **nefarius / ViGEmBus** — virtual controller driver
- **OpenAI** — PPO algorithm
- The reinforcement learning community

---
---

# 🇫🇷 Version Française

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture-1)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation-1)
- [Configuration](#configuration-1)
- [Utilisation](#utilisation)
- [Modes d'entraînement](#modes-dentraînement)
- [Monitoring](#monitoring-1)
- [Résolution de problèmes](#résolution-de-problèmes)
- [Contribuer](#contribuer)

---

## Vue d'ensemble

Ce projet entraîne un agent **PPO (Proximal Policy Optimization)** à jouer à **Monster Hunter Tri** (Wii) via l'émulateur **Dolphin**. L'IA perçoit le jeu à travers une capture d'écran et une lecture directe de la RAM, et contrôle le personnage via une manette Xbox 360 virtuelle.

> ⚠️ **Statut alpha.** L'agent apprend de zéro et le système de récompenses est encore en cours de réglage. Les résultats constituent une preuve de concept.

> ⚠️ **Note sur les commentaires.** Une partie des commentaires et messages de log est en français et en cours de traduction progressive vers l'anglais.

---

## Architecture

### Perception — Entrée hybride

L'agent traite deux flux d'information en parallèle :

| Flux | Détails |
|---|---|
| **Vision (CNN)** | Frames de jeu redimensionnées à 84×84 px, traitées par un réseau convolutif (Nature DQN / IMPALA / Minigrid) |
| **Vecteur mémoire** | ~70 features lues directement dans la RAM de Dolphin via Dolphin Memory Engine : PV, endurance, position, orientation, inventaire, zone, chrono quête, PV monstres… |

### Exploration

Une **carte d'exploration dynamique 15×15×4** trace les zones visitées, les transitions de zone, les monstres détectés et les zones d'eau. Elle fournit un signal spatial structuré pour aider l'agent à sortir des optima locaux.

### Système de récompenses

Les récompenses sont décomposées en plusieurs catégories calculées à chaque pas :

- **Survie** — rester en vie, gérer les PV et l'endurance
- **Combat** — toucher les monstres, réduire leurs PV, éviter les coups
- **Exploration** — découvrir de nouvelles zones, transitions
- **Pénalités** — morts, abus de menu, comportement passif

### Support multi-agent

Jusqu'à **32 agents PPO** peuvent s'entraîner simultanément sur jusqu'à **16 instances Dolphin**, avec trois scénarios de répartition :

| Scénario | Description |
|---|---|
| Un pour un | 1 agent ↔ 1 instance (le plus simple) |
| Multi-instances | 1 agent contrôle N instances (vote majoritaire) |
| Partage d'instance | N agents partagent 1 instance (round-robin / pondéré) |

---

## Structure du projet

```
monster_hunter_ai/
│
├── config/
│   └── memory_addresses.py          # Adresses RAM Dolphin (DME)
│
├── core/
│   ├── dynamic_memory_reader.py     # Lecture RAM asynchrone
│   ├── state_fusion.py              # Fusion vision + mémoire
│   ├── controller.py                # WiiController (vgamepad / pynput)
│   └── exploration_map_incremental.py
│
├── vision/
│   ├── frame_capture.py             # FrameCapture (GDI + DLL)
│   ├── preprocessing.py             # Recadrage, resize, normalisation
│   └── feature_extractor.py        # Architectures CNN
│
├── environment/
│   ├── mh_env.py                    # Environnement Gymnasium principal
│   ├── reward_calculator.py
│   ├── exploration_tracker.py
│   └── cube_markers.py
│
├── agent/
│   └── ppo_agent.py                 # Agent PPO (Stable-Baselines3)
│
├── utils/
│   ├── multi_agent_scheduler.py
│   ├── multi_agent_trainer.py
│   ├── genetic_trainer.py
│   ├── hidhide_manager.py
│   ├── training_gui.py              # Interface temps réel
│   ├── advanced_logging.py
│   ├── module_logger.py
│   └── safe_float.py
│
├── train.py                         # Point d'entrée principal
├── check_setup.py                   # Outil de diagnostic  ← lancer en premier
├── launch_dolphin_instances.ps1     # Lanceur multi-instances PowerShell
└── requirements.txt
```

---

## Prérequis

### Logiciels

| Prérequis | Version | Notes |
|---|---|---|
| Python | 3.8+ | Testé sur 3.10 |
| Dolphin Emulator | 2509+ | [dolphin-emu.org](https://dolphin-emu.org) |
| Dolphin Memory Engine | 1.3.0-preview2+ | |
| Monster Hunter Tri | NTSC-U ou PAL | ISO / WBFS / RVZ |
| ViGEmBus | dernière version | Requis pour la manette virtuelle |
| HidHide | dernière version | Optionnel — isolation manettes multi-instances |

### Matériel

| Composant | Minimum | Recommandé |
|---|---|---|
| CPU | 4 cœurs | 6+ cœurs (multi-instances) |
| RAM | 8 Go | 16 Go+ |
| GPU | — | NVIDIA GTX 1060+ (CUDA) |
| Stockage | 10 Go libres | — |

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Dmsday/Monster-Hunter-Tri-IA.git
cd Monster-Hunter-Tri-IA
```

### 2. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

**Avec GPU (CUDA 12.4) :**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CPU uniquement :**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Installer ViGEmBus

1. Télécharger [ViGEmBus](https://github.com/nefarius/ViGEmBus/releases) → exécuter `ViGEmBusSetup_x64.msi` en tant qu'Administrateur
2. **Redémarrer Windows** (obligatoire)

### 4. Installer HidHide *(optionnel — multi-instances uniquement)*

1. Télécharger [HidHide](https://github.com/ViGEm/HidHide/releases) → installer en tant qu'Administrateur
2. **Redémarrer Windows**

### 5. Configurer Dolphin

#### Activer le mode portable

Créer un fichier vide nommé `portable.txt` à la racine du dossier Dolphin. Cela force Dolphin à stocker sa configuration dans un sous-dossier `User/` local — indispensable pour le multi-instances.

#### Configurer la manette

1. Dolphin → **Options → Paramètres manette**
2. **Wiimote 1 → Wiimote émulé → Configurer**
3. Choisir le **périphérique** : `WGInput/0/Xbox 360 Controller for Windows`

> ⚠️ La manette virtuelle doit déjà exister pour apparaître dans la liste.  
> Lancer `python core/controller.py` d'abord, puis rouvrir la fenêtre de configuration.

#### Paramètres graphiques

- **Backend :** `Direct3D 11` ou `Vulkan` (pas OpenGL)
- **Mode :** Fenêtré (pas plein écran)
- **Rendu dans la fenêtre principale :** désactivé

### 6. Vérifier l'installation

```bash
python check_setup.py           # Vérifications statiques — Dolphin inutile
python check_setup.py --full    # Vérification complète — Dolphin doit tourner
```

---

## Configuration

### Instance unique (défaut)

Aucune configuration supplémentaire. Lancer directement :

```bash
python train.py --timesteps 100000 --name premier_run
```

### Multi-instances

#### Préparer les dossiers utilisateurs

Pour chaque instance Dolphin supplémentaire, dupliquer le dossier `User/` :

```bat
cd "C:\Chemin\Vers\Dolphin"
xcopy User User1 /E /I
xcopy User User2 /E /I
```

#### Configurer le lanceur PowerShell

Copier `launch_dolphin_instances.ps1` dans le dossier Dolphin et modifier les chemins :

```powershell
$DolphinPath = "C:\Chemin\Vers\Dolphin\Dolphin.exe"
$GameISO    = "C:\Chemin\Vers\MHTri.rvz"
```

Tester le script :

```powershell
powershell -ExecutionPolicy Bypass -File launch_dolphin_instances.ps1
```

#### Créer un état de sauvegarde

Charger Monster Hunter Tri, entrer dans une quête, puis créer un état de sauvegarde dans le slot 5 (**Émulation → Sauvegarder l'état → Slot 5**). Le script d'entraînement rechargera depuis cet état au début de chaque épisode.

---

## Utilisation

### Démarrage rapide

```bash
# Vérifier l'installation
python check_setup.py

# Entraînement basique (instance unique, 1M steps)
python train.py --timesteps 1000000 --name mon_run
```

### Toutes les options

```bash
python train.py --help
```

#### Arguments courants

| Argument | Défaut | Description |
|---|---|---|
| `--timesteps N` | — | Nombre total de pas d'entraînement |
| `--name TEXT` | auto | Nom de l'expérience |
| `--lr FLOAT` | 0.0001 | Taux d'apprentissage |
| `--save-state N` | 5 | Slot d'état de sauvegarde à recharger (1–8) |
| `--resume PATH` | — | Reprendre depuis un checkpoint `.zip` |
| `--grayscale` | off | Frames en niveaux de gris (plus rapide) |
| `--rtvision` | off | Fenêtre vision temps réel |
| `--rtminimap` | off | Carte d'exploration temps réel |
| `--cpu` | off | Forcer l'entraînement sur CPU |
| `--keyboard` | off | Utiliser le clavier au lieu de la manette |

#### Arguments multi-agents

| Argument | Défaut | Description |
|---|---|---|
| `--num-agents N` | 1 | Nombre d'agents PPO (1–32) |
| `--num-instances N` | 1 | Nombre d'instances Dolphin (1–16) |
| `--multi-agent-mode MODE` | independent | `independent` / `round_robin` / `majority_vote` / `weighted` / `genetic` |
| `--allocation-mode MODE` | auto | `auto` / `manual` / `weighted` |
| `--steps-per-agent N` | 4096 | Pas collectés par agent avant mise à jour |

#### Arguments algorithme génétique

| Argument | Défaut | Description |
|---|---|---|
| `--genetic-generations N` | 10 | Nombre de générations |
| `--genetic-elite-ratio F` | 0.25 | Fraction des meilleurs agents conservés |
| `--genetic-mutation-rate F` | 0.3 | Taux de mutation des poids |

### Exemples

```bash
# 8 agents sur 4 instances, round-robin
python train.py \
  --num-agents 8 --num-instances 4 \
  --multi-agent-mode round_robin \
  --timesteps 2000000

# Algorithme génétique, 16 agents sur 8 instances
python train.py \
  --num-agents 16 --num-instances 8 \
  --multi-agent-mode genetic \
  --genetic-generations 10 \
  --timesteps 5000000

# Reprendre un checkpoint avec vision temps réel
python train.py \
  --resume models/mon_run/checkpoint_500000.zip \
  --timesteps 500000 \
  --rtvision --rtminimap
```

### Tester un agent entraîné

```bash
python test.py \
  --model-path models/mon_run/final_model.zip \
  --n-episodes 10 \
  --deterministic
```

---

## Modes d'entraînement

| Mode | Description | Idéal pour |
|---|---|---|
| `independent` | Chaque agent s'entraîne séparément sur sa propre instance | Baseline, instance unique |
| `round_robin` | Les agents alternent pour collecter des pas | Multi-agent équilibré |
| `majority_vote` | Plusieurs agents votent pour chaque action | Comportement collectif robuste |
| `weighted` | La répartition s'adapte aux performances de chaque agent | Curriculum automatique |
| `genetic` | Sélection évolutionnaire entre générations | Optimisation long terme |

---

## Monitoring

### Interface temps réel (GUI)

Quand active, l'interface affiche :
- Récompense et durée des épisodes
- Stats joueur : PV, endurance, morts
- Décomposition des récompenses par catégorie
- Carte d'exploration 3D
- PV des monstres

Appuyer sur **Stop** pour un arrêt propre avec sauvegarde du modèle.

### TensorBoard

```bash
tensorboard --logdir ./logs/
```

Suivi : perte de politique, perte de valeur, entropie, taux d'apprentissage, stats épisodes.

### Phases d'apprentissage attendues

En v0.1-alpha :
- **0–100k pas :** exploration aléatoire, morts fréquentes
- **100k–500k pas :** comportements de survie basiques émergents
- **500k+ pas :** patterns de combat précoces, navigation de zones

Les résultats varient significativement selon le matériel, le nombre d'agents et le réglage des récompenses.

---

## Résolution de problèmes

### La manette virtuelle ne fonctionne pas

1. ViGEmBus est-il installé et Windows redémarré ?
2. Lancer `python core/controller.py` pour créer la manette virtuelle
3. Rouvrir les paramètres manette Dolphin — le périphérique Xbox 360 doit apparaître
4. **Options → Interface → décocher** « Les raccourcis clavier nécessitent le focus de la fenêtre »
5. Cliquer sur la fenêtre de jeu avant de lancer l'entraînement (instance unique)

### Frames noires ou vides

- Garder la fenêtre Dolphin **visible** — ne pas la minimiser
- Utiliser le **mode fenêtré**, pas plein écran
- Backend graphique : `Direct3D 11` ou `Vulkan`
- Désactiver **Rendu dans la fenêtre principale**

### CUDA manque de mémoire

```bash
python train.py --grayscale --cpu   # repli sur CPU
python train.py --grayscale         # réduire la taille des frames
```

### Multi-instances : tous les agents capturent la même fenêtre

Le lanceur PowerShell renomme les fenêtres Dolphin en `MHTri-0`, `MHTri-1`, etc. En cas d'échec :

1. Vérifier la sortie PowerShell pour les erreurs
2. Augmenter le paramètre `--dolphin-delay` dans le script
3. Vérifier que `portable.txt` existe dans le dossier Dolphin

---

## Contribuer

Les contributions sont les bienvenues. Axes prioritaires :

- **Traduction** — convertir les commentaires et logs français en anglais
- **Adresses mémoire** — trouver les adresses manquantes (`LMONSTER1_HP`, `LMONSTER1_POS`, `SMONSTER_NUMBER`…)
- **Réglage des récompenses** — améliorer le signal de récompense pour un apprentissage plus rapide
- **Tests** — tests unitaires, tests d'intégration
- **Documentation** — tutoriels, guides de configuration

```bash
git checkout -b feature/ma-feature
git commit -am "Ajout ma feature"
git push origin feature/ma-feature
# Ouvrir une Pull Request
```

---

*For license and acknowledgments, see the [English section](#license) above.*
