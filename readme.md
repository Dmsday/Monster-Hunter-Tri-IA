# Monster Hunter Tri AI — Reinforcement Learning Project

Un projet d'IA utilisant l'apprentissage par renforcement pour maîtriser Monster Hunter Tri sur émulateur Dolphin.  
A Reinforcement Learning AI project to master Monster Hunter Tri on the Dolphin emulator.

---

## 🌍 Languages / Langues

- [English 🇬🇧](#english-version)
- [Français 🇫🇷](#version-française)

---

# English Version

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Python Dependencies](#python-dependencies)
  - [Dolphin Configuration](#dolphin-configuration)
  - [Build the DLLs](#build-the-dlls)
  - [Multi-Instance Setup](#multi-instance-setup)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project trains a Deep Reinforcement Learning agent to play **Monster Hunter Tri** using **PPO (Proximal Policy Optimization)**. The AI learns by combining three modalities:

- **Vision (CNN):** Processes game frames (84×84 pixels, 4-frame stack)
- **Memory Reading:** Direct RAM access via Dolphin Memory Engine (70-feature vector)
- **DLL-based Input:** Focus-free key injection via a custom Rust DLL hooked into Dolphin's DInput layer

> **⚠️ Note on Comments:** Code comments are primarily in French. English translation is in progress. Feel free to ask questions or contribute!

---

## Key Features

### Hybrid Observation Architecture
| Modality        | Shape                       | Description                                                                               |
|-----------------|-----------------------------|-------------------------------------------------------------------------------------------|
| Visual          | `(84, 84, 12)`              | 4 stacked RGB frames, preprocessed by GPU                                                 |
| Memory          | `(70,)`                     | HP, stamina, position, orientation, zone, inventory (24 slots), monsters, sharpness, etc. |
| Exploration map | `(15, 15, 4)`               | Local minimap with 4 channels: visits, player position, recent cubes, markers             |

> **Action head coordination:** The 7 heads use **Transformer self-attention** (2 layers, 4 attn heads) to coordinate decisions before producing logits. See [Action heads](#action-heads) for details

### Multi-Head Action Space
The agent controls **7 action heads** simultaneously via `MultiDiscrete([5, 5, 5, 2, 3, 8, 2])`:

| Head                           | Actions | Description                                                                                                 |
|--------------------------------|---------|-------------------------------------------------------------------------------------------------------------|
| Movement                       | 5       | nothing / forward / backward / strafe L / strafe R                                                          |
| Camera                         | 5       | nothing / up / down / left / right                                                                          |
| Combat                         | 5       | nothing / attack / dodge / draw-sheath / kick                                                               |
| Use Item                       | 2       | nothing / use                                                                                               |
| Select Item                    | 3       | nothing / radial left / radial right                                                                        |
| Menu (**disabled by default**) | 8       | nothing / start / nav up/down/left/right / confirm / back — re-enable with `--disabled-heads` (no argument) |
| Sprint                         | 2       | nothing / sprint                                                                                            |

Keys are **held across steps** (hold/release model) — no jerky single-frame taps.

#### Transformer Cross-Attention (default)

By default, the 7 action heads are **not independent**: they coordinate decisions through a **Transformer self-attention** module (`TransformerActionHead`). Instead of 7 separate Linear layers reading the same latent vector (like before), each head receives its own token embedding with a positional identity, then all 7 tokens pass through 2 layers of multi-head self-attention before producing logits.

This allows heads to learn cross-head coordination — e.g. "don't sprint while attacking", "don't open menu while dodging" — **proactively in the network**, rather than relying solely on post-hoc compatibility masking.

| Property        | Value                                |
|-----------------|--------------------------------------|
| Architecture    | Pre-norm self-attention, 2 layers    |
| Embedding dim   | 48 per head token                    |
| Attention heads | 4                                    |
| Parameters      | ~130K (vs ~8K for standard Linear)   |
| Overhead        | ~0.1ms per forward pass (negligible) |

To disable the Transformer and revert to standard independent Linear heads, use `--no-transformer-heads`.

### Focus-Free DLL Injection
Inputs are injected directly into Dolphin's `IDirectInputDevice8::GetDeviceState` via a Rust DLL (`dolphin_input_hook.dll`). No ViGEmBus, no HidHide, no window focus required. Multiple instances can run fully minimized.

**Complete input isolation per instance:** Each Dolphin instance has its own injected DLL with its own shared memory channel. Agent inputs (keyboard, mouse buttons) are written directly into each instance's DInput buffer — they **never reach your real OS keyboard/mouse stack**. Conversely, your own keyboard and mouse inputs are **blocked from reaching the Dolphin instances**, so you can work normally on your PC while agents train in the background. This was the main limitation in earlier versions.

**Audio disabled automatically:** When launching via the PowerShell script, each instance's `Dolphin.ini` is patched with `[DSP] Backend = No audio` so no instance emits sound during training.

### Multi-Agent Support
- Up to **32 PPO agents** training simultaneously
- Up to **16 Dolphin instances**
- Flexible allocation modes: one-to-one, multiple instances per agent, instance sharing
- **FedAvg weight synchronization** across agents every N update cycles

### Training Modes
| Mode            | Description                                            |
|-----------------|--------------------------------------------------------|
| `independent`   | Each agent acts independently on its assigned instance |
| `round_robin`   | Agents alternate in blocks of N steps                  |
| `majority_vote` | All agents predict; most common action wins            |
| `weighted`      | Allocation adapts based on per-agent episode rewards   |
| `genetic`       | Evolutionary selection with mutation and crossover     |

### Reward System
Multi-category reward with separate trackers:
- Combat hits, exploration (cube discovery), zone changes
- Damage taken, death penalty, idle/stationary penalty, menu penalty
- Camp timer, monster zone presence, oxygen (underwater), sharpness

---

## Project Structure

The full annotated project tree is available in **`structure-en.txt`** (English) and **`structure-fr.txt`** (French) at the root of the repository. It lists every file and folder with its purpose.

The two directories below are **not present in the repository** — they are created automatically once training starts (or is interrupted):

- `logs/<experiment>/<timestamp>/` — per-agent/per-env log files, reward breakdowns, session summaries
- `models/<experiment>/<timestamp>/` — checkpoints, final model, interrupted saves

## Requirements

### Software
| Software           | Version       | Notes                                                               |
|--------------------|---------------|---------------------------------------------------------------------|
| Python             | 3.8+          | Tested with 3.10                                                    |
| Dolphin Emulator   | 2509+         | Standard launch, no admin rights needed                             |
| Monster Hunter Tri | NTSC-U or PAL | ISO / WBFS / RVZ                                                    |
| Rust / cargo       | latest        | Only if `dolphin_input_hook.dll` is missing (pre-compiled included) |
| Visual Studio C++  | 2019 or 2022  | Only if `DolphinCapture.dll` is missing (pre-compiled included)     |

> **No ViGEmBus, no HidHide, no vgamepad** — the new DLL injection system requires none of these.

### Hardware
| Component | Minimum | Recommended               |
|-----------|---------|---------------------------|
| CPU       | 4 cores | 8+ cores (multi-instance) |
| RAM       | 8 GB    | 16 GB+                    |
| GPU       | —       | NVIDIA GTX 1060+ (CUDA)   |
| Storage   | 5 GB    | 15 GB+ (logs + models)    |

---

## Installation

### Python Dependencies

```bash
pip install -r requirements.txt
```

For GPU (CUDA 12.4):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Build the DLLs

Both DLLs are **already pre-compiled** and included in the repository (`vision/` folder). You do **not** need Visual Studio or Rust unless the files are missing or you explicitly want to rebuild from source.

**1. `dolphin_input_hook.dll`** (Rust — input injection)
```bash
# Requires: cargo (https://rustup.rs)
cd hook
cargo build --release
copy target\release\dolphin_input_hook.dll ..\vision\
```
> Alternatively, just run `train.py` — it will auto-build via `dll_utils.py` if cargo is in PATH.

**2. `DolphinCapture.dll`** (C++ — screen capture)
```bash
# Requires: Visual Studio 2019/2022 with "Desktop development with C++"
cd _build
python build_dll.py
# Output: DolphinCapture.dll → copy to vision/
```
> Without this DLL, frame capture falls back to GDI (Dolphin window must remain visible).

### Dolphin Configuration

**Step 1: Basic Setup**
1. Download Dolphin 2509+ from [dolphin-emu.org](https://dolphin-emu.org)
2. Create `portable.txt` in the Dolphin root folder (enables per-folder User profiles for multi-instance)
3. Launch Dolphin once to generate the `User/` folder, then close it

**Step 2: Graphics Settings**
- Config → Interface: **uncheck** "Confirm on Stop"
- Config → Interface: **uncheck** "Pause on Focus Lost"
- GFX.ini → `[General]` → `RenderToMain = False` (auto-set by the launcher)

**Step 3: Controller Configuration (Wiimote → Keyboard mapping)**

Go to **Controllers → Wiimote 1 → Configure → Device: DInput/0/Keyboard Mouse** and verify that the following mappings are set:

| Wiimote / Nunchuk input  | DInput key    |
|--------------------------|---------------|
| Nunchuk stick ↑          | `Z`           |
| Nunchuk stick ↓          | `S`           |
| Nunchuk stick ←          | `Q`           |
| Nunchuk stick →          | `D`           |
| D-pad ↑ (camera)         | `Up arrow`    |
| D-pad ↓ (camera)         | `Down arrow`  |
| D-pad ← (camera)         | `Left arrow`  |
| D-pad → (camera)         | `Right arrow` |
| Button A (attack)        | `Mouse Left`  |
| Button B (dodge)         | `Mouse Right` |
| Button 2                 | `E`           |
| Button 1                 | `A`           |
| Button +                 | `P`           |
| Button -                 | `M`           |
| Nunchuk C (draw/sheath)  | `Shift`       |
| Nunchuk Z (sprint/block) | `Left Ctrl`   |

> The DLL injects these keyboard/mouse inputs directly into each Dolphin instance's DInput buffer. Your own keyboard and mouse inputs are blocked from reaching the instances.

>NOTE : On an AZERTY keyboard, Dolphin interprets inputs as if the keyboard were QWERTY, so the displayed keys may not match the physical ones. As long as the key lights up red in the configuration when you press it, the mapping is correct.

**Step 4: Dolphin path**
- On first launch of `train.py`, you'll be prompted for the Dolphin folder path.
- The path is saved to `config/dolphin_path_config.json` for future runs.
- Or pass it directly: `--dolphin-path "C:\Dolphin-x64"`

**Step 5: ROM path (automatic)**
- The ROM (`.rvz`, `.iso`, `.wbfs`) is auto-detected from `Jeux/`, `Games/`, or `ROMs/` folders next to the Dolphin directory.
- The path is saved to `config/rom_path_config.json` for future runs.
- Or pass it directly: `--rom-path "D:\Games\MonsterHunterTri.rvz"`

**Step 6: HUD Crop Calibration (optional)**
```bash
python vision/hud_crop_tuner.py
# Interactive OpenCV tool — adjust crop to remove HP/minimap HUDs
# Saves to config/crop_config.json
```

### Multi-Instance Setup

**User folder preparation** is handled automatically by `launch_dolphin_instances.ps1`.  
When `train.py` launches multiple instances, it calls the PowerShell script which:
1. Detects if `User1/`, `User2/`, etc. folders exist
2. **Auto-creates missing ones** by copying the base `User/` folder
3. Disables audio (`[DSP] Backend = No audio`) per instance
4. Renames render windows to `MHTri-0`, `MHTri-1`, etc. for per-instance targeting

Expected Dolphin folder structure:
```
Dolphin-x64\
├── Dolphin.exe
├── portable.txt
├── launch_dolphin_instances.ps1   #  MUST be here
├── Dolphin_Multi_Instance.bat     #  MUST be here (manual launcher)
├── User\           # Instance 0 (base)
├── User1\          # Instance 1 (auto-created)
├── User2\          # Instance 2 (auto-created)
└── ...
```

> **⚠️ Important:** `launch_dolphin_instances.ps1` and `Dolphin_Multi_Instance.bat` **must be placed inside the Dolphin directory** (next to `Dolphin.exe`). The script auto-detects Dolphin's path from its own location. If these files are elsewhere, pass `--dolphin-path` explicitly (for the first time only).

The ROM file (`.rvz`, `.iso`, `.wbfs`) is auto-detected by searching `Jeux/`, `Games/`, or `ROMs/` folders **next to** the Dolphin directory. Example layout:
```
MySetup\
├── Dolphin-x64\          # Contains Dolphin.exe + scripts
│   ├── Dolphin.exe
│   ├── launch_dolphin_instances.ps1
│   ├── Dolphin_Multi_Instance.bat
│   └── User\
└── Games\                  # ROM folder (sibling of Dolphin dir)
    └── MHtri\
        └── MonsterHunterTri.rvz
```

To override auto-detection, pass `--rom-path`:
```bash
python train.py --rom-path "D:\Games\MonsterHunterTri.rvz" --num-instances 6
```

---

## Usage

### Quick Start

```bash
# 1. Verify the entire setup
python check_setup.py

# 2. Basic single-instance training
python train.py --timesteps 100000 --name my_first_run

# 3. Resume from checkpoint
python train.py --resume ./models/my_first_run/checkpoint_50000_steps.zip --timesteps 200000
```

### Complete argument reference

All arguments with their defaults and usage:

#### Training
| Argument                   | Default  | Description                                                                                                                                                             |
|----------------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--timesteps N`            | `100000` | Total training timesteps                                                                                                                                                |
| `--name NAME`              | auto     | Experiment name (used for `logs/` and `models/` subfolders)                                                                                                             |
| `--resume PATH`            | —        | Resume from a `.zip` checkpoint. ⚠️ If you also change `--env`, `--grayscale`, or any observation-space setting, add `--force-new-vecnormalize` to avoid shape mismatch |
| `--force-new-vecnormalize` | off      | Discard the saved VecNormalize stats and start fresh (required when observation space changes)                                                                          |
| `--save-state N`           | `5`      | Dolphin save state slot to auto-reload (1–8, maps to F1–F8)                                                                                                             |
| `--lr LR`                  | `3e-4`   | PPO learning rate                                                                                                                                                       |
| `--cpu`                    | off      | Force CPU only (ignore CUDA)                                                                                                                                            |

#### Environment
| Argument      | Default  | Description                                                                    |
|---------------|----------|--------------------------------------------------------------------------------|
| `--env MODE`  | `hybrid` | `hybrid` (vision + memory), `visual` (CNN only), `memory` (vector only)        |
| `--grayscale` | off      | Convert frames to grayscale (1 channel instead of 3, less VRAM)                |
| `--rtvision`  | off      | Show real-time AI vision in an OpenCV window (use GPU ressources)              |
| `--rtminimap` | off      | Show real-time exploration minimap (requires `--rtvision`, use GPU ressources) |

#### Action heads
| Argument                   | Default   |  Description                                                                                                                                                                                                                                                               |
|----------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--disabled-heads HEAD...` | `menu`    | Space-separated list of heads to disable. Pass with no argument to enable all heads. Valid values: `movement`, `camera`, `combat`, `use_item`, `select_item`, `menu`, `sprint`. **The `menu` head is disabled by default** to prevent the agent from getting stuck in menus. |
| `--no-transformer-heads`   | off       | Disable Transformer cross-attention between the 7 action heads. Reverts to standard independent Linear layers (SB3 default). Use this for smaller experiments or to match old checkpoints trained without Transformer. |

Examples:
```bash
# Default: menu disabled, Transformer heads active
python train.py

# Disable menu AND use_item
python train.py --disabled-heads menu use_item

# Enable ALL heads (including menu)
python train.py --disabled-heads

# Use standard Linear heads instead of Transformer
python train.py --no-transformer-heads
```

#### Dolphin
| Argument                    | Default | Description                                                                                                                                                      |
|-----------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dolphin-path PATH`       | auto    | Path to `Dolphin.exe` or its parent folder. Auto-detected from common locations, then saved to `config/dolphin_path_config.json` for future runs.                |
| `--rom-path PATH`           | auto    | Path to the Monster Hunter Tri ROM (`.rvz`, `.iso`, `.wbfs`). Auto-detected from `Jeux/`, `Games/`, or `ROMs/` folders next to the Dolphin directory.            |
| `--dolphin-timeout SECONDS` | `60`    | Timeout waiting for Dolphin windows to appear after launch                                                                                                       |

#### Multi-agent / Multi-instance
| Argument                    | Default                   | Description                                                                                                                                                                                                    |
|-----------------------------|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--num-agents N`            | same as `--num-instances` | Number of independent PPO agents (1–10, or up to 32 with `--nolimit`)                                                                                                                                          |
| `--num-instances N`         | `1`                       | Number of Dolphin instances to launch (1–10, or up to 16 with `--nolimit`)                                                                                                                                     |
| `--nolimit`                 | off                       | **Bypass the 10-agent / 10-instance safety cap.** This cap exists to avoid accidentally freezing your PC. With `--nolimit` you can go up to 32 agents / 16 instances. A confirmation prompt will still appear. |
| `--multi-agent-mode MODE`   | `independent`             | `independent`, `round_robin`, `majority_vote`, `weighted`, `genetic`                                                                                                                                           |
| `--allocation-mode MODE`    | `auto`                    | `auto`, `manual`, `weighted`                                                                                                                                                                                   |
| `--allocation-map MAP`      | —                         | Manual mapping string, e.g. `"0:0,1;1:2,3"`                                                                                                                                                                    |
| `--steps-per-agent N`       | `4096`                    | Rollout steps collected per agent before each PPO update                                                                                                                                                       |
| `--block-size N`            | `100`                     | Steps per agent per block in `round_robin` mode                                                                                                                                                                |
| `--weighted-eval-freq N`    | `100`                     | Episodes between allocation re-evaluations in `weighted` mode                                                                                                                                                  |
| `--genetic-generations N`   | `10`                      | Number of generations (genetic mode)                                                                                                                                                                           |
| `--genetic-elite-ratio R`   | `0.25`                    | Fraction of agents kept as elites (genetic mode)                                                                                                                                                               |
| `--genetic-mutation-rate R` | `0.3`                     | Mutation rate (genetic mode)                                                                                                                                                                                   |

#### Interface
| Argument            | Default   | Description                                                 |
|---------------------|-----------|-------------------------------------------------------------|
| `--no-gui`          | off       | Disable the training GUI (wich can save you GPU ressources) |
| `--log-level LEVEL` | `WARNING` | Console log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`  |

#### Debug
| Argument          | Default | Description                                                 |
|-------------------|---------|-------------------------------------------------------------|
| `--debug-steps N` | —       | Override `--timesteps` with a small value for quick testing |
| `--small-rollout` | off     | Use `n_steps=512` (short rollouts, useful for debugging)    |

---

> **Resuming with changed observation space:** If you resume a checkpoint with `--resume` but change how the agent perceives the game (e.g. switching `--env hybrid` to `--env memory`, adding or removing `--grayscale`), the saved VecNormalize statistics will no longer match the new observation shape and training will crash. Add `--force-new-vecnormalize` to discard the old stats:
> ```bash
> python train.py --resume ./models/my_run/checkpoint.zip --env memory --force-new-vecnormalize
> ```

### Logs and Models

```
logs/<experiment>/<timestamp>/
├── agent_0/
│   ├── env_0/
│   │   ├── console.log         # All module logs (DEBUG+)
│   │   ├── reward_debug.log    # Per-step reward breakdown
│   │   └── training_data.jsonl # Step/episode data (JSONL)
│   ├── errors.log              # ERROR+ with full tracebacks
│   └── session_summary.json    # Run statistics

models/<experiment>/
├── checkpoint_NNNNN_steps.zip  # Periodic checkpoints (every ~10%)
└── final_model.zip             # End-of-training model
```

TensorBoard:
```bash
tensorboard --logdir ./logs/
```

---

## Troubleshooting

### Dolphin Not Detected
```
RuntimeError: No Dolphin window found for 'MHTri-0'
```
- Make sure Dolphin is running and Monster Hunter Tri is loaded
- **Enter a quest** (not the village — the agent needs to be in-game)
- If launching manually (not via `train.py`), check that the window title contains `MHTri` or `Monster Hunter`
- Re-run the script

### Memory Values Look Strange
Example: `player_hp: 3.456789e37` — this is **normal** for some addresses.  
The agent learns from **relative changes** (delta HP), not absolute values. VecNormalize handles the rest.

### DLL Not Injected
```
TimeoutError: Shared memory 'DolphinInputHook_SharedMem_XXXX' not available
```
- Check `vision/dolphin_hook_debug.txt` (written next to `Dolphin.exe`) for DLL-side error messages
- The pre-compiled DLL is in `vision/dolphin_input_hook.dll` — verify it wasn't accidentally deleted
- To rebuild: `cd hook && cargo build --release` (requires cargo: https://rustup.rs)
- Verify cargo is in PATH: `cargo --version`

### Black / Empty Frames
- Make sure Dolphin is not minimized (or build `DolphinCapture.dll` for focus-free capture)
- Graphics backend must be `Direct3D 11` or `Vulkan` — not OpenGL
- Check `vision/debug/crop_verification_training.png` after the first reset

### CUDA Out of Memory
```bash
python train.py --grayscale              # 1 channel instead of 3 per frame
python train.py --cpu                    # Force CPU (slower)
python train.py --small-rollout          # n_steps=512, batch_size=64
```

### Multi-Instance: Wrong Window Captured
```
WARNING: Window mismatch: expected 'MHTri-1', got 'MHTri-0'
```
- Windows may still be loading — the launcher retries automatically
- Check that `launch_dolphin_instances.ps1` completed without errors
- Increase `--dolphin-timeout` (default: 60s): `--dolphin-timeout 90`

### Save State Not Reloading
- The save state **must be saved INSIDE an active quest** (not in the village, not on the reward screen)
- Verify slot: `--save-state 5` (F5 key)
- Check that the quest timer is > 10 seconds in your save

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -am 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

### Areas for Contribution
- **Translation** — convert French code comments to English
- **Memory addresses** — find missing addresses (weapon/armor ID)
- **Reward tuning** — improve the reward shaping for faster learning
- **New CNN architectures** — try EfficientNet, ResNet, etc.
- **Tests** — add unit tests in `pytest`
- **Documentation** — tutorials, video guides

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Capcom** for Monster Hunter Tri
- **Dolphin Emulator Team** for the excellent emulator and open architecture
- **Stable-Baselines3** for clean PPO implementations
- **Microsoft Detours / Rust `windows` crate** for DLL injection foundations
- **OpenAI / DeepMind** for PPO and RL research
- The entire **RL community**

---

---

# Version Française

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités Principales](#fonctionnalités-principales)
- [Structure du Projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation-1)
  - [Dépendances Python](#dépendances-python)
  - [Configuration Dolphin](#configuration-dolphin)
  - [Compiler les DLLs](#compiler-les-dlls)
  - [Configuration Multi-Instance](#configuration-multi-instance)
- [Utilisation](#utilisation)
- [Problèmes](#Problèmes-courants)
- [Contribuer](#contribuer)
- [Licence](#licence)

---

## Vue d'ensemble

Ce projet entraîne une IA par **Apprentissage par Renforcement Profond** à jouer à **Monster Hunter Tri** en utilisant **PPO (Proximal Policy Optimization)**. L'IA apprend en combinant trois modalités :

- **Vision (CNN)** : Traitement des frames du jeu (84×84 pixels, 4 frames empilées)
- **Lecture Mémoire** : Accès direct à la RAM via Dolphin Memory Engine (vecteur 70 features)
- **Injection DLL** : Envoi des inputs sans focus via une DLL Rust hookée dans la couche DInput de Dolphin

> **⚠️ Note sur les Commentaires** : Les commentaires de code sont encore parfois en français. La traduction anglaise est en cours.

---

## Fonctionnalités Principales

### Architecture d'Observation Hybride
| Modalité          | Forme          | Description                                                                                |
|-------------------|----------------|--------------------------------------------------------------------------------------------|
| Visuelle          | `(84, 84, 12)` | 4 frames RGB empilées, préprocessées sur GPU                                               |
| Mémoire           | `(70,)`        | HP, stamina, position, orientation, zone, inventaire (24 slots), monstres, tranchant, etc. |
| Carte exploration | `(15, 15, 4)`  | Minimap locale avec 4 canaux : visites, position joueur, cubes récents, marqueurs          |

> **Coordination des têtes :** Les 7 têtes utilisent un **Transformer à self-attention** (2 couches, 4 têtes d'attention) pour coordonner leurs décisions avant de produire les actions. Voir [Têtes d'action](#têtes-daction) pour les détails.

### Espace d'Action Multi-Têtes
L'agent contrôle **7 têtes d'action** simultanément via `MultiDiscrete([5, 5, 5, 2, 3, 8, 2])` :

| Tête              | Actions | Description                                                    |
|-------------------|---------|----------------------------------------------------------------|
| Mouvement         | 5       | rien / avant / arrière / gauche / droite                       |
| Caméra            | 5       | rien / haut / bas / gauche / droite                            |
| Combat            | 5       | rien / attaque / esquive / dégainer / coup de pied             |
| Utiliser Item     | 2       | rien / utiliser                                                |
| Sélectionner Item | 3       | rien / radial gauche / radial droite                           |
| Menu              | 8       | rien / start / nav haut/bas/gauche/droite / confirmer / retour |
| Sprint            | 2       | rien / sprint                                                  |

Les touches sont **maintenues entre les steps** (modèle hold/release) — pas de simples tapotements.

#### Cross-Attention Transformer (par défaut)

Par défaut, les 7 têtes ne sont **pas indépendantes** : elles coordonnent leurs décisions via un module **Transformer à self-attention** (`TransformerActionHead`). Au lieu de 7 couches linéaires lisant le même vecteur latent, chaque tête reçoit son propre token avec une identité positionnelle, puis les 7 tokens passent par 2 couches de multi-head self-attention avant de produire les logits.

Cela permet aux têtes d'apprendre la coordination inter-têtes — ex. « ne pas sprinter en attaquant », « ne pas ouvrir le menu en esquivant » — **de manière proactive dans le réseau**, plutôt que de dépendre uniquement du masquage de compatibilité post-hoc.

| Propriété          | Valeur                                |
|--------------------|---------------------------------------|
| Architecture       | Self-attention pre-norm, 2 couches    |
| Dimension embedding | 48 par token de tête                  |
| Têtes d'attention  | 4                                     |
| Paramètres         | ~130K (vs ~8K pour linéaire standard) |
| Surcoût            | ~0.1ms par forward pass (négligeable) |

Pour désactiver le Transformer et revenir aux couches linéaires indépendantes, utiliser `--no-transformer-heads`.

### Injection DLL sans Focus
Les inputs sont injectés directement dans `IDirectInputDevice8::GetDeviceState` de Dolphin via une DLL Rust (`dolphin_input_hook.dll`). Aucun ViGEmBus, aucun HidHide, aucun focus de fenêtre requis. Plusieurs instances peuvent tourner entièrement minimisées.

**Isolation totale des inputs par instance :** Chaque instance Dolphin possède sa propre DLL injectée avec son propre canal de mémoire partagée. Les inputs des agents (clavier, boutons souris) sont écrits directement dans le buffer DInput de chaque instance — ils **n'atteignent jamais la pile clavier/souris réelle de Windows**. Inversement, tes propres inputs clavier et souris sont **bloqués pour les instances Dolphin**, ce qui te permet de travailler normalement sur ton PC pendant que les agents s'entraînent. C'était la limitation principale des versions alpha.

**Audio désactivé automatiquement :** Lors du lancement de plusieurs instances via le script PowerShell, le `Dolphin.ini` de chaque instance est patché avec `[DSP] Backend = No audio` afin qu'aucune instance n'émette de son pendant l'entraînement.

### Support Multi-Agents
- Jusqu'à **32 agents PPO** entraînés simultanément
- Jusqu'à **16 instances Dolphin**
- Modes d'allocation flexibles : un-pour-un, instances multiples par agent, partage d'instances
- **Synchronisation FedAvg** des poids entre agents tous les N cycles de mise à jour

### Modes d'Entraînement
| Mode            | Description                                                         |
|-----------------|---------------------------------------------------------------------|
| `independent`   | Chaque agent agit indépendamment sur son instance assignée          |
| `round_robin`   | Les agents alternent par blocs de N steps                           |
| `majority_vote` | Tous les agents prédisent ; l'action la plus fréquente est exécutée |
| `weighted`      | L'allocation s'adapte selon les rewards des épisodes par agent      |
| `genetic`       | Sélection évolutionnaire avec mutation et croisement                |

### Système de Récompenses
Récompense multi-catégories avec trackers séparés :
- Survie (par step), coups sur monstre, exploration (découverte de cubes), changements de zone
- Dégâts reçus, pénalité de mort, pénalité d'immobilité, pénalité de menu
- Timer de camp, présence en zone monstre, oxygène (sous l'eau), tranchant

---

## Structure du Projet

L'arborescence complète annotée est disponible dans **`structure-fr.txt`** (français) et **`structure-en.txt`** (anglais) à la racine du dépôt. Chaque fichier et dossier y est décrit.

Les deux répertoires ci-dessous sont **absents du dépôt** - ils sont créés automatiquement dès qu'un entraînement se termine (ou est interrompu) :

- `logs/<expérience>/<horodatage>/` — logs par agent/env, breakdowns reward, résumés de session
- `models/<expérience>/<horodatage>/` — checkpoints, modèle final, sauvegardes en cas d'interruption

## Prérequis

### Logiciels
| Logiciel           | Version       | Notes                                                                         |
|--------------------|---------------|-------------------------------------------------------------------------------|
| Python             | 3.8+          | Testé avec 3.10                                                               |
| Dolphin Emulator   | 2509+         | Lancement standard, pas besoin de droits admin                                |
| Monster Hunter Tri | NTSC-U ou PAL | ISO / WBFS / RVZ                                                              |
| Rust / cargo       | dernière      | Seulement si `dolphin_input_hook.dll` est manquant (DLL pré-compilée incluse) |
| Visual Studio C++  | 2019 ou 2022  | Seulement si `DolphinCapture.dll` est manquante (DLL pré-compilée incluse)    |

### Matériel
| Composant   | Minimum   | Recommandé                |
|-------------|-----------|---------------------------|
| CPU         | 4 cœurs   | 8+ cœurs (multi-instance) |
| RAM         | 8 Go      | 16 Go+                    |
| GPU         | —         | NVIDIA GTX 1060+ (CUDA)   |
| Stockage    | 5 Go      | 15 Go+ (logs + modèles)   |

---

## Installation

### Dépendances Python

```bash
pip install -r requirements.txt
```

Pour GPU (CUDA 12.4) :
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Compiler les DLLs

Les deux DLLs sont **déjà pré-compilées** et incluses dans le dépôt (dossier `vision/`). Tu n'as **pas besoin** de Visual Studio ni de Rust sauf si les fichiers sont manquants ou tu veux explicitement recompiler depuis les sources.

**1. `dolphin_input_hook.dll`** (Rust — injection d'inputs)
```bash
# Nécessite : cargo (https://rustup.rs)
cd hook
cargo build --release
copy target\release\dolphin_input_hook.dll ..\vision\
```
> Alternative : lancer `train.py` directement — il auto-compile via `dll_utils.py` si cargo est dans le PATH.

**2. `DolphinCapture.dll`** (C++ — capture d'écran)
```bash
# Nécessite : Visual Studio 2019/2022 avec "Développement Desktop en C++"
cd _build
python build_dll.py
# Sortie : DolphinCapture.dll → copier dans vision/
```
> Sans cette DLL, la capture de frames utilise le fallback GDI (la fenêtre Dolphin doit rester visible).

### Configuration Dolphin

**Étape 1 : Configuration de base**
1. Télécharger Dolphin 2509+ depuis [dolphin-emu.org](https://dolphin-emu.org)
2. Créer `portable.txt` dans le dossier racine Dolphin (active les profils User séparés pour le multi-instance)
3. Lancer Dolphin une fois pour générer le dossier `User/`, puis fermer

**Étape 2 : Paramètres graphiques**
- Config → Interface : **décocher** "Confirm on Stop"
- Config → Interface : **décocher** "Pause on Focus Lost"
- GFX.ini → `[General]` → `RenderToMain = False` (configuré automatiquement par le launcher)

**Étape 3 : Configuration du contrôleur (mapping Wiimote → Clavier)**

Aller dans **Manette → Wiimote 1 → Configurer → Périphérique : DInput/0/Keyboard Mouse** et vérifier que les mappings suivants sont configurés :

| Entrée Wiimote / Nunchuk       | Touche associé  |
|--------------------------------|-----------------|
| Stick Nunchuk ↑                | `Z`             |
| Stick Nunchuk ↓                | `S`             |
| Stick Nunchuk ←                | `Q`             |
| Stick Nunchuk →                | `D`             |
| D-pad ↑ (caméra)               | `Flèche haut `  |
| D-pad ↓ (caméra)               | `Flèche bas`    |
| D-pad ← (caméra)               | `Flèche gauche` |
| D-pad → (caméra)               | `Flèche droite` |
| Bouton A (attaque)             | `Clique gauche` |
| Bouton B (esquive)             | `Clique droit`  |
| Bouton 2                       | `E`             |
| Bouton 1                       | `A`             |
| Bouton +                       | `P`             |
| Bouton -                       | `M`             |
| Nunchuk C (dégainer/rengainer) | `Shift`         |
| Nunchuk Z (courir/bloquer)     | `Ctrl gauche`   |

> La DLL injecte ces inputs clavier/souris directement dans le buffer DInput de chaque instance Dolphin. Tes propres inputs clavier et souris sont bloqués pour les instances d'entraînement.

>NOTE : Sur un clavier AZERTY, Dolphin interprète les touches comme si le clavier était en QWERTY, donc les lettres affichées ne correspondent pas forcément aux touches physiques. Tant que la touche s’illumine en rouge dans la configuration lorsque vous appuyez dessus, le mapping est correct.

**Étape 4 : Chemin Dolphin**
- Au premier lancement de `train.py`, le chemin vers le dossier Dolphin sera demandé.
- Le chemin est sauvegardé dans `config/dolphin_path_config.json` pour les prochains runs.
- Ou passe-le directement : `--dolphin-path "C:\Dolphin-x64"`

**Étape 5 : Chemin ROM (automatique)**
- La ROM (`.rvz`, `.iso`, `.wbfs`) est auto-détectée depuis les dossiers `Jeux/`, `Games/` ou `ROMs/` à côté du répertoire Dolphin.
- Le chemin est sauvegardé dans `config/rom_path_config.json` pour les prochains runs.
- Ou passe-le directement : `--rom-path "D:\Games\MonsterHunterTri.rvz"`

**Étape 6 : Calibration du crop HUD (optionnel)**
```bash
python vision/hud_crop_tuner.py
# Outil OpenCV interactif — ajuster le crop pour supprimer la barre de vie/minimap
# Sauvegarde dans config/crop_config.json
```

### Configuration Multi-Instance

La **préparation des dossiers User** est gérée automatiquement par `launch_dolphin_instances.ps1`.  
Quand `train.py` lance plusieurs instances, il appelle le script PowerShell qui :
1. Détecte si les dossiers `User1/`, `User2/`, etc. existent
2. **Crée automatiquement les manquants** en copiant le dossier `User/` de base
3. Active `BackgroundInput = True` dans le `Dolphin.ini` de chaque instance
4. Désactive l'audio (`[DSP] Backend = No audio`) par instance
5. Renomme les fenêtres de rendu en `MHTri-0`, `MHTri-1`, etc.

Structure attendue du dossier Dolphin :
```
Dolphin-x64\
├── Dolphin.exe
├── portable.txt
├── launch_dolphin_instances.ps1   #  OBLIGATOIREMENT ici
├── Dolphin_Multi_Instance.bat     #  OBLIGATOIREMENT ici (lanceur manuel de multiple instances)
├── User\           # Instance 0 (base)
├── User1\          # Instance 1 (auto-créé)
├── User2\          # Instance 2 (auto-créé)
└── ...
```

> **⚠️ Important :** Les fichiers `launch_dolphin_instances.ps1` et `Dolphin_Multi_Instance.bat` **doivent être placés dans le répertoire Dolphin** (à côté de `Dolphin.exe`). Le script auto-détecte le chemin de Dolphin depuis sa propre position. Si ces fichiers sont ailleurs, passe `--dolphin-path` explicitement.

La ROM (`.rvz`, `.iso`, `.wbfs`) est auto-détectée en cherchant dans les dossiers `Jeux/`, `Games/` ou `ROMs/` **à côté** du répertoire Dolphin. Exemple de structure :
```
MonSetup\
├── Dolphin-x64\              # Contient Dolphin.exe + scripts
│   ├── Dolphin.exe
│   ├── launch_dolphin_instances.ps1
│   ├── Dolphin_Multi_Instance.bat
│   └── User\
└── Jeux\                      # Dossier ROM (à côté du dossier Dolphin)
    └── MHtri\
        └── MonsterHunterTri.rvz
```

Pour forcer un chemin ROM spécifique :
```bash
python train.py --rom-path "D:\Jeux\MonsterHunterTri.rvz" --num-instances 6
```

---

## Utilisation

### Démarrage Rapide

```bash
# 1. Vérifier toute la configuration
python check_setup.py

# 2. Entraînement basique (instance unique)
python train.py --timesteps 100000 --name mon_premier_run

# 3. Reprendre depuis un checkpoint
python train.py --resume ./models/mon_premier_run/checkpoint_50000_steps.zip --timesteps 200000
```

### Référence complète des arguments

#### Entraînement
| Argument                   | Défaut   |  Description                                                                                                                                                                                     |
|----------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--timesteps N`            | `100000` | Steps d'entraînement totaux                                                                                                                                                                      |
| `--name NAME`              | auto     | Nom de l'expérience (utilisé pour les dossiers `logs/` et `models/`)                                                                                                                             |
| `--resume PATH`            | —        | Reprendre depuis un checkpoint `.zip`. ⚠️ Si tu changes aussi `--env`, `--grayscale` ou tout paramètre d'espace d'observation, ajoute `--force-new-vecnormalize` pour éviter une erreur de shape |
| `--force-new-vecnormalize` | off      | Ignorer les stats VecNormalize sauvegardées et repartir de zéro (requis si l'espace d'observation change)                                                                                        |
| `--save-state N`           | `5`      | Slot save state Dolphin à recharger automatiquement (1–8, correspond à F1–F8)                                                                                                                    |
| `--lr LR`                  | `3e-4`   | Taux d'apprentissage PPO                                                                                                                                                                         |
| `--cpu`                    | off      | Forcer le CPU uniquement (ignorer CUDA)                                                                                                                                                          |

#### Environnement
| Argument      | Défaut   | Description                                                               |
|---------------|----------|---------------------------------------------------------------------------|
| `--env MODE`  | `hybrid` | `hybrid` (vision + mémoire), `visual` (CNN seul), `memory` (vecteur seul) |
| `--grayscale` | off      | Frames en niveaux de gris (1 canal au lieu de 3, moins de VRAM)           |
| `--rtvision`  | off      | Afficher la vision IA en temps réel dans une fenêtre OpenCV               |
| `--rtminimap` | off      | Afficher la minimap d'exploration en temps réel (nécessite `--rtvision`)  |

#### Têtes d'action
| Argument                   | Défaut   | Description                                                                                                                                                                                                                                                                                      |
|----------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--disabled-heads HEAD...` | `menu`   | Liste de têtes à désactiver (séparées par des espaces). Passer sans argument pour tout activer. Valeurs valides : `movement`, `camera`, `combat`, `use_item`, `select_item`, `menu`, `sprint`. **La tête `menu` est désactivée par défaut** pour éviter que l'agent reste bloqué dans les menus. |
| `--no-transformer-heads`   | off      | Désactiver la cross-attention Transformer entre les 7 têtes d'action. Revient aux têtes indépendantes (défaut SB3). Utile pour les petites expériences ou pour reprendre d'anciens checkpoints entraînés sans Transformer.                                                                       |

Exemples :
```bash
# Défaut : menu désactivé, Transformer actif
python train.py

# Désactiver menu ET use_item
python train.py --disabled-heads menu use_item

# Activer TOUTES les têtes (menu inclus)
python train.py --disabled-heads

# Utiliser des têtes linéaires standard au lieu du Transformer
python train.py --no-transformer-heads
```

#### Dolphin
| Argument                    | Défaut | Description                                                                                                                                                           |
|-----------------------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dolphin-path PATH`       | auto   | Chemin vers `Dolphin.exe` ou son dossier parent. Auto-détecté depuis les emplacements courants, puis sauvegardé dans `config/dolphin_path_config.json`.               |
| `--rom-path PATH`           | auto   | Chemin vers la ROM Monster Hunter Tri (`.rvz`, `.iso`, `.wbfs`). Auto-détecté depuis les dossiers `Jeux/`, `Games/` ou `ROMs/` à côté du répertoire Dolphin.          |
| `--dolphin-timeout SECONDS` | `60`   | Timeout pour la détection des fenêtres Dolphin après le lancement                                                                                                     |

#### Multi-agent / Multi-instance
| Argument                    | Défaut                 | Description                                                                                                                                                                                                                                             |
|-----------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--num-agents N`            | idem `--num-instances` | Nombre d'agents PPO indépendants (1–10, ou jusqu'à 32 avec `--nolimit`)                                                                                                                                                                                 |
| `--num-instances N`         | `1`                    | Nombre d'instances Dolphin à lancer (1–10, ou jusqu'à 16 avec `--nolimit`)                                                                                                                                                                              |
| `--nolimit`                 | off                    | **Désactiver la limite de sécurité de 10 agents / 10 instances.** Cette limite existe pour éviter de bloquer accidentellement le PC. Avec `--nolimit`, tu peux aller jusqu'à 32 agents / 16 instances. Une invite de confirmation s'affiche quand même. |
| `--multi-agent-mode MODE`   | `independent`          | `independent`, `round_robin`, `majority_vote`, `weighted`, `genetic`                                                                                                                                                                                    |
| `--allocation-mode MODE`    | `auto`                 | `auto`, `manual`, `weighted`                                                                                                                                                                                                                            |
| `--allocation-map MAP`      | —                      | Mapping manuel, ex. `"0:0,1;1:2,3"`                                                                                                                                                                                                                     |
| `--steps-per-agent N`       | `4096`                 | Steps de rollout collectés par agent avant chaque update PPO                                                                                                                                                                                            |
| `--block-size N`            | `100`                  | Steps par agent par bloc en mode `round_robin`                                                                                                                                                                                                          |
| `--weighted-eval-freq N`    | `100`                  | Épisodes entre deux réévaluations de l'allocation en mode `weighted`                                                                                                                                                                                    |
| `--genetic-generations N`   | `10`                   | Nombre de générations (mode génétique)                                                                                                                                                                                                                  |
| `--genetic-elite-ratio R`   | `0.25`                 | Fraction d'agents conservés comme élites (mode génétique)                                                                                                                                                                                               |
| `--genetic-mutation-rate R` | `0.3`                  | Taux de mutation (mode génétique)                                                                                                                                                                                                                       |

#### Interface
| Argument            | Défaut    | Description                                             |
|---------------------|-----------|---------------------------------------------------------|
| `--no-gui`          | off       | Désactiver l'interface graphique                        |
| `--log-level LEVEL` | `WARNING` | Verbosité console : `DEBUG`, `INFO`, `WARNING`, `ERROR` |

#### Debug
| Argument          | Défaut   | Description                                                          |
|-------------------|----------|----------------------------------------------------------------------|
| `--debug-steps N` | —        | Remplacer `--timesteps` par une petite valeur pour tester rapidement |
| `--small-rollout` | off      | Utiliser `n_steps=512` (rollouts courts, pour le debug)              |

---

> **Reprendre avec un espace d'observation modifié :** Si tu reprends un checkpoint avec `--resume` mais que tu changes la façon dont l'agent perçoit le jeu (ex. passer de `--env hybrid` à `--env memory`, ajouter ou retirer `--grayscale`), les statistiques VecNormalize sauvegardées ne correspondent plus à la nouvelle shape et l'entraînement plantera. Ajoute `--force-new-vecnormalize` pour ignorer les anciennes stats :
> ```bash
> python train.py --resume ./models/mon_run/checkpoint.zip --env memory --force-new-vecnormalize
> ```

### Logs et Modèles

```
logs/<expérience>/<horodatage>/
├── agent_0/
│   ├── env_0/
│   │   ├── console.log         # Tous les logs modules (DEBUG+)
│   │   ├── reward_debug.log    # Breakdown reward par step
│   │   └── training_data.jsonl # Données step/épisode (JSONL)
│   ├── errors.log              # ERROR+ avec tracebacks complets
│   └── session_summary.json    # Statistiques du run

models/<expérience>/<horodatage>/
├── checkpoint_NNNNN_steps.zip  # Checkpoints périodiques (~tous les 10%)
└── final_model.zip             # Modèle final
```

TensorBoard :
```bash
tensorboard --logdir ./logs/
```

---

## Problèmes courants

### Dolphin Non Détecté
```
RuntimeError: No Dolphin window found for 'MHTri-0'
```
- Vérifier que Dolphin est bien lancé avec Monster Hunter Tri chargé
- **Lancer une quête** (pas dans le village — l'agent doit être en jeu)
- Si tu as lancé Dolphin manuellement (pas via `train.py`), vérifier que le titre de la fenêtre contient `MHTri` ou `Monster Hunter`
- Relancer le script

### Valeurs Mémoire Bizarres
Exemple : `player_hp: 3.456789e37` — c'est **normal** pour certaines adresses.  
L'agent apprend des **variations relatives** (delta HP), pas des valeurs absolues. VecNormalize gère la normalisation.

### DLL non injectée
```
TimeoutError: Mémoire partagée 'DolphinInputHook_SharedMem_XXXX' non disponible
```
* Vérifiez `vision/dolphin_hook_debug.txt` (écrit à côté de `Dolphin.exe`) pour les messages d’erreur côté DLL
* La DLL précompilée se trouve dans `vision/dolphin_input_hook.dll` — vérifiez qu’elle n’a pas été supprimée accidentellement
* Pour recompiler : `cd hook && cargo build --release` (nécessite cargo : [https://rustup.rs](https://rustup.rs))
* Vérifiez que cargo est dans le PATH : `cargo --version`


### Frames Noires / Vides
- S'assurer que Dolphin n'est pas minimisé (ou compiler `DolphinCapture.dll` pour capture sans focus)
- Backend graphique : `Direct3D 11` ou `Vulkan` — pas OpenGL
- Vérifier `vision/debug/crop_verification_training.png` après le premier reset

### CUDA Hors Mémoire
```bash
python train.py --grayscale              # 1 canal au lieu de 3 par frame
python train.py --cpu                    # Forcer CPU (plus lent)
python train.py --small-rollout          # n_steps=512, batch_size=64
```

### Multi-Instance : Mauvaise Fenêtre Capturée
```
WARNING: Window mismatch: expected 'MHTri-1', got 'MHTri-0'
```
- Les fenêtres sont peut-être encore en train de charger — le launcher réessaie automatiquement
- Vérifier que `launch_dolphin_instances.ps1` s'est terminé sans erreur
- Augmenter `--dolphin-timeout` (défaut : 60s) : `--dolphin-timeout 90`

### Save State Non Rechargée
- La save state **doit être sauvegardée DANS une quête active** (pas dans le village, pas sur l'écran de récompense)
- Vérifier le slot : `--save-state 5` (touche F5)
- S'assurer que le timer de quête est > 10 secondes dans la sauvegarde

---

## Contribuer

Les contributions sont bienvenues !

1. Forker le dépôt
2. Créer une branche : `git checkout -b feature/ma-feature`
3. Committer : `git commit -am 'Ajouter ma feature'`
4. Pousser : `git push origin feature/ma-feature`
5. Ouvrir une Pull Request

### Domaines de Contribution
- **Traduction** — convertir les commentaires français en anglais
- **Adresses mémoire** — trouver les adresses manquantes (ID arme, armures)
- **Réglage des récompenses** — améliorer la reward shaping pour un apprentissage plus rapide
- **Nouvelles architectures CNN** — EfficientNet, ResNet, etc.
- **Tests** — ajouter des tests unitaires pytest
- **Documentation** — tutoriels, guides vidéo

---

## Licence

Licence MIT — voir [LICENSE](LICENSE) pour les détails.

---

## Remerciements

- **Capcom** pour Monster Hunter Tri
- **L'équipe Dolphin Emulator** pour l'excellent émulateur et son architecture ouverte
- **Stable-Baselines3** pour les implémentations PPO propres
- **Microsoft Detours / crate Rust `windows`** pour les bases d'injection DLL
- **OpenAI / DeepMind** pour la recherche PPO et RL
- Toute la **communauté RL**