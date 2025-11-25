# Monster-Hunter-Tri-AI

*Un projet d'IA utilisant l'apprentissage par renforcement pour maîtriser Monster Hunter Tri sur émulateur Dolphin.*

*A Reinforcement Learning AI project to master Monster Hunter Tri on Dolphin emulator.*

---

## 🌍 Languages / Langues

- [English Version](#english-version) 🇬🇧
- [Version Française (bientôt)](#version-française) 🇫🇷

---

# English Version

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Python Dependencies](#python-dependencies)
  - [Dolphin Memory Engine](#dolphin-memory-engine)
  - [ViGEmBus Installation](#vigembus-installation)
  - [Dolphin Configuration](#dolphin-configuration)
  - [Multi-Instance Setup](#multi-instance-setup)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Training Options](#training-options)
  - [Testing Trained Agent](#testing-trained-agent)
- [Training Results](#training-results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project trains a Deep Reinforcement Learning agent to play **Monster Hunter Tri** using **PPO (Proximal Policy Optimization)**. The AI learns by combining:

- **Vision (CNN)**: Processes game frames (84x84 pixels)
- **Memory Reading**: Direct RAM access via Dolphin Memory Engine
- **Virtual Controller**: Sends inputs using vgamepad (ViGEmBus)

**⚠️ Note on Comments**: All code comments are currently in **French**, but I plan to translate them to English progressively. Feel free to ask questions or contribute translations!

---

## Key Features

### Hybrid Architecture
- **Vision System**: Convolutional Neural Network for visual processing
- **Memory Vector**: Direct game state reading (70 features including HP, stamina, position, inventory)
- **Exploration Map**: 15x15x4 dynamic map with markers (zone transitions, monsters, water)
- **Advanced Rewards**: Multi-category reward system with exploration tracking

### Multi-Agent Support
- **Up to 32 agents** training simultaneously
- **Multiple Dolphin instances** (up to 16)
- **Flexible allocation modes**:
  - One-to-One (1 agent = 1 instance)
  - Multiple instances per agent
  - Instance sharing with round-robin/majority vote/weighted allocation
- **HidHide integration** for virtual controller isolation

### Training Modes
- **Independent**: Each agent trains separately
- **Round-robin**: Agents take turns in blocks
- **Majority vote**: Democratic action selection
- **Weighted**: Adaptive allocation based on performance
- **Genetic**: Evolutionary algorithm with elitism

---

## Project Structure

```
monster_hunter_ai/
│
├── config/
│   └── memory_addresses.py          # Dolphin memory addresses (DME)
│
├── core/
│   ├── dynamic_memory_reader.py     # Dolphin RAM reading (async mode)
│   ├── state_fusion.py              # Vision + memory fusion
│   ├── controller.py                # Virtual controller (vgamepad)
│   └── exploration_map_incremental.py  # Optimized exploration map
│
├── vision/
│   ├── frame_capture.py             # Dolphin window capture
│   ├── preprocessing.py             # Frame preprocessing (crop, resize, normalize)
│   └── feature_extractor.py         # CNN architectures (Nature/IMPALA/Minigrid)
│
├── environment/
│   ├── mh_env.py                    # Main Gymnasium environment
│   ├── reward_calculator.py         # Reward computation
│   ├── exploration_tracker.py       # Exploration cube system
│   └── cube_markers.py              # Zone markers (water, monsters, transitions)
│
├── agent/
│   └── ppo_agent.py                 # PPO agent (Stable-Baselines3)
│
├── utils/
│   ├── multi_agent_scheduler.py     # Multi-agent coordination
│   ├── multi_agent_trainer.py       # Training loop for shared instances
│   ├── genetic_trainer.py           # Genetic algorithm trainer
│   ├── hidhide_manager.py           # Virtual controller isolation
│   ├── training_gui.py              # Real-time training GUI
│   ├── advanced_logging.py          # Structured logging system
│   └── safe_float.py                # Safe float conversion (NaN/Inf protection)
│
├── train.py                         # Training script (single/multi-agent)
├── test.py                          # Testing script
├── check_setup.py                   # Setup diagnostic tool
├── launch_dolphin_instances.ps1     # PowerShell script for multi-instance
└── requirements.txt                 # Python dependencies
```

---

## Requirements

### Software
- **Python 3.8+** (tested with 3.10)
- **Dolphin Emulator** (version 2509 recommended)
- **Dolphin Memory Engine** (1.3.0-preview2 recommended)
- **Monster Hunter Tri** (NTSC-U/PAL ISO/WBFS)
- **ViGEmBus** (for virtual Xbox 360 controller)
- **HidHide** (optional, for multi-instance controller isolation)

### Hardware
- **CPU**: Multi-core recommended (6+ cores for multi-instance)
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: CUDA-compatible GPU recommended (NVIDIA GTX 1060+)
- **Storage**: 10GB+ free space

---

## Installation

### Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**Complete dependency list:**

```
# Memory Reading
pymem>=1.13.1
dolphin-memory-engine>=1.1.0

# Image Processing & Capture
opencv-python>=4.8.0
pillow>=10.0.0
pywin32>=305
mss>=9.0.1

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Reinforcement Learning
stable-baselines3>=2.1.0
gymnasium>=0.29.0

# Virtual Controller
vgamepad>=0.1.0

# Keyboard Simulation
pynput>=1.7.6

# Real-time GUI
matplotlib>=3.7.0

# Utilities
numpy>=1.24.0
tensorboard>=2.13.0
tqdm>=4.65.0

# Logging & Debug
wandb>=0.15.0
pytest>=7.4.0

# Analysis (Optional)
pandas>=2.0.0
seaborn>=0.12.0
```

**For GPU acceleration (Torch CUDA 12.4):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

### Dolphin Memory Engine

Dolphin Memory Engine (DME) is required for reading game memory.

**Installation:**

```bash
pip install dolphin-memory-engine
```

**If installation fails:**

```bash
pip install git+https://github.com/henriquegemignani/py-dolphin-memory-engine.git
```

**Verification:**

```python
import dolphin_memory_engine as dme
print("DME installed successfully!")
```

---

### ViGEmBus Installation

ViGEmBus is required to create virtual Xbox 360 controllers.

**Steps:**

1. **Download**: [ViGEmBus Releases](https://github.com/nefarius/ViGEmBus/releases)
2. **Install**: Run `ViGEmBusSetup_x64.msi` as Administrator
3. **Restart Windows** (required!)
4. **Verify**:
   ```bash
   python core/controller.py
   ```
   You should see "Virtual Xbox 360 Controller created!"

---

### Dolphin Configuration

#### Step 1: Basic Setup

1. **Download Dolphin**: Version 2509+ from [dolphin-emu.org](https://dolphin-emu.org/)

2. **Enable Portable Mode** (important for multi-instance):
   - Navigate to your Dolphin folder
   - Create an empty file named `portable.txt` in the Dolphin root directory
   - This ensures Dolphin stores configuration in `User/` subfolder

3. **First Launch**:
   - Launch Dolphin once to generate the `User/` folder structure

#### Step 2: Controller Configuration

1. **Options → Controller Settings**
2. **Wiimote 1 → Emulated Wiimote**
3. **Configure → Device**: Select **WGInput/0/Xbox 360 Controller for Windows**
   
   **⚠️ Important**: You must run `python core/controller.py` first to create the virtual controller, then restart Wiimote configuration to see it in the device list.

4. **Button Mapping**:
   - **Note**: Exact button mapping will be documented soon. The agent learns from scratch anyway, so any reasonable mapping works.
   - Default Xbox controller mapping is ok

5. **Verify**:
   - Load Monster Hunter Tri
   - Test controller inputs in-game

#### Step 3: Graphics Settings

- **Graphics → General → Backend**: `Direct3D 11` or `Vulkan` (not OpenGL)
- **Graphics → Enhancements**: Keep default or adjust for performance

---

### Multi-Instance Setup

To train multiple agents simultaneously on different Dolphin instances:

#### Step 1: Prepare User Folders

In your Dolphin directory, duplicate the `User` folder for each instance:

```bash
cd "C:\Path\To\Dolphin"

xcopy User User1 /E /I
xcopy User User2 /E /I
xcopy User User3 /E /I
# Repeat for as many instances as needed
```

**Example structure:**
```
C:\Dolphin\
├── Dolphin.exe
├── portable.txt
├── User\          # Instance 0
├── User1\         # Instance 1
├── User2\         # Instance 2
└── User3\         # Instance 3
```

#### Step 2: Setup PowerShell Script

1. **Copy** `launch_dolphin_instances.ps1` from the project to your Dolphin folder:
   
   ```
   Source: monster_hunter_ai/launch_dolphin_instances.ps1
   Target: C:\Users\USER\Desktop\Dolphin-x64\launch_dolphin_instances.ps1
   ```

2. **Edit paths** in the script:
   ```powershell
   $DolphinPath = "C:\Path\To\Dolphin\Dolphin.exe"
   $GameISO = "C:\Path\To\MHTri.rvz"
   ```

3. **Test**:
   ```powershell
   cd C:\Dolphin
   powershell -ExecutionPolicy Bypass -File launch_dolphin_instances.ps1
   ```

#### Step 3: Configure HidHide (Optional)

For proper controller isolation:

1. **Install HidHide**: [Download](https://github.com/ViGEm/HidHide/releases)
2. **Restart Windows** (required!)
3. **Run training script as Administrator**
4. Automatic device isolation will be configured

**Without HidHide**: All instances share controllers (may cause conflicts but still works).

---

## Usage

### Quick Start

**1. Verify Setup:**
```bash
python check_setup.py
```

This checks:
- ✅ Python version & dependencies
- ✅ GPU/CUDA availability
- ✅ Dolphin connection
- ✅ Memory reading
- ✅ Frame capture
- ✅ Virtual controller

**2. Basic Training (single instance):**
```bash
python train.py --timesteps 100000 --name my_first_training
```

**3. Monitor Progress:**
- Real-time GUI shows stats, rewards, exploration map
- Press "Stop" button for clean shutdown
- Check `./logs/` for detailed logs

---

### Training Options

You can type
```bash
python train.py -h
```
or
```bash
python train.py --help
```
to get the full list directly (this doesn't launch training)

#### Single Agent Training

**Basic training:**
```bash
python train.py --timesteps 1000000 --name full_training
```

**Advanced training:**
```bash
python train.py \
  --timesteps 1000000 \
  --name advanced_training \
  --lr 1e-4 \
  --save-state 5 \
  --grayscale \
  --rtvision
```

**Common arguments:**
- `--timesteps N` - Total training steps
- `--name TEXT` - Experiment name (auto-generated if omitted)
- `--lr FLOAT` - Learning rate (default: 0.0001)
- `--save-state N` - Save state slot to reload (1-8, default: 5), this admit that you have save a savestate which is in quest (ideally at the beginning)
- `--resume PATH.zip` - Resume from checkpoint
- `--keyboard` - Use keyboard instead of virtual controller
- `--grayscale` - Use grayscale frames (faster)
- `--rtvision` - Display real-time vision window
- `--rtminimap` - Display real-time exploration map
- `--cpu` - Force CPU training (disable GPU)

#### Multi-Agent Training

**Example: 8 agents on 4 instances**
```bash
python train.py \
  --num-agents 8 \
  --num-instances 4 \
  --timesteps 2000000 \
  --allocation-mode auto \
  --multi-agent-mode independent
```

**Multi-agent arguments:**
- `--num-agents N` - Number of PPO agents (1-32)
- `--num-instances N` - Number of Dolphin instances (1-16)
- `--allocation-mode MODE` - `auto`, `manual`, or `weighted`
- `--multi-agent-mode MODE` - `independent`, `round_robin`, `majority_vote`, `genetic`
- `--allocation-map MAP` - Manual allocation (format: "0:0,1;1:2,3")
- `--steps-per-agent N` - Steps collected per agent before update (default: 4096)
- `--block-size N` - Block size for round-robin mode (default: 100)

**Scenarios:**
1. **One-to-One** (N agents = N instances): Each agent has dedicated instance
2. **Multiple Instances** (N agents < M instances): Each agent controls multiple instances
3. **Instance Sharing** (N agents > M instances): Multiple agents share instances

#### Genetic Algorithm

```bash
python train.py \
  --num-agents 16 \
  --num-instances 8 \
  --multi-agent-mode genetic \
  --genetic-generations 10 \
  --genetic-elite-ratio 0.25 \
  --genetic-mutation-rate 0.3
```

**Genetic arguments:**
- `--genetic-generations N` - Number of generations (default: 10)
- `--genetic-elite-ratio FLOAT` - Elite preservation ratio (default: 0.25)
- `--genetic-mutation-rate FLOAT` - Mutation rate (default: 0.3)

---

### Testing Trained Agent

```bash
python test.py \
  --model-path models/my_experiment/final_model.zip \
  --n-episodes 10 \
  --deterministic
```

**Test arguments:**
- `--model-path PATH` - Path to trained model (.zip)
- `--n-episodes N` - Number of test episodes (default: 10)
- `--deterministic` - Use deterministic policy (no exploration)
- `--render` - Display game window during testing

---

## Training Results

### Expected Learning Phases

I haven’t improved or tested the model enough yet to know if it’s really good, this is still like an alpha.

### Monitoring Training

**Real-time GUI:**
- Episode rewards and length
- Player stats (HP, stamina, deaths)
- Reward breakdown by category
- Exploration map (3D visualization)
- Monster HP tracking

**TensorBoard:**
```bash
tensorboard --logdir ./logs/
```
- Policy loss, value loss, entropy
- Learning rate schedule
- Episode statistics

**Console Logs:**
- Episode summaries every 100 steps
- Reward components (survival, combat, exploration, penalties)
- Monster zone detection
- Cube compression events

---

### Virtual Controller Not Working

**Checklist:**
1. ViGEmBus installed?
2. Windows restarted after installation?
3. `python core/controller.py` creates controller?
4. Dolphin detects "Xbox 360 Controller"?
5. Controller mapped to Wiimote 1?
6. In Options > Interface, uncheck ‘keyboard shortcuts require the window to be focused.’
7. Ultimetely : Game window had the focus? (start training with only one instance and click on the game window)

**If still not working:**
- Close/reopen Dolphin
- Check Device Manager → Xbox Peripherals
- Reinstall ViGEmBus

### Black/Empty Frames

**Solutions:**
- Keep Dolphin window **visible** (not minimized)
- Use **Windowed** mode (not fullscreen)
- Graphics backend: `Direct3D 11` or `Vulkan`
- Disable "Render to Main Window"

### CUDA Out of Memory

**Solutions:**

```bash
# Reduce batch size
python train.py --batch-size 32

# Use grayscale
python train.py --grayscale

# Reduce frame stack
python train.py --frame-stack 2

# Force CPU
python train.py --cpu
```

### Multi-Instance: Windows Not Renamed

**Symptoms**: All instances capture same window

**Solutions:**
1. Verify PowerShell script includes renaming logic
2. Check PowerShell output for errors
3. Increase `--dolphin-delay` in launch script
4. Manually rename windows if needed

---

## Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -am 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution

- **Translation**: Convert French comments to English
- **Documentation**: Improve README, add tutorials
- **Features**: New reward components, alternative CNNs
- **Bug fixes**: Improve stability, fix edge cases
- **Testing**: Add unit tests, integration tests

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Capcom** for Monster Hunter Tri
- **Dolphin Emulator Team** for the excellent emulator
- **Stable-Baselines3** for RL implementations
- **ViGEmBus** for virtual controller support
- **OpenAI** for PPO algorithm
- The entire **RL community**

---
