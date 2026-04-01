"""
check_setup.py — Diagnostic script for Monster Hunter Tri RL project.
Run this BEFORE starting training to verify the entire setup.

Checks performed (in order):
    1.  Python version (3.8+ required)
    2.  Required Python packages
    3.  Project folder/file structure (per structure-en.txt)
    4.  Config files (dolphin_path_config.json, crop_config.json)
    5.  Dolphin.exe path validity
    6.  Dolphin input hook DLL (dolphin_input_hook.dll / Rust source)
    7.  DolphinCapture.dll (screen capture)
    8.  GPU / CUDA availability
    9.  CNN inference speed benchmark
   10.  Memory address validation (DME ranges)
   11.  Dolphin memory connection (live — dolphin-memory-engine)
   12.  Live game state dump (HP, zone, inventory, monsters, menu...)
   13.  Frame capture (window detection + DLL + brightness)
   14.  Multi-agent module imports

Usage:
    python check_setup.py               # full check (skips live checks if Dolphin not running)
    python check_setup.py --state       # only dump live game state
    python check_setup.py --quick       # skip all live checks (memory, frame, game state)
    python check_setup.py --no-live     # alias for --quick
"""

import sys
import os
import json
import time
import argparse
import ctypes
from pathlib import Path
from typing import Tuple, List, Optional

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

SEP  = "=" * 72
SEP2 = "-" * 72

def _hdr(text: str):
    print(f"\n{SEP}\n  {text}\n{SEP}")

def _ok(label: str, detail: str = ""):
    suffix = f"  →  {detail}" if detail else ""
    print(f"  ✅  {label}{suffix}")

def _warn(label: str, detail: str = ""):
    suffix = f"  →  {detail}" if detail else ""
    print(f"  ⚠️   {label}{suffix}")

def _fail(label: str, detail: str = ""):
    suffix = f"  →  {detail}" if detail else ""
    print(f"  ❌  {label}{suffix}")

def _info(text: str):
    print(f"       {text}")

def _tip(text: str):
    print(f"  💡  {text}")

def _skip(label: str, reason: str = ""):
    suffix = f"  ({reason})" if reason else ""
    print(f"  ⏭️   {label}{suffix}")


# ===========================================================================
# 1. Python version
# ===========================================================================

def check_python_version() -> bool:
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if v.major >= 3 and v.minor >= 8:
        _ok(label, "required 3.8+")
        return True
    _fail(label, "Python 3.8+ required — please upgrade")
    return False


# ===========================================================================
# 2. Required packages
# ===========================================================================

# (import_name, display_label, critical, install_hint)
REQUIRED_PACKAGES: List[Tuple[str, str, bool, str]] = [
    ("numpy",               "numpy",                    True,  "numpy"),
    ("torch",               "PyTorch",                  True,  "torch (see pytorch.org)"),
    ("gymnasium",           "Gymnasium",                True,  "gymnasium"),
    ("stable_baselines3",   "Stable-Baselines3",        True,  "stable-baselines3"),
    ("cv2",                 "OpenCV (cv2)",              True,  "opencv-python"),
    ("win32gui",            "PyWin32 (win32gui)",        True,  "pywin32"),
    ("win32process",        "PyWin32 (win32process)",   True,  "pywin32"),
    ("dolphin_memory_engine","dolphin-memory-engine",   True,  "dolphin-memory-engine"),
    ("psutil",              "psutil",                   True,  "psutil"),
    ("matplotlib",          "matplotlib",               False, "matplotlib"),
    ("tqdm",                "tqdm",                     False, "tqdm"),
]

def check_packages() -> bool:
    all_critical_ok = True
    for module, label, critical, hint in REQUIRED_PACKAGES:
        try:
            pkg = __import__(module)
            version = getattr(pkg, "__version__", "?")
            _ok(label, f"v{version}")
        except ImportError:
            if critical:
                _fail(label, f"MISSING — pip install {hint}")
                all_critical_ok = False
            else:
                _warn(label, f"optional, not installed — pip install {hint}")
    return all_critical_ok


# ===========================================================================
# 3. Project structure
# ===========================================================================

EXPECTED_DIRS = [
    "agent",
    "config",
    "config/user",
    "core",
    "core/controller",
    "environment",
    "reward",
    "GUI",
    "hook",
    "hook/src",
    "info",
    "multi",
    "train",
    "utils",
    "vision",
]

EXPECTED_FILES = [
    # Root
    "train.py",
    "check_setup.py",
    # agent/
    "agent/__init__.py",
    "agent/extractors.py",
    "agent/ppo_agent.py",
    # config/
    "config/__init__.py",
    "config/memory_addresses.py",
    # core/
    "core/__init__.py",
    "core/dynamic_memory_reader.py",
    "core/exploration_map_incremental.py",
    "core/memory_normalizer.py",
    "core/memory_state_builder.py",
    "core/state_fusion.py",
    # core/controller/
    "core/controller/__init__.py",
    "core/controller/action_heads.py",
    "core/controller/action_resolver.py",
    "core/controller/constants.py",
    "core/controller/dll_utils.py",
    "core/controller/key_state_manager.py",
    "core/controller/wii_controller.py",
    # environment/
    "environment/__init__.py",
    "environment/mh_env.py",
    "environment/capture_mixin.py",
    "environment/episode_mixin.py",
    "environment/observation_mixin.py",
    "environment/realtime_display.py",
    "environment/reward_bridge_mixin.py",
    "environment/sanitizer.py",
    "environment/spaces.py",
    # reward/
    "reward/__init__.py",
    "reward/reward_calculator.py",
    "reward/exploration_tracker.py",
    "reward/cube_markers.py",
    "reward/camp_tracker.py",
    "reward/monster_zone_tracker.py",
    "reward/oxygen_tracker.py",
    # GUI/
    "GUI/__init__.py",
    "GUI/gui_header.py",          # Header panel (agent selector, metrics, stop button)
    "GUI/gui_map3d.py",           # 3D exploration map panel
    "GUI/gui_state.py",           # Centralized data store (no tkinter dependency)
    "GUI/gui_statusbar.py",       # Bottom status bar (FPS, isolation warnings)
    "GUI/gui_tab_actions.py",     # Actions tab (7-head schematic + compatibility)
    "GUI/gui_tab_charts.py",      # Charts tab (reward/length/hits matplotlib)
    "GUI/gui_tab_combat.py",      # Combat tab (monster HP, zones, combat state)
    "GUI/gui_tab_overview.py",    # Overview tab (HP, stamina, 24-slot inventory)
    "GUI/gui_tab_rewards.py",     # Rewards tab (16-category breakdown bars)
    "GUI/gui_theme.py",           # Dark terminal theme + reusable widgets
    "GUI/training_gui.py",        # Main GUI assembler (delegates to modules)
    # hook/
    "hook/Cargo.toml",
    "hook/src/lib.rs",
    # info/
    "info/__init__.py",
    "info/advanced_logging.py",
    "info/agent_context.py",
    "info/module_logger.py",
    # multi/
    "multi/__init__.py",
    "multi/genetic_trainer.py",
    "multi/multi_agent_scheduler.py",
    "multi/multi_agent_trainer.py",
    # train/
    "train/__init__.py",
    "train/__main__.py",
    "train/agents.py",
    "train/allocation.py",
    "train/callbacks.py",
    "train/cli.py",
    "train/dolphin.py",
    "train/environment.py",
    "train/logging_setup.py",
    "train/runner.py",
    # utils/
    "utils/__init__.py",
    "utils/item_id.txt",
    "utils/weapon_id.txt",
    "utils/memory_vector.py",
    "utils/safe_float.py",
    # vision/
    "vision/__init__.py",
    "vision/dolphin_capture_dll.py",
    "vision/feature_extractor.py",
    "vision/frame_capture.py",
    "vision/hud_crop_tuner.py",
    "vision/preprocessing.py",
]

# Files that are generated/optional — warn instead of fail
OPTIONAL_FILES = [
    "config/crop_config.json",
    "config/dolphin_path_config.json",
    "config/user/gui_config.json",
    "vision/DolphinCapture.dll",
    "vision/dolphin_input_hook.dll",
]

def check_structure() -> bool:
    ok = True

    print("\n  Packages / folders:")
    for d in EXPECTED_DIRS:
        if os.path.isdir(d):
            _ok(f"  {d}/")
        else:
            _fail(f"  {d}/", "missing")
            ok = False

    print("\n  Source files:")
    for f in EXPECTED_FILES:
        if os.path.isfile(f):
            _ok(f"  {f}")
        else:
            _fail(f"  {f}", "missing")
            ok = False

    print("\n  Generated / optional files:")
    for f in OPTIONAL_FILES:
        if os.path.isfile(f):
            size_kb = os.path.getsize(f) // 1024
            _ok(f"  {f}", f"{size_kb} KB")
        else:
            _warn(f"  {f}", "not present (may be auto-generated or not yet configured)")

    return ok



# ===========================================================================
# 5. Config files
# ===========================================================================

def check_config_files() -> bool:
    ok = True

    # --- dolphin_path_config.json ---
    path = Path("config/dolphin_path_config.json")
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            dolphin = data.get("dolphin_path", "")
            _ok("dolphin_path_config.json",
                f"dolphin_path = {dolphin}" if dolphin else "dolphin_path EMPTY")
            if not dolphin:
                _warn("dolphin_path is empty",
                      "will be prompted automatically on first launch")
        except json.JSONDecodeError as e:
            _fail("dolphin_path_config.json", f"invalid JSON: {e}")
            ok = False
    else:
        _warn("dolphin_path_config.json",
              "not found — will be prompted at first launch and auto-saved")

    # --- crop_config.json ---
    crop = Path("config/crop_config.json")
    if crop.exists():
        try:
            data = json.loads(crop.read_text(encoding="utf-8"))
            keys = {"top_crop", "bottom_crop", "left_crop", "right_crop"}
            if keys.issubset(data.keys()):
                _ok("crop_config.json",
                    f"top={data['top_crop']}  bottom={data['bottom_crop']}  "
                    f"left={data['left_crop']}  right={data['right_crop']}")
            else:
                _warn("crop_config.json", f"missing keys: {keys - data.keys()}")
        except json.JSONDecodeError as e:
            _fail("crop_config.json", f"invalid JSON: {e}")
            ok = False
    else:
        _warn("crop_config.json",
              "not found — default crop values will be used (run hud_crop_tuner.py to calibrate)")

    # --- gui_config.json (user preference, truly optional) ---
    gui_cfg = Path("config/user/gui_config.json")
    if gui_cfg.exists():
        _ok("config/user/gui_config.json", "GUI preferences saved")
    else:
        _info("config/user/gui_config.json — not present (created automatically by the GUI)")

    return ok


# ===========================================================================
# 6. Dolphin.exe path
# ===========================================================================

def check_dolphin_path() -> Tuple[bool, Optional[str]]:
    """Returns (ok, dolphin_exe_path_or_None)."""
    config_path = Path("config/dolphin_path_config.json")
    if not config_path.exists():
        _warn("Dolphin path", "config file not found — skipping")
        return True, None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        dolphin = data.get("dolphin_path", "").strip()
        if not dolphin:
            _warn("Dolphin path", "dolphin_path is empty in config")
            return True, None

        p = Path(dolphin)
        if p.is_dir():
            exe = p / "Dolphin.exe"
            if exe.is_file():
                _ok("Dolphin.exe found", str(exe))
                return True, str(exe)
            _fail("Dolphin.exe", f"not found inside folder: {p}")
            _tip(f"Expected: {exe}")
            return False, None
        elif p.is_file() and p.name.lower() == "dolphin.exe":
            _ok("Dolphin.exe found", str(p))
            return True, str(p)
        else:
            _fail("Dolphin path", f"path does not exist or is not Dolphin.exe: {p}")
            return False, None

    except Exception as e:
        _fail("Dolphin path config", str(e))
        return False, None


# ===========================================================================
# 7. Input hook DLL (dolphin_input_hook.dll + Rust source)
# ===========================================================================

def check_input_hook_dll() -> bool:
    # Check prebuilt DLL locations
    dll_candidates = [
        Path("vision/dolphin_input_hook.dll"),
        Path("dolphin_input_hook.dll"),
        Path("hook/target/release/dolphin_input_hook.dll"),
    ]
    for p in dll_candidates:
        if p.exists():
            _ok("dolphin_input_hook.dll",
                f"found at {p}  ({p.stat().st_size // 1024} KB)")
            break
    else:
        # DLL absent — check if Rust source available for auto-build
        if Path("hook/Cargo.toml").exists() and Path("hook/src/lib.rs").exists():
            _warn("dolphin_input_hook.dll",
                  "not built yet — will be auto-compiled on first training run")
            _tip("Requires Rust/cargo in PATH: https://rustup.rs")
            # Check if cargo is available
            import subprocess
            try:
                subprocess.run(["cargo", "--version"],
                               check=True, capture_output=True, timeout=5)
                _ok("cargo (Rust toolchain)", "available — auto-build will work")
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                _warn("cargo", "not found in PATH — install from https://rustup.rs")
        else:
            _fail("dolphin_input_hook.dll",
                  "not found and hook/Cargo.toml or hook/src/lib.rs missing")
            _tip("Rebuild the hook/ folder or copy a prebuilt DLL to vision/")
            return False

    return True


# ===========================================================================
# 8. DolphinCapture.dll (screen capture)
# ===========================================================================

def check_capture_dll() -> bool:
    candidates = [
        Path("vision/DolphinCapture.dll"),
        Path("DolphinCapture.dll"),
    ]
    for p in candidates:
        if p.exists():
            _ok("DolphinCapture.dll",
                f"found at {p}  ({p.stat().st_size // 1024} KB)")
            return True

    _warn("DolphinCapture.dll", "not found in vision/ or project root")
    _tip("Frame capture will fall back to GDI — Dolphin window must remain visible")
    return False


# ===========================================================================
# 9. GPU / CUDA
# ===========================================================================

def check_cuda() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            name    = torch.cuda.get_device_name(0)
            mem_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram    = f"{mem_gb:.1f} GB VRAM"
            _ok("CUDA", f"{name}  ({vram})")
            if mem_gb < 4:
                _warn("VRAM < 4 GB", "training may be slow or OOM with large batches")
            return True
        else:
            _warn("CUDA not available",
                  "training will run on CPU — significantly slower")
            _tip("Install CUDA-enabled PyTorch: https://pytorch.org/get-started")
            return False
    except ImportError:
        _fail("PyTorch not installed", "cannot check CUDA")
        return False
    except Exception as e:
        _fail("CUDA check error", str(e))
        return False


# ===========================================================================
# 10. CNN inference speed benchmark
# ===========================================================================

def check_inference_speed() -> bool:
    try:
        import torch
        from vision.feature_extractor import NatureCNN

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = NatureCNN(input_channels=4, features_dim=256).to(device)
        dummy  = torch.randn(4, 4, 84, 84).to(device)

        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                model(dummy)

        # Benchmark 50 batches
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                model(dummy)
        elapsed = time.perf_counter() - t0

        fps   = (50 * 4) / elapsed
        label = f"~{fps:.0f} inferences/sec  (batch=4, 84×84, stack=4, device={device.upper()})"

        if fps > 500:
            _ok("CNN speed", label)
        elif fps > 100:
            _warn("CNN speed", f"{label} — acceptable but GPU recommended")
        else:
            _warn("CNN speed", f"{label} — very slow, consider enabling CUDA")

        return True

    except ImportError as e:
        _fail("CNN benchmark", f"import error: {e}")
        return False
    except Exception as e:
        _fail("CNN benchmark", str(e))
        return False


# ===========================================================================
# 11. Memory address validation
# ===========================================================================

def check_memory_addresses() -> bool:
    try:
        import config.memory_addresses as addr
    except ImportError as e:
        _fail("config/memory_addresses.py", str(e))
        return False

    invalid   = []
    valid_mem1 = 0
    valid_mem2 = 0

    for name in dir(addr):
        if name.startswith("_") or name == "validate_addresses":
            continue
        value = getattr(addr, name)

        # Resolve dual-context tuples → quest address (index 1)
        if isinstance(value, tuple) and len(value) == 2:
            value = value[1]

        if not isinstance(value, int) or value <= 0:
            continue

        in_mem1 = 0x80000000 <= value < 0x81800000
        in_mem2 = 0x90000000 <= value < 0x94000000

        if in_mem1:
            valid_mem1 += 1
        elif in_mem2:
            valid_mem2 += 1
        else:
            invalid.append((name, hex(value)))

    total = valid_mem1 + valid_mem2 + len(invalid)
    if not invalid:
        _ok("Memory addresses",
            f"{total} addresses — MEM1={valid_mem1}  MEM2={valid_mem2}  invalid=0")
        return True
    else:
        _warn("Memory addresses",
              f"{len(invalid)} address(es) outside valid DME ranges "
              f"(MEM1: 0x80000000–0x81800000 / MEM2: 0x90000000–0x94000000):")
        for name, addr_hex in invalid:
            _info(f"    {name}: {addr_hex}")
        return False


# ===========================================================================
# 12. Dolphin memory connection (live)
# ===========================================================================

def check_dolphin_memory() -> Tuple[bool, Optional[object]]:
    """Returns (ok, reader_or_None)."""
    try:
        from core.dynamic_memory_reader import MemoryReader
    except ImportError as e:
        _fail("MemoryReader import", str(e))
        return False, None

    try:
        reader = MemoryReader(force_quest_mode=True, async_mode=False)
        _ok("Dolphin memory hook", "connected successfully")
        return True, reader
    except ImportError as e:
        _fail("dolphin-memory-engine", f"not installed: {e}")
        _tip("pip install dolphin-memory-engine")
        return False, None
    except ConnectionError as e:
        _fail("Dolphin connection", str(e))
        _tip("1. Start Dolphin")
        _tip("2. Load Monster Hunter Tri")
        _tip("3. Enter a quest (be IN-GAME, not in village)")
        return False, None
    except Exception as e:
        _fail("Dolphin memory", f"unexpected error: {e}")
        return False, None


# ===========================================================================
# 13. Live game state dump
# ===========================================================================

def dump_game_state(reader) -> bool:
    try:
        state = reader.read_game_state()
    except Exception as e:
        _fail("read_game_state()", str(e))
        return False

    print()

    # --- Map / quest status ---
    current_map = state.get("current_map")
    if current_map == 45:
        _warn("Map",
              "MAP=45 → reward/end screen — load a save state INSIDE an active quest")
    elif current_map == 100:
        _ok("Map", "MAP=100 → in quest ✓")
    elif current_map == 0:
        _warn("Map", "MAP=0 → title/loading screen — load a quest")
    else:
        _warn("Map", f"MAP={current_map} — unexpected value")

    quest_ended = state.get("quest_ended", False)
    if quest_ended:
        _warn("Quest ended flag", "set — this save state has an expired quest")

    # --- Player HP / Stamina ---
    hp      = state.get("player_hp")
    stamina = state.get("player_stamina")
    hp_raw  = state.get("player_hp_raw")

    if hp is not None:
        color = "✓" if hp > 50 else "low!"
        _ok("HP", f"{hp:.1f}/100  ({color})  raw={hp_raw}")
    else:
        _warn("HP", "None — are you in a quest?")

    if stamina is not None:
        _ok("Stamina", f"{stamina:.1f}/100")
    else:
        _warn("Stamina", "None")

    # --- Position / zone ---
    x, y, z = state.get("player_x"), state.get("player_y"), state.get("player_z")
    zone     = state.get("current_zone")
    orient   = state.get("player_orientation")

    if x is not None:
        _ok("Position", f"({x:.1f}, {y:.1f}, {z:.1f})")
        _info(f"Orientation: {orient}°   Zone: {zone}")
        if zone == 0:
            _warn("Zone 0", "player is in the starting camp")
    else:
        _warn("Position", "None — not in quest?")

    # --- Quest time ---
    qt = state.get("quest_time")
    if qt is not None:
        mins, secs = divmod(int(qt), 60)
        _ok("Quest time", f"{mins}:{secs:02d} remaining ({qt}s)")
        if qt <= 0:
            _warn("Quest time", "EXPIRED — save state has a dead quest")
        elif qt < 60:
            _warn("Quest time", "less than 1 minute left")
    else:
        _warn("Quest time", "None")

    # --- Deaths ---
    deaths = state.get("death_count")
    if deaths is not None:
        _ok("Deaths", f"{deaths}/3")
        if deaths >= 3:
            _warn("Deaths ≥ 3", "quest FAILED state — save state is unusable, reload")
    else:
        _warn("Deaths", "None")

    # --- Sharpness ---
    sharp = state.get("sharpness")
    if sharp == -2:
        _warn("Sharpness", "-2 (weapon bouncing)")
    elif sharp is not None:
        _ok("Sharpness", str(sharp))

    # --- In-game menu ---
    in_menu = state.get("in_game_menu", False)
    if in_menu:
        _warn("In-game menu", "OPEN — close it before training starts")
    else:
        _ok("In-game menu", "closed ✓")

    # --- Oxygen ---
    oxy       = state.get("time_underwater")
    oxy_valid = state.get("oxygen_valid", False)
    if oxy_valid and oxy is not None:
        _ok("Oxygen", f"{oxy}/100")
        if oxy < 25:
            _warn("Oxygen", "CRITICALLY LOW — player is drowning")
    else:
        _info("Oxygen: n/a (player not underwater)")

    # --- Monsters ---
    any_monster = False
    for i in range(1, 6):
        hp_m = state.get(f"smonster{i}_hp")
        if hp_m is not None and hp_m > 0:
            _ok(f"Small monster {i}", f"{hp_m} HP")
            any_monster = True
    lm1 = state.get("lmonster1_hp")
    if lm1 is not None and lm1 > 0:
        _ok("Large monster 1", f"{lm1} HP")
        any_monster = True
    if not any_monster:
        _info("No monsters detected in current zone")

    # --- Inventory ---
    inventory = state.get("inventory_items", [])
    if inventory:
        _ok("Inventory", f"{len(inventory)} item(s)")
        for item in inventory[:5]:
            _info(f"  Slot {item['slot']:2d}: {item.get('name', '?'):35s}  x{item.get('quantity', '?')}")
        if len(inventory) > 5:
            _info(f"  ... and {len(inventory) - 5} more")
    else:
        _warn("Inventory", "empty or unreadable (normal in camp before quest start)")

    # --- Memory vector sanity ---
    try:
        from utils.memory_vector import build_memory_vector
        vec = build_memory_vector(state)
        import numpy as np
        has_nan = bool(np.any(np.isnan(vec)) or np.any(np.isinf(vec)))
        if has_nan:
            _warn("Memory vector (70 features)", "contains NaN/Inf — check memory_normalizer.py")
        else:
            _ok("Memory vector (70 features)", f"shape={vec.shape}  range=[{vec.min():.2f}, {vec.max():.2f}]")
    except Exception as e:
        _warn("Memory vector", f"could not build: {e}")

    return True


# ===========================================================================
# 14. Frame capture (live)
# ===========================================================================

def check_frame_capture() -> bool:
    try:
        from vision.frame_capture import FrameCapture
    except ImportError as e:
        _fail("frame_capture import", str(e))
        return False

    try:
        cap = FrameCapture(use_dll=True)

        if cap.hwnd is None:
            _fail("Dolphin window", "not found — is Dolphin running?")
            _tip("Window title must contain 'MHTri' or 'Monster Hunter'")
            return False

        import win32gui
        title = win32gui.GetWindowText(cap.hwnd)
        _ok("Dolphin window", f"hwnd={cap.hwnd}   title='{title}'")

        if cap.use_dll and cap.dll_instance_id >= 0:
            _ok("DolphinCapture.dll", "active (robust, focus-free capture)")
        else:
            _warn("DolphinCapture.dll", "not active — falling back to GDI (window must stay visible)")

        frame = cap.capture_frame()
        if frame is None or frame.size == 0:
            _fail("Frame capture", "returned an empty frame")
            return False

        brightness = float(frame.mean())
        if brightness < 5:
            _warn("Frame brightness", f"{brightness:.1f} — frame is black")
            _tip("Make sure the Dolphin window is NOT minimized (or use DolphinCapture.dll)")
            return False

        _ok("Frame captured",
            f"shape={frame.shape}   dtype={frame.dtype}   brightness={brightness:.1f}")
        return True

    except ValueError as e:
        _fail("Window detection", str(e))
        _tip("Ensure Dolphin is running and a game window is visible")
        return False
    except Exception as e:
        _fail("Frame capture", str(e))
        return False


# ===========================================================================
# 15. Multi-agent module imports
# ===========================================================================

def check_multi_agent_modules() -> bool:
    modules = [
        ("multi.multi_agent_scheduler", "MultiAgentScheduler"),
        ("multi.multi_agent_trainer",   "MultiAgentTrainer"),
        ("multi.genetic_trainer",       "GeneticTrainer"),
        ("train.allocation",            "calculate_agent_allocation"),
        ("train.agents",                "create_multi_agents"),
        ("train.callbacks",             "GUIUpdateCallback"),
        ("train.runner",                "main"),
    ]
    ok = True
    for module, symbol in modules:
        try:
            mod = __import__(module, fromlist=[symbol])
            getattr(mod, symbol)
            _ok(f"{module}.{symbol}")
        except ImportError as e:
            _fail(f"{module}", f"import error: {e}")
            ok = False
        except AttributeError:
            _fail(f"{module}.{symbol}", "symbol not found in module")
            ok = False
    return ok


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Monster Hunter Tri RL — Full Setup Diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_setup.py             # full check (auto-detects if Dolphin is live)
  python check_setup.py --state     # only dump live game state
  python check_setup.py --quick     # skip all live checks (fast, offline)
        """,
    )
    parser.add_argument("--state",   action="store_true",
                        help="Only dump live game state (Dolphin must be running)")
    parser.add_argument("--quick",   action="store_true",
                        help="Skip live checks (memory, frame, game state)")
    parser.add_argument("--no-live", action="store_true",
                        help="Alias for --quick")
    args = parser.parse_args()

    skip_live = args.quick or args.no_live

    # ---- Special mode: game state dump only ----
    if args.state:
        _hdr("LIVE GAME STATE DUMP")
        mem_ok, reader = check_dolphin_memory()
        if mem_ok and reader:
            dump_game_state(reader)
        print()
        return

    # ====================================================================
    # FULL DIAGNOSTIC
    # ====================================================================
    _hdr("MONSTER HUNTER TRI RL — SETUP DIAGNOSTIC")
    _info(f"Run from : {os.getcwd()}")
    _info(f"Platform : {sys.platform}  |  Python {sys.version.split()[0]}")
    _info(f"Mode     : {'quick (no live checks)' if skip_live else 'full'}")

    results = {}

    _hdr("1 — Python version")
    results["python"] = check_python_version()

    _hdr("2 — Python packages")
    results["packages"] = check_packages()

    _hdr("3 — Project structure")
    results["structure"] = check_structure()

    _hdr("4 — Config files")
    results["config"] = check_config_files()

    _hdr("5 — Dolphin.exe path")
    dolphin_ok, _dolphin_exe = check_dolphin_path()
    results["dolphin_path"] = dolphin_ok

    _hdr("6 — Input hook DLL  (dolphin_input_hook.dll)")
    results["input_hook_dll"] = check_input_hook_dll()

    _hdr("7 — Screen capture DLL  (DolphinCapture.dll)")
    results["capture_dll"] = check_capture_dll()

    _hdr("8 — GPU / CUDA")
    results["cuda"] = check_cuda()

    _hdr("9 — CNN inference speed")
    results["cnn_speed"] = check_inference_speed()

    _hdr("10 — Memory address validation")
    results["addresses"] = check_memory_addresses()

    _hdr("11 — Multi-agent module imports")
    results["multi_agent"] = check_multi_agent_modules()

    # ---- Live checks ----
    if skip_live:
        _hdr("12-14 — Live checks  (SKIPPED — use --quick / --no-live to enable)")
        _skip("Dolphin memory connection", "--quick active")
        _skip("Game state dump",           "--quick active")
        _skip("Frame capture",             "--quick active")
    else:
        _hdr("12 — Dolphin memory connection  (live)")
        mem_ok, reader = check_dolphin_memory()
        results["memory"] = mem_ok

        if mem_ok and reader:
            _hdr("13 — Live game state")
            results["game_state"] = dump_game_state(reader)
        else:
            _skip("13 — Game state dump", "memory not connected")

        _hdr("14 — Frame capture  (live)")
        results["capture"] = check_frame_capture()

    # ====================================================================
    # SUMMARY
    # ====================================================================
    _hdr("SUMMARY")

    groups = {
        "🔴 Critical  (training will NOT start)": [
            "python", "packages", "structure",
        ],
        "🟠 Important  (training may misbehave)": [
            "config", "dolphin_path", "addresses", "multi_agent",
        ],
        "🟡 Hardware / performance": [
            "cuda", "cnn_speed", "input_hook_dll", "capture_dll",
        ],
        "🟢 Live checks  (Dolphin must be running)": [
            "memory", "game_state", "capture",
        ],
    }

    all_critical_ok = True
    all_important_ok = True

    for group_label, keys in groups.items():
        print(f"\n  {group_label}:")
        for k in keys:
            if k not in results:
                _skip(f"  {k}", "not run")
            elif results[k]:
                _ok(f"  {k}")
            else:
                _fail(f"  {k}")
                if group_label.startswith("🔴"):
                    all_critical_ok = False
                elif group_label.startswith("🟠"):
                    all_important_ok = False

    print()
    if all_critical_ok and all_important_ok:
        print(f"  {'='*60}")
        print(f"  ✅  READY TO TRAIN!")
        print(f"  {'='*60}")
        _info("")
        _info("Suggested first run:")
        _info("  python train.py --timesteps 10000 --name test_run")
        _info("")
        _info("Multi-instance run (6 agents, 6 instances):")
        _info("  python train.py --num-agents 6 --num-instances 6 --timesteps 100000")
    else:
        print(f"  {'='*60}")
        print(f"  ❌  SETUP INCOMPLETE — fix the issues above")
        print(f"  {'='*60}")
        _tip("")
        _tip("Quick checklist:")
        _tip("  1. pip install -r requirements.txt")
        _tip("  2. Launch Dolphin")
        _tip("  3. Load Monster Hunter Tri → enter a quest")
        _tip("  4. Re-run:  python check_setup.py")

    # ---- System info footer ----
    _hdr("System info")
    print(f"  Python  : {sys.version.split()[0]}  ({sys.executable})")
    try:
        import torch
        cuda_str = (f"CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}"
                    if torch.cuda.is_available() else "not available")
        print(f"  PyTorch : {torch.__version__}  |  CUDA: {cuda_str}")
    except ImportError:
        print(f"  PyTorch : not installed")
    try:
        import stable_baselines3 as sb3
        print(f"  SB3     : {sb3.__version__}")
    except ImportError:
        pass
    try:
        import gymnasium
        print(f"  Gym     : {gymnasium.__version__}")
    except ImportError:
        pass
    try:
        import cv2
        print(f"  OpenCV  : {cv2.__version__}")
    except ImportError:
        pass
    try:
        import numpy as np
        print(f"  NumPy   : {np.__version__}")
    except ImportError:
        pass
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()