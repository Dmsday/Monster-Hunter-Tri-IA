"""
check_setup.py — Monster Hunter Tri IA — Setup Diagnostic Tool
==============================================================
Run this script BEFORE launching training to verify your environment.

Usage:
    python check_setup.py               # Static checks only (no Dolphin needed)
    python check_setup.py --live        # + Live checks (Dolphin must be running with game loaded)
    python check_setup.py --controller  # + Virtual controller creation test
    python check_setup.py --full        # All checks

Each section prints:  ✅ OK  |  ⚠️  WARNING  |  ❌ FAIL
"""

import sys
import os
import argparse
import importlib
import platform
import time

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_results = {"ok": 0, "warn": 0, "fail": 0}

def ok(msg):
    print(f"  ✅  {msg}")
    _results["ok"] += 1

def warn(msg):
    print(f"  ⚠️   {msg}")
    _results["warn"] += 1

def fail(msg):
    print(f"  ❌  {msg}")
    _results["fail"] += 1

def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def check_import(package_name, import_name=None, min_version=None, attr_version=None):
    """
    Try to import a package and optionally verify its version.
    package_name   : display name (e.g. "opencv-python")
    import_name    : actual import name if different (e.g. "cv2")
    min_version    : minimum required version string (e.g. "4.8.0")
    attr_version   : attribute holding version on the module (default: "__version__")
    """
    name = import_name or package_name
    try:
        mod = importlib.import_module(name)
        version_attr = attr_version or "__version__"
        version = getattr(mod, version_attr, None)
        version_str = f"v{version}" if version else "(version unknown)"

        if min_version and version:
            try:
                from packaging.version import Version
                if Version(str(version)) < Version(min_version):
                    warn(f"{package_name} {version_str} — expected >={min_version}")
                    return mod
            except Exception:
                pass  # packaging not available, skip version check

        ok(f"{package_name} {version_str}")
        return mod
    except ImportError as e:
        fail(f"{package_name} — NOT INSTALLED  ({e})")
        return None
    except Exception as e:
        fail(f"{package_name} — import error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def check_system():
    section("1 / SYSTEM")

    if platform.system() == "Windows":
        ok(f"OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    else:
        fail(f"OS: {platform.system()} — This project is Windows-only "
             "(win32gui, ViGEmBus, Dolphin are Windows dependencies)")

    major, minor = sys.version_info[:2]
    version_str = f"Python {major}.{minor}.{sys.version_info[2]}"
    if (major, minor) >= (3, 8):
        ok(f"{version_str}")
    else:
        fail(f"{version_str} — Python 3.8+ required")

    cwd = os.getcwd()
    expected_files = ["train.py", "launch_dolphin_instances.ps1"]
    if all(os.path.isfile(f) for f in expected_files):
        ok(f"Working directory looks correct: {cwd}")
    else:
        missing = [f for f in expected_files if not os.path.isfile(f)]
        warn(f"Run check_setup.py from the project root. Missing: {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROJECT FILE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def check_structure():
    section("2 / PROJECT FILE STRUCTURE")

    required_dirs = [
        "config", "core", "vision", "environment", "agent", "utils",
    ]

    required_files = {
        "config/memory_addresses.py":           "Memory address constants",
        "core/dynamic_memory_reader.py":        "Async RAM reader",
        "core/state_fusion.py":                 "Vision + memory fusion",
        "core/controller.py":                   "WiiController",
        "core/exploration_map_incremental.py":  "Exploration map",
        "vision/frame_capture.py":              "FrameCapture",
        "vision/preprocessing.py":             "Frame preprocessing",
        "vision/feature_extractor.py":          "CNN architectures",
        "environment/mh_env.py":                "Main Gymnasium environment",
        "environment/reward_calculator.py":     "Reward computation",
        "environment/exploration_tracker.py":   "Exploration tracker",
        "environment/cube_markers.py":          "Zone markers",
        "agent/ppo_agent.py":                   "PPO agent (SB3)",
        "utils/multi_agent_scheduler.py":       "Multi-agent scheduler",
        "utils/multi_agent_trainer.py":         "Multi-agent training loop",
        "utils/genetic_trainer.py":             "Genetic algorithm trainer",
        "utils/hidhide_manager.py":             "HidHide controller isolation",
        "utils/training_gui.py":                "Real-time training GUI",
        "utils/advanced_logging.py":            "Structured logging",
        "utils/module_logger.py":               "Module-level logger",
        "utils/safe_float.py":                  "Safe float conversion",
        "train.py":                             "Main training script",
        "launch_dolphin_instances.ps1":         "PowerShell multi-instance launcher",
    }

    optional_files = {
        "requirements.txt": "Python dependency list",
        "test.py":          "Agent test script",
    }

    for d in required_dirs:
        if os.path.isdir(d):
            ok(f"[DIR]  {d}/")
        else:
            fail(f"[DIR]  {d}/  — MISSING")

    print()
    for path, desc in required_files.items():
        if os.path.isfile(path):
            ok(f"[FILE] {path}")
        else:
            fail(f"[FILE] {path}  — MISSING  ({desc})")

    print()
    for path, desc in optional_files.items():
        if os.path.isfile(path):
            ok(f"[OPT]  {path}")
        else:
            warn(f"[OPT]  {path}  — not found  ({desc})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PYTHON DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies():
    section("3 / PYTHON DEPENDENCIES")

    check_import("pymem",                 min_version="1.13.1")
    check_import("dolphin-memory-engine", import_name="dolphin_memory_engine")
    check_import("opencv-python",         import_name="cv2",   min_version="4.8.0")
    check_import("Pillow",                import_name="PIL",   min_version="10.0.0",
                                          attr_version="__version__")
    check_import("mss",                   min_version="9.0.1")
    check_import("pywin32",               import_name="win32gui")
    check_import("numpy",                 min_version="1.24.0")
    check_import("psutil",                min_version="5.9.0")
    check_import("tqdm",                  min_version="4.65.0")
    check_import("matplotlib",            min_version="3.7.0")
    check_import("pynput",                min_version="1.7.6")

    print()
    check_import("torch",                 min_version="2.0.0")
    check_import("torchvision",           min_version="0.15.0")

    print()
    check_import("stable_baselines3",     min_version="2.1.0")
    check_import("gymnasium",             min_version="0.29.0")

    print()
    check_import("vgamepad",              min_version="0.1.0")

    print()
    check_import("tensorboard",           min_version="2.13.0")
    check_import("wandb",                 min_version="0.15.0")

    print()
    check_import("pandas",                min_version="2.0.0")
    check_import("seaborn",               min_version="0.12.0")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GPU / CUDA
# ─────────────────────────────────────────────────────────────────────────────

def check_gpu():
    section("4 / GPU & CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem_gb = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
                ok(f"CUDA device [{i}]: {name}  ({mem_gb} GB VRAM)")
            ok(f"CUDA version: {torch.version.cuda}")
            try:
                t = torch.zeros(1).cuda()
                del t
                ok("CUDA tensor allocation: OK")
            except Exception as e:
                warn(f"CUDA tensor allocation failed: {e}")
        else:
            warn("CUDA not available — training will run on CPU (much slower)")
            warn("Install CUDA PyTorch: pip install torch torchvision "
                 "--index-url https://download.pytorch.org/whl/cu124")
    except ImportError:
        fail("torch not installed — cannot check GPU")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PROJECT MODULE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

def check_modules():
    section("5 / PROJECT MODULE IMPORTS")

    required_modules = [
        ("config.memory_addresses",          "Memory address constants"),
        ("core.dynamic_memory_reader",       "Async RAM reader"),
        ("core.state_fusion",                "State fusion"),
        ("core.controller",                  "WiiController"),
        ("core.exploration_map_incremental", "Exploration map"),
        ("vision.frame_capture",             "FrameCapture"),
        ("vision.preprocessing",             "Frame preprocessing"),
        ("vision.feature_extractor",         "CNN feature extractor"),
        ("environment.mh_env",               "Gymnasium environment"),
        ("environment.reward_calculator",    "Reward calculator"),
        ("environment.exploration_tracker",  "Exploration tracker"),
        ("environment.cube_markers",         "Cube markers"),
        ("agent.ppo_agent",                  "PPO agent"),
        ("utils.advanced_logging",           "Advanced logging"),
        ("utils.module_logger",              "Module logger"),
        ("utils.safe_float",                 "Safe float"),
        ("utils.training_gui",               "Training GUI"),
    ]

    optional_modules = [
        ("utils.multi_agent_scheduler", "Multi-agent scheduler"),
        ("utils.multi_agent_trainer",   "Multi-agent trainer"),
        ("utils.genetic_trainer",       "Genetic trainer"),
        ("utils.hidhide_manager",       "HidHide manager"),
    ]

    for module_path, desc in required_modules:
        try:
            importlib.import_module(module_path)
            ok(f"{module_path}")
        except ImportError as e:
            fail(f"{module_path}  — ImportError: {e}  ({desc})")
        except Exception as e:
            fail(f"{module_path}  — Error: {e}  ({desc})")

    print()
    for module_path, desc in optional_modules:
        try:
            importlib.import_module(module_path)
            ok(f"{module_path}  [optional]")
        except ImportError as e:
            warn(f"{module_path}  — not available: {e}  ({desc})")
        except Exception as e:
            warn(f"{module_path}  — error: {e}  ({desc})")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MEMORY ADDRESS VALIDATION  (static — no Dolphin needed)
# ─────────────────────────────────────────────────────────────────────────────

def check_memory_addresses():
    section("6 / MEMORY ADDRESS VALIDATION  [static]")
    print("  ℹ️   Checks that defined addresses fall within valid Dolphin\n"
          "       MEM1 (0x80000000–0x81800000) / MEM2 (0x90000000–0x94000000) ranges.\n"
          "       No Dolphin connection required.\n")

    try:
        import config.memory_addresses as ma

        # Addresses to validate — matches what validate_addresses() uses internally
        addresses = {
            "PLAYER_CURRENT_HP":       ma.PLAYER_CURRENT_HP,
            "PLAYER_RECOVERABLE_HP":   ma.PLAYER_RECOVERABLE_HP,
            "PLAYER_CURRENT_STAMINA":  ma.PLAYER_CURRENT_STAMINA,
            "PLAYER_STAMINA_MAX":      ma.PLAYER_STAMINA_MAX,
            "TIME_SPENT_UNDERWATER":   ma.TIME_SPENT_UNDERWATER,
            "DAMAGE_RECEIVE_LAST_HIT": ma.DAMAGE_RECEIVE_LAST_HIT,
            "PLAYER_X":                ma.PLAYER_X,
            "PLAYER_Y":                ma.PLAYER_Y,
            "PLAYER_Z":                ma.PLAYER_Z,
            "CURRENT_ZONE":            ma.CURRENT_ZONE,
            "SHARPNESS":               ma.SHARPNESS,
            "QUEST_TIME_SPENT":        ma.QUEST_TIME_SPENT,
        }

        MEM1 = (0x80000000, 0x81800000)
        MEM2 = (0x90000000, 0x94000000)

        for name, addr in addresses.items():
            in_mem1 = MEM1[0] <= addr < MEM1[1]
            in_mem2 = MEM2[0] <= addr < MEM2[1]
            if in_mem1 or in_mem2:
                region = "MEM1" if in_mem1 else "MEM2"
                ok(f"{name:<30s}  0x{addr:08X}  [{region}]")
            else:
                fail(f"{name:<30s}  0x{addr:08X}  [INVALID — outside DME ranges]")

        # Warn about None addresses (still to find in-game)
        none_attrs = [
            attr for attr in dir(ma)
            if not attr.startswith("_")
            and not callable(getattr(ma, attr))
            and getattr(ma, attr) is None
        ]
        if none_attrs:
            print()
            for attr in none_attrs:
                warn(f"{attr} = None  (address not yet found — expected)")

    except ImportError as e:
        fail(f"config.memory_addresses import failed: {e}")
    except AttributeError as e:
        fail(f"Missing expected constant in memory_addresses.py: {e}")
    except Exception as e:
        fail(f"Memory address validation error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. DOLPHIN CONNECTION & LIVE MEMORY READ  (--live)
# ─────────────────────────────────────────────────────────────────────────────

def check_dolphin_live():
    section("7 / DOLPHIN CONNECTION & LIVE MEMORY READ  [--live]")
    print("  ℹ️   Dolphin must be running with Monster Hunter Tri loaded.\n"
          "       Best run from inside a quest (not the village).\n")

    # 7a — Dolphin process running?
    try:
        import psutil
        dolphin_procs = [
            p for p in psutil.process_iter(["name", "pid"])
            if p.info["name"] and "dolphin" in p.info["name"].lower()
        ]
        if dolphin_procs:
            for p in dolphin_procs:
                ok(f"Dolphin process: {p.info['name']}  (PID {p.info['pid']})")
        else:
            fail("No Dolphin process found — launch Dolphin and load the game first")
            return
    except Exception as e:
        fail(f"Cannot list processes: {e}")
        return

    # 7b — Hook DME
    dme = None
    try:
        import dolphin_memory_engine as dme
        dme.hook()
        if dme.is_hooked():
            ok("dolphin_memory_engine hooked successfully")
        else:
            fail("dolphin_memory_engine hook failed — is the game running past the title screen?")
            return
    except Exception as e:
        fail(f"dolphin_memory_engine hook error: {e}")
        return

    # 7c — Live reads using known addresses from memory_addresses.py
    # PLAYER_CURRENT_HP  = 0x9014AEAF  float  MEM2
    # PLAYER_X           = 0x900AD764  float  MEM2
    # PLAYER_STAMINA_MAX = 0x806C02A8  float  MEM1
    # CURRENT_ZONE       = 0x806BAC64  byte   MEM1
    live_reads = [
        ("PLAYER_CURRENT_HP",  0x9014AEAF, "float"),
        ("PLAYER_X",           0x900AD764, "float"),
        ("PLAYER_STAMINA_MAX", 0x806C02A8, "float"),
        ("CURRENT_ZONE",       0x806BAC64, "byte"),
    ]
    for label, addr, dtype in live_reads:
        try:
            if dtype == "float":
                val = dme.read_float(addr)
                ok(f"[{label}] @ 0x{addr:08X} = {val:.4f}")
            else:
                val = dme.read_byte(addr)
                ok(f"[{label}] @ 0x{addr:08X} = {val}")
        except Exception as e:
            warn(f"[{label}] read failed (may not be in-quest): {e}")

    # 7d — DynamicMemoryReader import
    try:
        from core.dynamic_memory_reader import DynamicMemoryReader
        ok("DynamicMemoryReader class importable")
    except Exception as e:
        fail(f"DynamicMemoryReader import failed: {e}")

    # Unhook cleanly
    try:
        dme.un_hook()
        ok("dolphin_memory_engine unhooked cleanly")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 8. FRAME CAPTURE  (--live)
# ─────────────────────────────────────────────────────────────────────────────

def check_frame_capture():
    section("8 / FRAME CAPTURE  [--live]")
    print("  ℹ️   Dolphin window must be visible (not minimized).\n"
          "       Window title must start with 'MHTri' or contain 'Monster Hunter'.\n")

    # 8a — win32gui window scan
    try:
        import win32gui
        windows = []

        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                t = win32gui.GetWindowText(hwnd)
                tl = t.lower()
                if tl.startswith("mhtri") or "monster hunter" in tl:
                    windows.append((hwnd, t))
            return True

        win32gui.EnumWindows(_cb, None)
        if windows:
            for hwnd, title in windows:
                ok(f"Window found: '{title}'  (hwnd={hwnd})")
        else:
            fail("No Monster Hunter Tri window found — window title must start with 'MHTri' "
                 "or contain 'Monster Hunter'")
            return
    except Exception as e:
        fail(f"win32gui scan failed: {e}")
        return

    # 8b — FrameCapture instantiation
    # __init__ calls find_window() → raises ValueError if no game window
    # use_dll=False avoids requiring DolphinCapture.dll for this diagnostic
    capturer = None
    try:
        from vision.frame_capture import FrameCapture
        capturer = FrameCapture(use_dll=False)
        ok(f"FrameCapture instantiated  (hwnd={capturer.hwnd})")
    except ValueError as e:
        fail(f"FrameCapture could not find window: {e}")
        return
    except Exception as e:
        fail(f"FrameCapture instantiation error: {e}")
        return

    # 8c — Capture a frame via capture_frame()
    try:
        import numpy as np
        frame = capturer.capture_frame()
        if frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3:
            ok(f"capture_frame() OK — shape={frame.shape}  dtype={frame.dtype}")
            mean_val = float(frame.mean())
            if mean_val < 1.0:
                warn(f"Frame is all black (mean={mean_val:.1f}) — is Dolphin minimized "
                     "or the game on a black screen?")
            else:
                ok(f"Frame content valid  (mean pixel: {mean_val:.1f})")
        else:
            warn(f"capture_frame() returned unexpected result: {type(frame)}")
    except Exception as e:
        fail(f"capture_frame() error: {e}")
    finally:
        try:
            capturer.close()
        except Exception:
            pass

    # 8d — FramePreprocessor (pure import + dummy tensor, no Dolphin needed)
    try:
        from vision.preprocessing import FramePreprocessor
        import numpy as np
        preprocessor = FramePreprocessor()
        ok("FramePreprocessor instantiated")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        result = preprocessor.preprocess(dummy)
        ok(f"FramePreprocessor.preprocess() OK — output shape={result.shape}")
    except Exception as e:
        fail(f"FramePreprocessor test failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. VIRTUAL CONTROLLER  (--controller)
# ─────────────────────────────────────────────────────────────────────────────

def check_controller():
    section("9 / VIRTUAL CONTROLLER  [--controller]")
    print("  ℹ️   ViGEmBus driver must be installed and Windows restarted.\n"
          "       Download: https://github.com/nefarius/ViGEmBus/releases\n")

    # 9a — vgamepad import
    try:
        import vgamepad as vg
        ok("vgamepad importable")
    except ImportError as e:
        fail(f"vgamepad not installed: {e}")
        fail("→ pip install vgamepad")
        return

    # 9b — WiiController in gamepad mode
    # Constructor: WiiController(debug=False, use_controller=True, instance_id=0)
    # is_connected = True if ViGEmBus is available and controller was created
    try:
        from core.controller import WiiController

        controller = WiiController(debug=False, use_controller=True, instance_id=0)

        if not controller.is_connected:
            fail("WiiController.is_connected = False — ViGEmBus not available")
            fail("→ Install ViGEmBusSetup_x64.msi as Administrator then restart Windows")
            return

        ok("WiiController created (gamepad mode, ViGEmBus OK)")

        # 9c — Basic action tests
        controller.reset_all()
        ok("reset_all() OK")

        controller.execute_action(action_id=0, frames=1)   # no-op
        ok("execute_action(0) OK  [no-op / 1 frame]")

        controller.execute_action(action_id=1, frames=1)   # move forward 1 frame
        ok("execute_action(1) OK  [forward / 1 frame]")

        controller.execute_action(action_id=9, frames=1)   # attack1 / X button
        ok("execute_action(9) OK  [attack1 / X / 1 frame]")

        controller.cleanup()
        ok("WiiController.cleanup() OK")

    except ImportError as e:
        fail(f"core.controller import failed: {e}")
        return
    except Exception as e:
        fail(f"WiiController (gamepad mode) error: {e}")
        return

    # 9d — Keyboard fallback (debug=True = no real keys sent, safe anywhere)
    print()
    try:
        from core.controller import WiiController
        kb_ctrl = WiiController(debug=True, use_controller=False, instance_id=0)
        ok("WiiController keyboard fallback (debug=True) instantiated")
        kb_ctrl.cleanup()
    except Exception as e:
        warn(f"WiiController keyboard fallback test error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{'═' * 60}")
    print(f"  SUMMARY")
    print(f"{'═' * 60}")
    print(f"  ✅  {_results['ok']:>3}  passed")
    print(f"  ⚠️   {_results['warn']:>3}  warnings")
    print(f"  ❌  {_results['fail']:>3}  failed")
    print(f"{'─' * 60}")

    if _results["fail"] == 0 and _results["warn"] == 0:
        print("  🎉  Everything looks good! Ready to train.")
    elif _results["fail"] == 0:
        print("  🟡  Setup OK with minor warnings. Training should work.")
    else:
        print("  🔴  Fix the ❌ errors above before launching training.")

    print(f"{'═' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Monster Hunter Tri IA — Setup Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_setup.py               Static checks only (no Dolphin needed)
  python check_setup.py --live        + Dolphin memory hook + frame capture
  python check_setup.py --controller  + Virtual controller / ViGEmBus test
  python check_setup.py --full        All checks
        """
    )
    parser.add_argument("--live",       action="store_true",
                        help="Test Dolphin connection, memory reads, and frame capture "
                             "(requires Dolphin running with game loaded)")
    parser.add_argument("--controller", action="store_true",
                        help="Test virtual controller creation (requires ViGEmBus)")
    parser.add_argument("--full",       action="store_true",
                        help="Run all checks (equivalent to --live --controller)")

    args = parser.parse_args()
    if args.full:
        args.live = True
        args.controller = True

    print(f"\n{'═' * 60}")
    print(f"  Monster Hunter Tri IA — Setup Diagnostic")
    print(f"{'═' * 60}")
    if args.live or args.controller:
        flags = []
        if args.live:       flags.append("--live")
        if args.controller: flags.append("--controller")
        print(f"  Mode: {' '.join(flags)}")
    else:
        print("  Mode: static  (use --live / --controller / --full for more)")

    check_system()
    check_structure()
    check_dependencies()
    check_gpu()
    check_modules()
    check_memory_addresses()   # Static — no Dolphin needed

    if args.live:
        check_dolphin_live()
        check_frame_capture()

    if args.controller:
        check_controller()

    print_summary()


if __name__ == "__main__":
    main()
