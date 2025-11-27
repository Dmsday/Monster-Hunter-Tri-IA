"""
Training script v3.0
"""

# ============================================================================
# STANDARD PYTHON IMPORTS
# ============================================================================
import os                      # File/folder handling (models/, logs/)
import signal                  # Add signal handling for robust Ctrl+C
import sys                     # System (exit, args, platform check)
import time                    # Time and pauses (sleep, countdown)
import numpy as np
import argparse                # CLI arguments (--timesteps, --resume, etc.)
import traceback               # Detailed error display
import threading               # Threads (non-blocking input() with GUI)
import logging                 # Logging (for reconnecting handlers)
import subprocess              # Launching Dolphin process via PowerShell
from datetime import datetime  # Timestamps for experiment names
import atexit                  # Emergency cleanup on script exit
import json                    # Config persistence

# ============================================================================
# VISUALIZATION (for CNN testing/exploration)
# ============================================================================
import matplotlib.pyplot as plt  # Debug plots (minimap, crop, etc.)

# ============================================================================
# TKINTER (GRAPHICAL INTERFACE)
# ============================================================================
import tkinter as tk           # GUI Stop button control

# ============================================================================
# GYMNASIUM & STABLE-BASELINES3 (RL)
# ============================================================================
from gymnasium import spaces   # Action/observation spaces (Discrete, Box, Dict)

from stable_baselines3 import PPO  # Proximal Policy Optimization algorithm

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
# - BaseCallback: Base class for custom callbacks
# - CheckpointCallback: Automatic saving every N steps

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# - DummyVecEnv: Wrapper to use 1 env as vectorized
# - VecNormalize: Observation/reward normalization

# ============================================================================
# CUSTOM MODULES
# ============================================================================
# --- Logging ---
from utils.module_logger import set_global_log_level, get_module_logger
from utils.advanced_logging import TrainingLogger, LoggingCallback
# - TrainingLogger: Advanced logging system (errors, checkpoints, config)
# - LoggingCallback: SB3 callback to log during training

# It will automatically connect to console.log once TrainingLogger is created
logger = get_module_logger('train')

# --- Monster Hunter Environment ---
from environment.mh_env import MonsterHunterEnv
# Custom Gymnasium environment (vision + memory + rewards)

# --- AI Agent (PPO with CNN) ---
from agent.ppo_agent import create_ppo_agent
# Factory function to create PPO agent with CNN architecture

# --- Training GUI ---
from utils.training_gui import TrainingGUI
# Real-time GUI (stats, graphs, 3D exploration map)

# Multi-agent import
try:
    from utils.multi_agent_scheduler import MultiAgentScheduler
    from utils.multi_agent_trainer import MultiAgentTrainer
    from utils.genetic_trainer import GeneticTrainer
    MULTI_AGENT_AVAILABLE = True
except ImportError as multi_agent_import_error:
    MultiAgentScheduler = None
    MultiAgentTrainer = None
    GeneticTrainer = None
    MULTI_AGENT_AVAILABLE = False
    logger.warning(f"Multi-agent not available: {multi_agent_import_error}")

# HidHide import
try:
    from utils.hidhide_manager import HidHideManager, is_admin
    HIDHIDE_AVAILABLE = True
except ImportError as hidhide_import_error:
    HidHideManager = None
    is_admin = None
    HIDHIDE_AVAILABLE = False
    logger.warning(f"HidHide not available: {hidhide_import_error}")

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================
logger.debug("🔍 Checking dependencies...")

try:
    from vision.preprocessing import FramePreprocessor
    logger.debug("  FramePreprocessor")
except Exception as import_preprocessing_error:
    logger.error(f"FramePreprocessor: {import_preprocessing_error}")

try:
    from utils.training_gui import TrainingGUI
    logger.debug("  TrainingGUI")
except Exception as import_training_gui_error:
    logger.error(f"TrainingGUI: {import_training_gui_error}")

# ============================================================================
# ACTIONS INFORMATION
# ============================================================================
logger.debug(f"🎮 Action configuration:")
logger.debug(f"Total: 19 possible actions (0-18)")


# ============================================================================
# MULTI-INSTANCE FUNCTIONS
# ============================================================================
# Global flag to prevent double cleanup (atexit + finally)
_cleanup_done = False

def launch_dolphin_instances_via_powershell(
        num_instances: int,
        dolphin_path: str,
        minimize_dolphin: bool = True,
        minimize_game: bool = False
) -> bool:
    """
    Launches Dolphin instances via the PowerShell script

    Args:
        num_instances: Number of instances to launch
        dolphin_path: Path to Dolphin.exe
        minimize_dolphin: Minimize Dolphin menu windows
        minimize_game: Minimize game windows

    Returns:
        True if successful, False otherwise
    """
    # Define script_dir early (for cwd parameter)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create temp directory for PID files FIRST (before calling PowerShell)
    # Store in vision/ folder to keep project root clean
    vision_dir = os.path.join(script_dir, "vision")
    temp_dir = os.path.join(vision_dir, "temp")

    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Created temp directory in vision/: {temp_dir}")
    except Exception as temp_create_error:
        logger.error(f"Failed to create temp directory: {temp_create_error}")
        logger.error(f"Path: {temp_dir}")
        return False

    # Verify temp directory was actually created
    if not os.path.exists(temp_dir):
        logger.error(f"temp directory does not exist after creation: {temp_dir}")
        return False

    logger.debug(f"Temp directory confirmed: {temp_dir}")

    # Create .gitignore inside temp/ and debug/ to exclude all files
    # This prevents accidentally committing temporary PID files to Git
    debug_dir = os.path.join(".", "vision", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_gitignore = os.path.join(debug_dir, ".gitignore")
    if not os.path.exists(debug_gitignore):
        try:
            with open(debug_gitignore, 'w') as f:
                f.write("# Ignore all debug visualization images\n")
                f.write("*.png\n")
                f.write("*.jpg\n")
                f.write("# Except this .gitignore itself\n")
                f.write("!.gitignore\n")
            logger.debug("Created .gitignore in vision/debug/")
        except Exception as gitignore_error:
            logger.debug(f"Could not create debug .gitignore: {gitignore_error}")

    logger.debug(f"Temp directory for PIDs: {temp_dir}")
    # This prevents accidentally committing temporary PID files to Git
    gitignore_path = os.path.join(temp_dir, ".gitignore")
    if not os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, 'w') as f:
                f.write("# Ignore all files in temp/ directory\n")
                f.write("*\n")
                f.write("# Except this .gitignore itself\n")
                f.write("!.gitignore\n")
            logger.debug("Created .gitignore in temp/ directory")
        except Exception as gitignore_error:
            logger.debug(f"Could not create .gitignore: {gitignore_error}")

    logger.debug(f"Temp directory for PIDs: {temp_dir}")

    # Normalize and validate dolphin_path
    dolphin_path = os.path.abspath(dolphin_path)

    # Handle both folder and .exe paths
    if os.path.isdir(dolphin_path):
        # User provided folder path, append Dolphin.exe
        dolphin_exe = os.path.join(dolphin_path, "Dolphin.exe")
        logger.debug(f"Folder path provided, looking for: {dolphin_exe}")

        if os.path.isfile(dolphin_exe):
            dolphin_path = dolphin_exe
        else:
            logger.error("=" * 70)
            logger.error("DOLPHIN.EXE NOT FOUND IN FOLDER")
            logger.error("=" * 70)
            logger.error(f"Folder provided: {dolphin_path}")
            logger.error(f"Expected file: {dolphin_exe}")
            logger.error(f"File exists: {os.path.exists(dolphin_exe)}")
            logger.error("")
            logger.error("SOLUTION:")
            logger.error("  Verify the folder contains Dolphin.exe")
            logger.error("=" * 70)
            return False

    # Validate the .exe file
    if not os.path.isfile(dolphin_path):
        logger.error("=" * 70)
        logger.error("DOLPHIN.EXE NOT FOUND")
        logger.error("=" * 70)
        logger.error(f"Provided path: {dolphin_path}")
        logger.error(f"Path exists: {os.path.exists(dolphin_path)}")
        logger.error(f"Is directory: {os.path.isdir(dolphin_path)}")
        logger.error("")
        logger.error("SOLUTION:")
        logger.error("Use --dolphin-path with EITHER:")
        logger.error("  1. Full path to Dolphin.exe:")
        logger.error(f"     --dolphin-path 'C:\\Path\\To\\Dolphin.exe'")
        logger.error("  2. Or folder containing Dolphin.exe:")
        logger.error(f"     --dolphin-path 'C:\\Path\\To\\Dolphin-Folder'")
        logger.error("=" * 70)
        return False

    logger.debug(f"Dolphin.exe validated: {dolphin_path}")

    # calculate dolphin_dir from validated path
    dolphin_dir = os.path.dirname(dolphin_path)

    # Search for PowerShell script in proper order
    # Priority 1: Same folder as Dolphin.exe
    ps_script = os.path.join(dolphin_dir, "launch_dolphin_instances.ps1")
    search_locations = [
        ("Dolphin folder", ps_script),
        ("train.py folder", os.path.join(script_dir, "launch_dolphin_instances.ps1")),
        ("Current directory", os.path.abspath("launch_dolphin_instances.ps1"))
    ]

    found = False
    for location_name, location_path in search_locations:
        if os.path.isfile(location_path):
            ps_script = location_path
            logger.debug(f"PowerShell script found in {location_name}: {ps_script}")
            found = True
            break

    if not found:
        logger.error("=" * 70)
        logger.error("POWERSHELL SCRIPT NOT FOUND")
        logger.error("=" * 70)
        logger.error("Searched in:")
        for i, (name, path) in enumerate(search_locations, 1):
            logger.error(f"  {i}. {name}: {path}")
            logger.error(f"     Exists: {os.path.exists(path)}")
        logger.error("")
        logger.error("SOLUTIONS:")
        logger.error("1. Copy launch_dolphin_instances.ps1 to Dolphin folder:")
        logger.error(f"   Target: {dolphin_dir}")
        logger.error("2. Or copy it to train.py folder:")
        logger.error(f"   Target: {script_dir}")
        logger.error("=" * 70)
        return False

    # Normalize paths to avoid issues
    ps_script = os.path.normpath(ps_script)
    dolphin_path = os.path.normpath(dolphin_path)
    script_dir = os.path.normpath(script_dir)

    # Validate script exists (final check)
    if not os.path.isfile(ps_script):
        logger.error("=" * 70)
        logger.error("CRITICAL: PowerShell script not accessible")
        logger.error("=" * 70)
        logger.error(f"Path: {ps_script}")
        logger.error(f"Exists: {os.path.exists(ps_script)}")
        logger.error(f"Is file: {os.path.isfile(ps_script)}")
        logger.error("")
        logger.error("SOLUTION:")
        logger.error("  Verify file permissions and antivirus settings")
        logger.error("=" * 70)
        return False

    # PowerShell arguments
    ps_args = [
        "powershell.exe",
        "-ExecutionPolicy", "Bypass",
        "-NoProfile",
        "-File", ps_script,
        "-NumInstances", str(num_instances),
        "-NoGUI",
        "-DolphinExePath", dolphin_path,
        "-PidDirectory", temp_dir  # Pass temp directory to PowerShell
    ]

    if minimize_dolphin:
        ps_args.append("-MinimizeDolphin")

    if minimize_game:
        ps_args.append("-MinimizeGame")

    # Compute timeout
    dynamic_timeout = 10 + (num_instances * 10)

    logger.debug("=" * 70)
    logger.debug("LAUNCHING DOLPHIN INSTANCES VIA POWERSHELL")
    logger.debug("=" * 70)
    logger.debug(f"PowerShell script: {ps_script}")
    logger.debug(f"Instances number: {num_instances}")
    logger.debug(f"Dolphin.exe: {dolphin_path}")
    logger.debug(f"Dolphin folder: {dolphin_dir}")
    logger.debug(f"Timeout: {dynamic_timeout}s")
    logger.debug(f"Working directory: {script_dir}")
    logger.debug("")

    # Show actual command
    logger.debug("Full command:")
    logger.debug(f"  {' '.join(ps_args)}")
    logger.debug("")

    # Use Popen instead of run to handle KeyboardInterrupt properly
    # noinspection PyUnusedLocal
    ps_process = None

    try:
        logger.debug("Starting PowerShell...")
        logger.debug("Waiting for script completion...")
        logger.debug("")

        # Launch PowerShell with Popen (keeps process reference)
        ps_process = subprocess.Popen(
            ps_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=script_dir,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        # Wait with timeout and capture output
        stdout, stderr = ps_process.communicate(timeout=dynamic_timeout)

        # Check return code manually
        if ps_process.returncode != 0:
            raise subprocess.CalledProcessError(
                ps_process.returncode,
                ps_args,
                output=stdout,
                stderr=stderr
            )

        # Create result object compatible with old code
        result = type('obj', (object,), {
            'stdout': stdout,
            'stderr': stderr,
            'returncode': ps_process.returncode
        })()

        logger.debug("PowerShell completed successfully")

        # Display stdout (PowerShell logs)
        if result.stdout:
            logger.debug("")
            logger.debug("PowerShell output:")
            logger.debug("-" * 70)
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.debug(f"{line}")
            logger.debug("-" * 70)

        # Active waiting for PID files with retry
        logger.debug("")
        logger.debug("Waiting for PID files to be created...")

        max_wait = 10  # Max 10 seconds
        check_interval = 0.5  # Check every 500ms
        elapsed = 0

        # PID files are in temp/ subdirectory
        expected_pid_files = [
            os.path.join(temp_dir, f"dolphin_pid_{i}.tmp")
            for i in range(num_instances)
        ]

        while elapsed < max_wait:
            # Check if all PID files exist
            existing_files = [f for f in expected_pid_files if os.path.exists(f)]

            if len(existing_files) == num_instances:
                logger.debug(f"All {num_instances} PID files found after {elapsed:.1f}s")
                break

            # Wait a bit
            time.sleep(check_interval)
            elapsed += check_interval

            # Log progress every 2 seconds
            if int(elapsed) % 2 == 0 and elapsed > 0:
                logger.debug(f"  Found {len(existing_files)}/{num_instances} PID files ({elapsed:.0f}s elapsed)...")

        # Final check
        existing_files = [f for f in expected_pid_files if os.path.exists(f)]
        if len(existing_files) < num_instances:
            logger.warning(f"Only {len(existing_files)}/{num_instances} PID files found after {max_wait}s")
            logger.warning("Some instances may not have started correctly")

            # DEBUG: Check if PIDs were created in script root instead of temp/
            logger.debug("")
            logger.debug("Checking if PIDs were created in script root...")
            script_root_pids = []
            for i in range(num_instances):
                root_pid_file = os.path.join(script_dir, f"dolphin_pid_{i}.tmp")
                if os.path.exists(root_pid_file):
                    script_root_pids.append(root_pid_file)
                    logger.warning(f"  Found PID in root: {root_pid_file}")

            if script_root_pids:
                logger.warning("PIDs were created in script root instead of temp/")
                logger.warning("This means PowerShell did not receive -PidDirectory parameter")
                logger.warning("Check PowerShell script and parameter passing")

        # Extra 1s buffer to ensure files are fully written
        time.sleep(1.0)

        return True

    # Handle Ctrl+C during PowerShell execution
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning("INTERRUPTED DURING DOLPHIN LAUNCH (Ctrl+C)")
        logger.warning("=" * 70)

        # Terminate PowerShell process if still running
        if ps_process and ps_process.poll() is None:
            logger.warning("Terminating PowerShell process...")
            ps_process.terminate()
            try:
                ps_process.wait(timeout=2)
                logger.debug("PowerShell terminated")
            except subprocess.TimeoutExpired:
                logger.warning("PowerShell did not terminate, forcing kill...")
                ps_process.kill()
                logger.debug("PowerShell killed")

        # Try to read any PIDs that were created before interruption
        logger.warning("Checking for Dolphin instances that were launched...")
        script_dir_local = os.path.dirname(os.path.abspath(__file__))
        temp_dir_local = os.path.join(script_dir_local, "temp")
        launched_pids = []

        for i in range(num_instances):
            pid_file = os.path.join(temp_dir_local, f"dolphin_pid_{i}.tmp")
            try:
                if os.path.exists(pid_file):
                    with open(pid_file, 'r') as f:
                        pid_str = f.read().strip()
                        if pid_str and pid_str != "-1":
                            pid = int(pid_str)
                            if pid > 0:
                                launched_pids.append(pid)
                                logger.debug(f"Found Dolphin PID {pid} from file")
                    # Clean up PID file
                    os.remove(pid_file)
                    logger.debug(f"Removed PID file: {pid_file}")
            except Exception as pid_read_error:
                logger.debug(f"Could not read PID file {pid_file}: {pid_read_error}")

        if launched_pids:
            logger.warning(f"Found {len(launched_pids)} Dolphin instance(s) to close...")

            # Store in global for signal handler
            global _global_dolphin_pids
            _global_dolphin_pids = launched_pids

            cleanup_dolphin_processes(launched_pids, emergency=True)
            logger.info("Dolphin instances cleanup completed")
        else:
            logger.warning("No Dolphin PIDs found")
            logger.warning("If Dolphin windows are open, close them manually")

        logger.warning("=" * 70)
        logger.warning("")

        # Re-raise KeyboardInterrupt to propagate to main() handler
        raise

    except subprocess.TimeoutExpired:
        logger.error(f"PowerShell timeout ({dynamic_timeout}s exceeded)")
        logger.error("Dolphin instances are taking too long to load")
        logger.error("SOLUTIONS:")
        logger.error("  1. Increase --dolphin-timeout")
        logger.error("  2. Launch fewer instances")
        logger.error("  3. Close other applications")
        return False

    except subprocess.CalledProcessError as ps_error:
        logger.error(f"PowerShell error (code {ps_error.returncode})")
        if ps_error.stdout:
            logger.error("Stdout:")
            for line in ps_error.stdout.split('\n'):
                if line.strip():
                    logger.error(f"  {line}")
        if ps_error.stderr:
            logger.error("Stderr:")
            for line in ps_error.stderr.split('\n'):
                if line.strip():
                    logger.error(f"  {line}")
        logger.error("")
        logger.error("COMMON CAUSES:")
        logger.error("  1. PowerShell execution policy blocked")
        logger.error("  2. Dolphin.exe path incorrect")
        logger.error("  3. ROM file not found")
        logger.error("  4. User folder path incorrect")
        return False

    except Exception as ps_error:
        logger.error(f"Unexpected error: {ps_error}")
        traceback.print_exc()
        return False

def auto_detect_or_prompt_dolphin_path() -> str:
    """
    Auto-detect Dolphin path or prompt user

    Search priority:
    1. Common installation paths
    2. Current directory
    3. Ask user via console input

    Returns:
        Valid path to Dolphin.exe or folder containing it

    Raises:
        SystemExit: If user cancels input (Ctrl+C)

    Note:
        This function ALWAYS returns a valid path or exits.
        It never returns None.
    """
    logger.info("Auto-detecting Dolphin path...")

    # Common paths to check
    common_paths = [
        # Portable installations
        "./Dolphin-x64",
        "./Dolphin",
        "../Dolphin-x64",
        "../Dolphin",

        # User Documents
        os.path.join(os.path.expanduser("~"), "Documents", "Dolphin-x64"),
        os.path.join(os.path.expanduser("~"), "Documents", "Dolphin"),

        # Program Files
        "C:/Program Files/Dolphin-x64",
        "C:/Program Files (x86)/Dolphin-x64",

        # Desktop
        os.path.join(os.path.expanduser("~"), "Desktop", "Dolphin-x64"),
        os.path.join(os.path.expanduser("~"), "Desktop", "Dolphin"),
    ]

    # Check each path
    for path in common_paths:
        if os.path.exists(path):
            # Check if it's a folder
            if os.path.isdir(path):
                dolphin_exe = os.path.join(path, "Dolphin.exe")
                if os.path.isfile(dolphin_exe):
                    logger.info(f"Found Dolphin at: {path}")
                    return path
            # Check if it's Dolphin.exe directly
            elif os.path.isfile(path) and path.endswith("Dolphin.exe"):
                logger.info(f"Found Dolphin.exe at: {path}")
                return os.path.dirname(path)

    # Not found - ask user
    logger.warning("Dolphin path not found automatically")
    logger.warning("")
    logger.warning("Please provide the path to Dolphin:")
    logger.warning("  - Either the folder containing Dolphin.exe")
    logger.warning("  - Or the full path to Dolphin.exe")
    logger.warning("")

    while True:
        try:
            user_path = input("Dolphin path: ").strip().strip('"').strip("'")

            if not user_path:
                logger.error("Empty path provided")
                continue

            # Normalize path
            user_path = os.path.abspath(user_path)

            # Check if valid
            if os.path.isdir(user_path):
                dolphin_exe = os.path.join(user_path, "Dolphin.exe")
                if os.path.isfile(dolphin_exe):
                    logger.info(f"Valid Dolphin folder: {user_path}")
                    return user_path
                else:
                    logger.error(f"Folder does not contain Dolphin.exe: {user_path}")
            elif os.path.isfile(user_path) and user_path.endswith("Dolphin.exe"):
                logger.info(f"Valid Dolphin.exe: {user_path}")
                return os.path.dirname(user_path)
            else:
                logger.error(f"Invalid path: {user_path}")

        except KeyboardInterrupt:
            logger.warning("")
            logger.warning("CANCELLED BY USER (Ctrl+C)")
            logger.warning("Dolphin path prompt cancelled")
            logger.warning("Exiting...")
            sys.exit(0)

        except Exception as prompt_error:
            logger.error(f"Error: {prompt_error}")
            continue

    # Unreachable line but satisfies PyCharm
    # This line will NEVER be executed because while True + sys.exit guarantees a return
    # But it removes the PyCharm warning "Missing return statement"
    raise RuntimeError("Unreachable code ; function always returns or exits")

def cleanup_dolphin_processes(dolphin_pids: list, emergency: bool = False):
    """
    Cleanup: forcefully close all Dolphin instances
    Called automatically on script exit (normal or crash)

    Args:
        dolphin_pids: List of Dolphin process IDs to terminate
        emergency: True if called during crash/interrupt, False for normal cleanup
    """
    # Declare global at the very beginning of function
    global _cleanup_done

    if not dolphin_pids:
        logger.debug("No Dolphin PIDs to cleanup")
        return

    cleanup_type = "EMERGENCY CLEANUP" if emergency else "CLEANUP"
    logger.warning("=" * 70)
    logger.warning(f"{cleanup_type}: Closing all Dolphin instances")
    logger.warning("=" * 70)

    import psutil

    closed_count = 0
    failed_count = 0

    for pid in dolphin_pids:
        if pid is None or pid < 0:
            continue

        try:
            # Check if process exists
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                process_name = process.name()

                logger.info(f"Terminating PID {pid} ({process_name})...")

                # Try graceful termination first
                process.terminate()

                # Wait up to 3 seconds for graceful shutdown
                try:
                    process.wait(timeout=3)
                    logger.info(f"  PID {pid} terminated gracefully")
                    closed_count += 1
                except psutil.TimeoutExpired:
                    # Force kill if still running
                    logger.warning(f"  PID {pid} did not respond, forcing kill...")
                    process.kill()
                    process.wait(timeout=2)
                    logger.info(f"  PID {pid} force killed")
                    closed_count += 1
            else:
                logger.debug(f"PID {pid} already closed")

        except psutil.NoSuchProcess:
            logger.debug(f"PID {pid} no longer exists")
        except psutil.AccessDenied:
            logger.error(f"Access denied to PID {pid} (requires admin rights)")
            failed_count += 1
        except Exception as cleanup_error:
            logger.error(f"Failed to close PID {pid}: {cleanup_error}")
            failed_count += 1

    logger.warning(f"Cleanup completed: {closed_count} closed, {failed_count} failed")
    logger.warning("=" * 70)


# ============================================================================
# GLOBAL SIGNAL HANDLER FOR ROBUST CLEANUP

_global_dolphin_pids = []  # Track PIDs globally for signal handler
_global_cleanup_done = False


def emergency_signal_handler(signum, _frame):
    """
    Global signal handler for SIGINT (Ctrl+C) and SIGTERM
    Ensures Dolphin instances are closed even if caught in blocking operations

    Args:
        signum: Signal number
        _frame: Stack frame (unused but required by signal handler signature)
    """
    global _global_cleanup_done

    if _global_cleanup_done:
        logger.debug("Cleanup already done, ignoring signal")
        return

    logger.warning("")
    logger.warning("=" * 70)
    logger.warning(f"SIGNAL RECEIVED: {signal.Signals(signum).name}")
    logger.warning("=" * 70)

    # Close Dolphin instances immediately
    if _global_dolphin_pids:
        logger.warning(f"Emergency cleanup: closing {len(_global_dolphin_pids)} Dolphin instance(s)...")
        try:
            cleanup_dolphin_processes(_global_dolphin_pids, emergency=True)
            logger.info("Dolphin instances closed")
        except Exception as cleanup_error:
            logger.error(f"Error during emergency cleanup: {cleanup_error}")

    _global_cleanup_done = True
    logger.warning("=" * 70)

    # Exit immediately
    sys.exit(0)
# ============================================================================

def wait_for_dolphin_windows(
        num_instances: int,
        timeout: int = 60,
        check_interval: int = 10
) -> bool:
    """
    Waits until the Dolphin windows are detected

    Args:
        num_instances: Number of expected instances
        timeout: Total timeout in seconds
        check_interval: Interval between checks (seconds)

    Returns:
        True if all windows are detected
    """
    import win32gui

    logger.debug("")
    logger.debug("=" * 70)
    logger.debug("DETECTING DOLPHIN WINDOWS")
    logger.debug("=" * 70)
    logger.debug(f"Expected instances : {num_instances}")
    logger.debug(f"Timeout            : {timeout}s")
    logger.debug(f"Check interval     : {check_interval}s")
    logger.debug("")

    start_time = time.time()
    attempt = 0
    # Don't initialize windows here anymore --> it will be created in the loop

    while time.time() - start_time < timeout:
        attempt += 1
        elapsed = int(time.time() - start_time)

        logger.debug(f"Attempt {attempt} ({elapsed}s elapsed)...")

        # Detect MHTri windows
        def callback(hwnd, wins):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                title_lower = title.lower()

                # Look for MHTri-X or Monster Hunter Tri
                if title_lower.startswith("mhtri") or "monster hunter" in title_lower:
                    wins.append({
                        'hwnd': hwnd,
                        'title': title
                    })
            return True

        # Create fresh list for each detection attempt
        windows = []
        win32gui.EnumWindows(callback, windows)

        # Sort by title for consistent ordering
        windows.sort(key=lambda x: x['title'])

        logger.debug(f"Detected windows : {len(windows)}/{num_instances}")

        if windows:
            for i, win in enumerate(windows):
                logger.debug(f"      [{i}] {win['title']}")

        # Check if we have all windows
        if len(windows) >= num_instances:
            logger.debug("")
            logger.debug("ALL WINDOWS DETECTED!")
            logger.debug("=" * 70)
            logger.debug("")
            return True

        # Wait before next check
        if time.time() - start_time < timeout:
            logger.debug(f"Waiting {check_interval}s before next check...")
            logger.debug("")
            time.sleep(check_interval)
        else:
            # Timeout exceeded - show what we found
            logger.error("")
            logger.error("TIMEOUT: Not all windows were detected")
            logger.error(f"   Expected : {num_instances}")
            logger.error(f"   Found    : {len(windows)}")

            # DEBUG : Show which windows we detected
            if windows:
                logger.error("Detected windows:")
                for i, win in enumerate(windows):
                    logger.error(f"[{i}] {win['title']}")

    return False

def calculate_agent_allocation(
        num_agents: int,
        num_instances: int,
        allocation_mode: str = 'auto',
        allocation_map: str = None,
        multi_agent_mode: str = 'independent',
) -> dict:
    """
    Calcule la répartition agents/instances selon le scénario

    Args:
        num_agents: Nombre d'agents PPO
        num_instances: Nombre d'instances Dolphin
        allocation_mode: Mode de répartition (auto, manual, weighted)
        allocation_map: Mapping manuel (format: "0:1,2;1:3,4")
        multi_agent_mode: Mode de partage (independent, round_robin, majority_vote)

    Returns:
        Dict avec la répartition calculée
    """
    logger.debug("")
    logger.debug("=" * 70)
    logger.debug("📊 CALCUL DE LA RÉPARTITION AGENTS/INSTANCES")
    logger.debug("=" * 70)
    logger.debug(f"Agents    : {num_agents}")
    logger.debug(f"Instances : {num_instances}")
    logger.debug(f"Mode      : {allocation_mode}")
    logger.debug("")

    # Détecter scénario
    if num_agents == num_instances:
        scenario = "ONE_TO_ONE"
        logger.debug("🎯 SCÉNARIO 1 : One-to-One (1 agent = 1 instance)")

        # Allocation fixe
        allocation = {i: [i] for i in range(num_agents)}

    elif num_agents < num_instances:
        scenario = "AGENT_MULTIPLE_INSTANCES"
        logger.debug("🎯 SCÉNARIO 2 : Agent avec Instances Multiples")
        logger.debug(f"   → Chaque agent contrôle plusieurs instances")

        if allocation_mode == 'manual' and allocation_map:
            # Parse allocation_map
            allocation = parse_allocation_map(allocation_map, num_agents, num_instances)
        else:
            # Auto : répartition équitable
            allocation = {}
            instances_per_agent = num_instances // num_agents
            remainder = num_instances % num_agents

            current_instance = 0
            for agent_id in range(num_agents):
                # Les premiers agents reçoivent +1 instance
                count = instances_per_agent + (1 if agent_id < remainder else 0)
                allocation[agent_id] = list(range(current_instance, current_instance + count))
                current_instance += count

    else:  # num_agents > num_instances
        scenario = "INSTANCE_SHARING"
        logger.debug("🎯 SCÉNARIO 3 : Partage d'Instances")
        logger.debug(f"   → Plusieurs agents partagent les instances")
        logger.debug(f"   → Mode de gestion : {multi_agent_mode}")

        if allocation_mode == 'manual' and allocation_map:
            allocation = parse_allocation_map(allocation_map, num_agents, num_instances)
        else:
            # Auto : répartition équitable
            allocation = {}
            agents_per_instance = num_agents // num_instances
            remainder = num_agents % num_instances

            current_agent = 0
            for instance_id in range(num_instances):
                # Les premières instances reçoivent +1 agent
                count = agents_per_instance + (1 if instance_id < remainder else 0)

                # Assigner ces agents à cette instance
                for _ in range(count):
                    if current_agent < num_agents:
                        allocation[current_agent] = [instance_id]
                        current_agent += 1

        # Vérifier qu'au moins 1 instance a plusieurs agents
        instances_usage = {}
        for agent_id, instances in allocation.items():
            for inst in instances:
                if inst not in instances_usage:
                    instances_usage[inst] = []
                instances_usage[inst].append(agent_id)

        has_sharing = any(len(agents) > 1 for agents in instances_usage.values())

        if not has_sharing:
            logger.error("ERREUR : Allocation SCÉNARIO 3 sans partage détecté!")
            logger.error(f"Agents par instance : {instances_usage}")
            logger.error("Au moins 1 instance doit avoir plusieurs agents")
            raise ValueError("Allocation SCÉNARIO 3 invalide : aucun partage")

    # Afficher répartition
    logger.debug("")
    logger.debug("📋 RÉPARTITION CALCULÉE :")
    logger.debug("-" * 70)

    for agent_id, instances in sorted(allocation.items()):
        instances_str = ", ".join(map(str, instances))
        logger.debug(f"   Agent {agent_id:2d} → Instances [{instances_str}]")

    logger.debug("-" * 70)
    logger.debug("")

    # Statistiques
    total_connections = sum(len(instances) for instances in allocation.values())
    logger.debug("📊 STATISTIQUES :")
    logger.debug(f"   Connexions totales : {total_connections}")
    logger.debug(f"   Moyenne par agent  : {total_connections / num_agents:.1f}")

    if scenario == "AGENT_MULTIPLE_INSTANCES":
        logger.debug(f"   Instances par agent : {[len(v) for v in allocation.values()]}")
    elif scenario == "INSTANCE_SHARING":
        # Compter agents par instance
        agents_per_inst = {}
        for agent_id, instances in allocation.items():
            for inst in instances:
                agents_per_inst[inst] = agents_per_inst.get(inst, 0) + 1
        logger.debug(f"   Agents par instance : {list(agents_per_inst.values())}")

    logger.debug("=" * 70)
    logger.debug("")

    return {
        'scenario': scenario,
        'allocation': allocation,
        'num_agents': num_agents,
        'num_instances': num_instances
    }

def detect_scenario(num_agents: int, num_instances: int) -> str:
    """
    Détecte automatiquement le scénario selon spécification
    """
    if num_agents == num_instances:
        return "ONE_TO_ONE"
    elif num_agents < num_instances:
        return "AGENT_MULTIPLE_INSTANCES"
    else:  # num_agents > num_instances
        return "INSTANCE_SHARING"

def parse_allocation_map(allocation_map: str, num_agents: int, num_instances: int) -> dict:
    """
    Parse l'allocation manuelle avec validation complète
    Format: "0:1,2;1:3,4" -> {0: [1,2], 1: [3,4]}

    SCÉNARIO 2 : Chaque instance ne peut être assignée qu'une fois
    SCÉNARIO 3 : Une instance peut être assignée plusieurs fois
    """
    allocation = {}
    instances_used = []  # Pour tracking SCÉNARIO 2

    try:
        # Séparer par ";"
        pairs = allocation_map.split(';')

        for pair in pairs:
            if ':' not in pair:
                raise ValueError(f"Format invalide : {pair} (attendu: 'agent:instances')")

            agent_str, instances_str = pair.split(':', 1)
            agent_id = int(agent_str.strip())

            # Parser instances (séparées par virgules)
            instances = [int(x.strip()) for x in instances_str.split(',')]

            # Ajouter ou fusionner
            if agent_id in allocation:
                # Agent déjà défini, fusionner instances
                allocation[agent_id].extend(instances)
            else:
                allocation[agent_id] = instances

            # Tracker instances SCÉNARIO 2
            instances_used.extend(instances)

        # === VALIDATION ===

        # 1. Tous les agents définis
        expected_agents = set(range(num_agents))
        actual_agents = set(allocation.keys())

        if expected_agents != actual_agents:
            missing = expected_agents - actual_agents
            extra = actual_agents - expected_agents

            error_msg = []
            if missing:
                error_msg.append(f"Agents manquants : {sorted(missing)}")
            if extra:
                error_msg.append(f"Agents invalides : {sorted(extra)}")

            raise ValueError('\n'.join(error_msg))

        # 2. Range instances valide
        for agent_id, instances in allocation.items():
            for inst in instances:
                if not (0 <= inst < num_instances):
                    raise ValueError(
                        f"Agent {agent_id} : Instance {inst} invalide "
                        f"(range: 0-{num_instances - 1})"
                    )

        # 3. Détection scénario et validation spécifique
        if num_agents == num_instances:
            scenario = "ONE_TO_ONE"
        elif num_agents < num_instances:
            scenario = "AGENT_MULTIPLE_INSTANCES"
        else:
            scenario = "INSTANCE_SHARING"

        if scenario == "AGENT_MULTIPLE_INSTANCES":
            # SCÉNARIO 2 : Pas de doublon d'instance
            unique_instances = set(instances_used)

            if len(unique_instances) != len(instances_used):
                # Trouver doublons
                from collections import Counter
                counts = Counter(instances_used)
                duplicates = [inst for inst, count in counts.items() if count > 1]

                raise ValueError(
                    f"SCÉNARIO 2 : Doublons d'instances détectés : {duplicates}\n"
                    "Chaque instance ne peut être assignée qu'une seule fois"
                )

            # Toutes les instances assignées
            if unique_instances != set(range(num_instances)):
                missing = set(range(num_instances)) - unique_instances
                raise ValueError(
                    f"SCÉNARIO 2 : Instances non assignées : {sorted(missing)}\n"
                    "Toutes les instances doivent être assignées"
                )

        elif scenario == "INSTANCE_SHARING":
            # SCÉNARIO 3 : Au moins 1 instance avec plusieurs agents
            instance_usage = {}
            for agent_id, instances in allocation.items():
                for inst in instances:
                    if inst not in instance_usage:
                        instance_usage[inst] = []
                    instance_usage[inst].append(agent_id)

            shared_instances = [inst for inst, agents in instance_usage.items()
                                if len(agents) > 1]

            if not shared_instances:
                raise ValueError(
                    "SCÉNARIO 3 : Aucune instance partagée détectée\n"
                    f"Utilisation actuelle : {instance_usage}\n"
                    "Au moins 1 instance doit avoir plusieurs agents"
                )

        return allocation

    except ValueError as validation_error:
        logger.error(f"Validation allocation_map échouée : {validation_error}")
        raise
    except Exception as parse_error:
        raise ValueError(f"Erreur parsing allocation_map : {parse_error}")


def validate_genetic_params(args_to_validate):
    """
    Valide les paramètres génétiques selon spécification
    """
    if args_to_validate.multi_agent_mode != 'genetic':
        return True

    errors = []

    # genetic_generations >= 1
    if args_to_validate.genetic_generations < 1:
        errors.append(f"genetic_generations doit être >= 1 (reçu: {args_to_validate.genetic_generations})")

    # 0.0 < genetic_elite_ratio < 1.0
    if not (0.0 < args_to_validate.genetic_elite_ratio < 1.0):
        errors.append(f"genetic_elite_ratio doit être entre 0 et 1 (reçu: {args_to_validate.genetic_elite_ratio})")

    # 0.0 <= genetic_mutation_rate <= 1.0
    if not (0.0 <= args_to_validate.genetic_mutation_rate <= 1.0):
        errors.append(f"genetic_mutation_rate doit être entre 0 et 1 (reçu: {args_to_validate.genetic_mutation_rate})")

    if errors:
        logger.error("ERREURS VALIDATION PARAMÈTRES GÉNÉTIQUES")
        logger.error("=" * 70)
        for error in errors:
            logger.error(f"   • {error}")
        logger.error("=" * 70)
        raise ValueError("\n".join(errors))

    return True


def validate_round_robin_params(args_to_validate):
    """
    Valide les paramètres round_robin
    """
    if args_to_validate.multi_agent_mode != 'round_robin':
        return True

    if args_to_validate.block_size < 1:
        error_msg = f"block_size doit être >= 1 (reçu: {args_to_validate.block_size})"
        logger.error(f"{error_msg}")
        raise ValueError(error_msg)

    return True

class ConsoleMessageManager:
    """
    Gestionnaire de messages console pour éviter le spam
    """

    def __init__(self):
        self.last_messages = {}
        self.message_counts = {}

    def print_grouped(self, key: str, message: str, update_same_line: bool = True):
        """
        Affiche un message en groupant les répétitions

        Args:
            key: Clé unique pour identifier le type de message
            message: Message à afficher
            update_same_line: Si True, écrase la ligne précédente
        """
        if key not in self.message_counts:
            self.message_counts[key] = 0

        self.message_counts[key] += 1
        count = self.message_counts[key]

        if update_same_line and count > 1:
            # Écraser la ligne précédente
            formatted = f"\r{message} x{count}"
            print(formatted, end='', flush=True)
        else:
            # Nouvelle ligne
            if count > 1:
                logger.debug(f"{message} x{count}")
            else:
                logger.debug(f"{message}")

        self.last_messages[key] = message

    def reset(self, key: str = None):
        """
        Reset un compteur spécifique ou tous
        """
        if key:
            self.message_counts[key] = 0
        else:
            self.message_counts.clear()
            self.last_messages.clear()


class GUIUpdateCallback(BaseCallback):
    """
    Callback pour mettre à jour la GUI + console propre
    """

    def __init__(self, gui: TrainingGUI, verbose=0):
        super().__init__(verbose)
        self.gui = gui
        self.episode_count = 0

        # Gestionnaire console
        self.console = ConsoleMessageManager()

        # Tracker l'item sélectionné
        self.last_item_selected = 24  # Aucun item au départ
        self.last_item_selected_name = "Aucun"

    def _on_step(self) -> bool:
        # Extraire infos
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
        else:
            info = {}

        # DEBUG : Logger épisodes avec reward explosive
        current_reward = self.locals.get('rewards', [0])[0]

        if abs(current_reward) > 50:  # Seuil d'alerte
            logger.warning(f"⚠️ REWARD ANORMALE DÉTECTÉE : {current_reward:.2f}")
            logger.warning(f"Episode: {info.get('episode_num', '?')}, Step: {info.get('episode_steps', '?')}")

            # Afficher breakdown pour diagnostic
            breakdown = info.get('reward_breakdown', {})
            if breakdown:
                logger.warning("Breakdown:")
                for cat, val in sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True):
                    if abs(val) > 1.0:
                        logger.warning(f"  {cat}: {val:+.2f}")

            # Afficher état du jeu
            logger.warning(f"État:")
            logger.warning(f"  HP: {info.get('hp', '?')}, Stamina: {info.get('stamina', '?')}")
            logger.warning(f"  Zone: {info.get('current_zone', '?')}, Deaths: {info.get('death_count', '?')}")
            logger.warning(f"  Position: ({info.get('player_x', 0):.0f}, {info.get('player_y', 0):.0f}, {info.get('player_z', 0):.0f})")

        # Vérifier si écran de fin détecté
        if info.get('quest_ended_screen') or info.get('quest_ended_after_action'):
            logger.info(f"FIN DE QUÊTE DÉTECTÉE")
            logger.info(f"   Épisode: {info.get('episode_num', '?')}")
            logger.info(f"   Steps: {info.get('episode_steps', '?')}")
            logger.info(f"   Raison: {info.get('forced_reset_reason', 'unknown')}")
            logger.info(f"   Current map: {info.get('current_map', '?')}")
            logger.info(f"Reset automatique effectué")

            # Incrémenter compteur d'épisodes
            self.episode_count += 1

            # Reset console counters
            self.console.reset()

            # Continuer normalement (pas d'arrêt)
            return True

        # Si flag quest_ended dans reward calc
        if info.get('quest_ended_in_reward_calc') or info.get('quest_ended_flag_in_calc'):
            logger.debug(f"Flag quest_ended détecté dans reward_calc")
            logger.debug(f"Épisode va se terminer proprement")
            return True

        # Stats étendues
        player_x = info.get('player_x', 0.0) or 0.0
        player_y = info.get('player_y', 0.0) or 0.0
        player_z = info.get('player_z', 0.0) or 0.0
        orientation = info.get('orientation', 0.0) or 0.0
        current_zone = info.get('current_zone', 0) or 0
        money = info.get('money', 0) or 0
        inventory = info.get('inventory', [])

        #  Item sélectionné avec nom
        item_selected = info.get('item_selected', 24) or 24
        item_selected_name = "Aucun"

        if item_selected != 24 and inventory:
            # Chercher l'item correspondant dans l'inventaire
            for item in inventory:
                if item.get('slot') == item_selected + 1:  # +1 car index 0-23 slot 1-24
                    item_selected_name = item.get('name', f"Item ID {item.get('item_id')}")
                    break

        # Reward breakdown
        reward_breakdown = info.get('reward_breakdown', {})
        reward_breakdown_detailed = info.get('reward_breakdown_detailed', {})

        # AJOUTER à l'historique
        if reward_breakdown:
            self.gui.reward_breakdown_history.append(reward_breakdown)

        if reward_breakdown_detailed:
            self.gui.reward_breakdown_detailed_history.append(reward_breakdown_detailed)

        # Stats exploration tracker complètes
        total_distance = info.get('total_distance', 0.0) or 0.0
        sharpness = info.get('sharpness', 100) or 100
        quest_time = info.get('quest_time', 0) or 0

        # Exploration tracker stats
        total_cubes = info.get('total_cubes', 0) or 0
        zones_discovered = info.get('zones_discovered', 0) or 0
        exploration_visits = info.get('exploration_visits', 0) or 0
        left_monster_zone_count = info.get('left_monster_zone_count', 0) or 0

        # Récupérer les cubes pour la carte 3D
        exploration_cubes = {}
        try:
            # Essayer de récupérer le reward_calc depuis l'env
            if hasattr(self.model, 'env') and hasattr(self.model.env, 'get_attr'):
                reward_calc = self.model.env.get_attr('reward_calc')[0]
                if reward_calc and hasattr(reward_calc, 'exploration_tracker'):
                    tracker = reward_calc.exploration_tracker

                    # Synchroniser les marqueurs AVANT de les envoyer au GUI
                    tracker.sync_all_markers_to_cubes()

                    # Récupérer cubes par zone
                    for zone_id, cubes_list in tracker.cubes_by_zone.items():
                        exploration_cubes[zone_id] = [
                            {
                                'center_x': cube.center_x,
                                'center_y': cube.center_y,
                                'center_z': cube.center_z,
                                'size_x': cube.size_x,
                                'size_y': cube.size_y,
                                'size_z': cube.size_z,
                                'size': cube.size,  # Garder pour compatibilité
                                'avg_size': (cube.size_x + cube.size_y + cube.size_z) / 3.0,
                                'visit_count': cube.visit_count,
                                'total_visits': cube.total_visits,
                                'zone_id': cube.zone_id,
                                'blocked_directions': cube.blocked_directions,
                                'markers': cube.markers,
                            }
                            for cube in cubes_list
                        ]
        except (AttributeError, KeyError, IndexError):
            pass  # Silencieux si échec d'accès aux attributs

        # Stats zone actuelle
        monsters_present = info.get('in_monster_zone', False)
        monster_count = info.get('monster_count', 0) or 0
        in_combat = info.get('in_combat', False)
        in_monster_zone = info.get('in_monster_zone', False)

        # HP monstres
        smonster1_hp = info.get('smonster1_hp', 0) or 0
        smonster2_hp = info.get('smonster2_hp', 0) or 0
        smonster3_hp = info.get('smonster3_hp', 0) or 0
        smonster4_hp = info.get('smonster4_hp', 0) or 0
        smonster5_hp = info.get('smonster5_hp', 0) or 0

        # Reward zone (calculer depuis breakdown)
        zone_reward = (reward_breakdown.get('monster_zone', 0) +
                       reward_breakdown.get('curiosity', 0))

        # Compteurs
        episode_num = info.get('episode_num', self.episode_count)
        episode_steps = info.get('episode_steps', 0)
        total_steps = info.get('total_steps', self.num_timesteps)

        # Assurer que episode_steps est bien un int
        if episode_steps is None:
            episode_steps = 0
        elif isinstance(episode_steps, dict):
            # Si c'est un dict, essayer d'extraire une valeur
            episode_steps = episode_steps.get('current', 0)

        # Conversion finale
        try:
            episode_steps = int(episode_steps)
        except (ValueError, TypeError):
            episode_steps = 0

        # Vérifier que c'est positif
        episode_steps = max(0, episode_steps)

        # Afficher l'item selectionne lorsque changement
        if item_selected != self.last_item_selected:
            if item_selected == 24:
                logger.info(f"Item désélectionné")
            else:
                logger.info(f"Item sélectionné: Slot {item_selected + 1} - {item_selected_name}")

            self.last_item_selected = item_selected
            self.last_item_selected_name = item_selected_name

        # Mettre à jour GUI
        self.gui.update_stats({
            'episode': episode_num,
            'step': episode_steps,
            'total_steps': total_steps,
            'reward': self.locals.get('rewards', [0])[0],
            'hp': info.get('hp', 100),
            'stamina': info.get('stamina', 100),
            'hits': info.get('hit_count', 0),
            'deaths': info.get('death_count', 0),
            'zone': current_zone,
            'action': self.locals.get('actions', [0])[0],
            'player_x': player_x,
            'player_y': player_y,
            'player_z': player_z,
            'orientation': orientation,
            'money': money,
            'distance': total_distance,
            'sharpness': sharpness,
            'quest_time': quest_time,
            'reward_breakdown': reward_breakdown,
            'reward_breakdown_detailed': reward_breakdown_detailed,
            'inventory': inventory,
            'total_cubes': total_cubes,
            'zones_discovered': zones_discovered,
            'exploration_visits': exploration_visits,
            'left_monster_zone_count': left_monster_zone_count,
            'monsters_present': monsters_present,
            'monster_count': monster_count,
            'in_monster_zone': in_monster_zone,
            'in_combat': in_combat,
            'smonster1_hp': smonster1_hp,
            'smonster2_hp': smonster2_hp,
            'smonster3_hp': smonster3_hp,
            'smonster4_hp': smonster4_hp,
            'smonster5_hp': smonster5_hp,
            'zone_reward_total': zone_reward,
            'exploration_cubes': exploration_cubes,
            'item_selected': item_selected,
            'item_selected_name': item_selected_name,
            'in_game_menu': info.get('in_game_menu', False),
            'game_menu_open_count': info.get('game_menu_open_count', 0),
            'game_menu_total_time': info.get('game_menu_total_time', 0.0),
        })

        # CONSOLE PROPRE : Messages groupés
        # Nombre de cubes en zone
        if info.get('zone_changed'):
            zone = info.get('current_zone', 0)
            total_cubes = info.get('total_cubes', 0)
            cubes_in_zone = 0

            # Récupérer le nombre de cubes DANS CETTE ZONE
            try:
                if hasattr(self.model, 'env') and hasattr(self.model.env, 'get_attr'):
                    reward_calc = self.model.env.get_attr('reward_calc')[0]
                    if reward_calc and hasattr(reward_calc, 'exploration_tracker'):
                        tracker = reward_calc.exploration_tracker
                        cubes_in_zone = len(tracker.cubes_by_zone.get(zone, []))

                logger.debug(f"Zone {zone} : {cubes_in_zone} cubes actifs (total global: {total_cubes})")

            except (AttributeError, KeyError, IndexError):
                logger.debug(f"Zone {zone} : {total_cubes} cubes actifs")

        # Fusion de cubes
        if info.get('compression_triggered'):
            logger.debug(f"FUSION DE CUBES déclenchée")
            logger.debug(f"Total cubes après compression: {info.get('total_cubes', 0)}")

        # Détection fin d'épisode
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            # Récupérer les vraies valeurs depuis info['episode'] (format SB3)
            episode_info = info.get('episode', {})

            if episode_info:
                current_episode_num = self.episode_count

                # Utiliser les valeurs de l'env (FIABLES)
                final_reward = float(episode_info.get('r', 0.0))
                final_length = int(episode_info.get('l', 0))
                final_hits = info.get('hit_count', 0)

                logger.info(f"📊 Épisode {current_episode_num} terminé:")
                logger.info(f"  Reward: {final_reward:.2f}")
                logger.info(f"  Length: {final_length} steps")
                logger.info(f"  Hits: {final_hits}")
                logger.info(f"  Zone finale: {current_zone}")

                # Ajouter à l'historique
                self.gui.add_episode_data(
                    episode=current_episode_num,
                    reward=final_reward,
                    length=final_length,
                    hits=final_hits
                )
            else:
                current_episode_num = self.episode_count
                # Fallback si info['episode'] absent (ne devrait jamais arriver)
                logger.warning("info['episode'] manquant - utilisation valeurs par défaut")
                self.gui.add_episode_data(
                    episode=current_episode_num,
                    reward=0.0,
                    length=0,
                    hits=0
                )

            self.episode_count += 1

            # Reset console counters
            self.console.reset()

        # Log périodique des stats d'épisode en cours (tous les 100 steps)
        if self.n_calls % 300 == 0:
            current_episode_reward = info.get('episode_reward', 0.0)
            logger.debug(f"Step {self.n_calls} | Ep {episode_num} | Reward accumulée: {current_episode_reward:.2f}")

        # Vérifier arrêt
        if self.gui.should_stop():
            logger.warning("Arrêt demandé via l'interface")
            return False

        return True

def load_model_if_resume(args, env, device, load_model_logger: TrainingLogger = None):
    """
    Charge un modèle existant si --resume
    """
    if args.resume:
        if not os.path.exists(args.resume):
            error_msg = f"Modèle introuvable : {args.resume}"
            logger.error(f"Modèle introuvable : {args.resume}")
            if load_model_logger:
                load_model_logger.log_error(FileNotFoundError(error_msg), context="Chargement modèle")
            logger.error(f"{error_msg}")
            sys.exit(1)

        logger.debug(f"📦 Chargement du modèle : {args.resume}")
        time.sleep(2.0)

        try:
            #Charger le modèle
            agent = PPO.load(args.resume, env=env, device=device)
            logger.debug(f"Modèle chargé avec succès")
            logger.debug(f"Timesteps précédents : {agent.num_timesteps:,}")

            # Note : VecNormalize est géré séparément dans la création d'environnement

            return agent, agent.num_timesteps
        except (FileNotFoundError, RuntimeError, ValueError) as load_error:
            # LOGGER L'ERREUR
            if load_model_logger:
                load_model_logger.log_error(load_error, context="Chargement checkpoint")
            logger.error(f"Erreur chargement : {load_error}")
            sys.exit(1)

    return None, 0


def main():
    """
    Script d'entraînement principal v2
    """
    global _cleanup_done

    # ============================================================
    # ARGUMENTS AVEC AIDE DÉTAILLÉE
    # ============================================================

    parser = argparse.ArgumentParser(
        description='🎮 Entraîner une IA pour Monster Hunter Tri avec PPO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
    Exemples d'utilisation :

      # Nouvel entraînement basique (clavier)
      python train.py --timesteps 100000

      # Avec manette virtuelle et grayscale
      python train.py --controller --grayscale --timesteps 50000

      # Reprendre un entraînement
      python train.py --resume ./models/mon_exp/checkpoint_50000_steps.zip

      # Reprendre mais forcer nouveau VecNormalize (après changement d'observation)
      python train.py --resume ./models/mon_exp/final_model.zip --force-new-vecnormalize

      # Test rapide en debug
      python train.py --debug-steps 1000 --small-rollout --no-gui
      
      # Multi-agents mode weighted (allocation adaptative)
      python train.py --num-agents 8 --num-instances 4 --allocation-mode weighted

      # Multi-agents mode genetic (évolution)
      python train.py --num-agents 16 --num-instances 8 --multi-agent-mode genetic --genetic-generations 10

    Pour plus d'informations : https://github.com/Dmsday/Monster-Hunter-Tri-IA
            '''
    )

    # ============================================================================
    # GROUPE 1 : ENTRAÎNEMENT
    # ============================================================================
    training_group = parser.add_argument_group('Entraînement')

    training_group.add_argument('--timesteps', type=int, default=100000,
                                metavar='N',
                                help='Nombre de timesteps (défaut: 100000)')

    training_group.add_argument('--name', type=str, default=None,
                                metavar='NOM',
                                help='Nom de l\'expérience (auto-généré si absent)')

    training_group.add_argument('--resume', type=str, default=None,
                                metavar='CHEMIN',
                                help='Reprendre depuis un checkpoint (.zip)')

    training_group.add_argument('--force-new-vecnormalize', action='store_true',
                                help='Forcer nouveau VecNormalize (ignore .pkl existant)')

    training_group.add_argument('--save-state', type=int, default=5,
                                choices=range(1, 9), metavar='N',
                                help='Numéro save state à charger (1-8, défaut: 5)')

    training_group.add_argument('--lr', type=float, default=1e-4,
                                metavar='LR',
                                help='Learning rate PPO (défaut: 0.0001)')

    training_group.add_argument('--cpu', action='store_true',
                                help='Forcer CPU uniquement (ignore CUDA)')

    # ============================================================================
    # GROUPE 2 : ENVIRONNEMENT
    # ============================================================================
    env_group = parser.add_argument_group('Environnement')

    env_group.add_argument('--env', type=str, default='hybrid',
                           choices=['visual', 'memory', 'hybrid'],
                           help='Mode environnement (défaut: hybrid = vision+mémoire)')

    env_group.add_argument('--keyboard', action='store_true',
                           help='Utiliser clavier (pynput) au lieu de manette virtuelle')

    env_group.add_argument('--controller', action='store_true',
                           help='Utiliser manette virtuelle (vgamepad) - DÉFAUT')

    # ============================================================================
    # GROUPE 3 : VISION
    # ============================================================================
    vision_group = parser.add_argument_group('Vision & Capture')

    vision_group.add_argument('--grayscale', action='store_true',
                              help='Vision en niveaux de gris (réduit dimensions)')

    vision_group.add_argument('--rtvision', action='store_true',
                              help='Afficher fenêtre OpenCV avec vision IA temps réel')

    vision_group.add_argument('--rtminimap', action='store_true',
                              help='Afficher minimap exploration temps réel (nécessite --rtvision)')

    # ============================================================================
    # GROUPE 4 : MULTI-AGENT & MULTI-INSTANCE
    # ============================================================================
    multi_group = parser.add_argument_group('Multi-Agent & Multi-Instance')

    multi_group.add_argument('--num-agents', type=int, default=1,
                             metavar='N',
                             help='Nombre d\'agents PPO (défaut: 1)')

    multi_group.add_argument('--num-instances', type=int, default=1,
                             metavar='N',
                             help='Nombre d\'instances Dolphin (défaut: 1)')

    multi_group.add_argument('--dolphin-path', type=str,
                             default=None,
                             metavar='PATH',
                             help='Chemin vers Dolphin.exe')

    multi_group.add_argument('--allocation-mode', type=str, default='auto',
                             choices=['auto', 'manual', 'weighted'],
                             help='Mode répartition instances/agents (défaut: auto)')

    multi_group.add_argument('--allocation-map', type=str, default=None,
                             metavar='MAP',
                             help='Mapping manuel agents→instances. '
                                  'Format: "agent_id:inst1,inst2;agent_id2:inst3". '
                                  'Exemple: "0:0,1;1:2,3" = Agent 0 contrôle instances 0 et 1')

    multi_group.add_argument('--multi-agent-mode', type=str, default='independent',
                             choices=['independent', 'round_robin', 'majority_vote', 'genetic'],
                             help='Mode gestion multi-agent (défaut: independent)')

    multi_group.add_argument('--steps-per-agent', type=int, default=4096,
                             metavar='N',
                             help='Steps collectés par agent avant update (défaut: 4096)')

    multi_group.add_argument('--genetic-generations', type=int, default=10,
                             help='Nombre de générations (mode genetic)')

    multi_group.add_argument('--genetic-elite-ratio', type=float, default=0.25,
                             help='Ratio d\'élites conservées (mode genetic)')

    multi_group.add_argument('--genetic-mutation-rate', type=float, default=0.3,
                             help='Taux de mutation (mode genetic)')

    multi_group.add_argument('--block-size', type=int, default=100,
                             help='Taille des blocs (mode round_robin)')

    multi_group.add_argument('--weighted-eval-freq', type=int, default=100,
                             metavar='N',
                             help='Fréquence réévaluation (mode weighted)')

    multi_group.add_argument('--dolphin-timeout', type=int, default=60,
                             metavar='SECONDS',
                             help='Timeout for Dolphin window detection (default: 60s)')

    # ============================================================================
    # GROUPE 5 : INTERFACE
    # ============================================================================
    interface_group = parser.add_argument_group('Interface & Visualisation')

    interface_group.add_argument('--no-gui', action='store_true',
                                 help='Désactiver interface graphique')

    # ============================================================================
    # GROUPE 6 : DEBUG & TESTS
    # ============================================================================
    debug_group = parser.add_argument_group('Debug & Tests')

    debug_group.add_argument('--debug-steps', type=int, default=None,
                             metavar='N',
                             help='Nombre de steps pour test rapide (override --timesteps)')

    debug_group.add_argument('--log-memory', action='store_true',
                             help='Logger les vecteurs mémoire périodiquement')

    debug_group.add_argument('--small-rollout', action='store_true',
                             help='Utiliser n_steps=512 (rollouts courts pour debug)')

    debug_group.add_argument('--log-level', type=str, default='WARNING',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                             help='Niveau de log (défaut: WARNING)')

    args = parser.parse_args()

    # Create config/user/ directory and .gitignore
    import os
    config_user_dir = os.path.join(".", "config", "user")
    os.makedirs(config_user_dir, exist_ok=True)

    config_user_gitignore = os.path.join(config_user_dir, ".gitignore")
    if not os.path.exists(config_user_gitignore):
        try:
            with open(config_user_gitignore, 'w') as f:
                f.write("# Ignore all user-specific config files\n")
                f.write("*\n\n")
                f.write("# Except this .gitignore\n")
                f.write("!.gitignore\n")
            logger.debug("Created .gitignore in config/user/")
        except Exception as gitignore_error:
            logger.debug(f"Could not create config/user .gitignore: {gitignore_error}")

    # Auto-detection du nom d'experience depuis --resume
    if args.resume and args.name is None:
        # Extraire nom depuis chemin : ./models/exp_name/checkpoint.zip
        import os
        resume_path = os.path.abspath(args.resume)

        # Remonter au dossier parent du checkpoint
        # Exemple: ./models/mon_exp/checkpoint_1000.zip -> mon_exp
        model_dir = os.path.dirname(resume_path)
        exp_name_from_resume = os.path.basename(model_dir)

        # Verifier que c'est un nom valide (pas "models" ou chemin bizarre)
        if exp_name_from_resume and exp_name_from_resume != 'models':
            args.name = exp_name_from_resume
            logger.debug(f"Nom d'expérience détecté depuis --resume : {args.name}")
        else:
            # Fallback : generer nouveau nom
            logger.warning(f"Impossible de détecter le nom depuis --resume")
            logger.warning(f"Chemin : {args.resume}")

    # ============================================================
    # FONCTIONS MULTI-INSTANCE
    # ============================================================
    def validate_multi_agent_args(multi_agent_args):
        """
        Validates multi-agent/instance arguments according to specification v1.0

        Rules:
        - 1 <= num_agents <= 32
        - 1 <= num_instances <= 16
        - allocation_map valid if mode manual
        - genetic parameters valid if mode genetic
        """

        # Basic validation
        if not (1 <= multi_agent_args.num_agents <= 32):
            raise ValueError(
                f"num_agents must be between 1 and 32 (received: {multi_agent_args.num_agents})\n"
                f"Recommended limit: 16 agents maximum for optimal performance"
            )

        if not (1 <= multi_agent_args.num_instances <= 16):
            raise ValueError(
                f"num_instances must be between 1 and 16 (received: {multi_agent_args.num_instances})\n"
                f"System limit: 16 instances maximum"
            )

        # Validate steps_per_agent
        if hasattr(multi_agent_args, 'steps_per_agent'):
            if multi_agent_args.steps_per_agent < 256:
                logger.warning(f"steps_per_agent very low ({multi_agent_args.steps_per_agent})")
                logger.warning("Recommended: >= 2048 for PPO stability")

        # USE detect_scenario() to determine allocation strategy
        # Renamed variable to avoid shadowing
        current_scenario = detect_scenario(
            multi_agent_args.num_agents,
            multi_agent_args.num_instances
        )

        # Scenario-specific validation
        if current_scenario == "ONE_TO_ONE":
            logger.debug("📊 SCENARIO 1: One-to-One (1 agent = 1 instance)")

            # Ignore certain arguments in this mode
            if multi_agent_args.allocation_mode != 'auto':
                logger.warning(f"allocation_mode ignored in One-to-One mode")
            if multi_agent_args.multi_agent_mode != 'independent':
                logger.warning(f"multi_agent_mode ignored in One-to-One mode")

        elif current_scenario == "AGENT_MULTIPLE_INSTANCES":
            logger.debug("📊 SCENARIO 2: Agent with Multiple Instances")
            logger.debug(f"   {multi_agent_args.num_agents} agents, {multi_agent_args.num_instances} instances")

            # multi_agent_mode not used in this scenario
            if multi_agent_args.multi_agent_mode != 'independent':
                logger.warning(f"multi_agent_mode ignored (each agent has its own instances)")

        else:  # current_scenario == "INSTANCE_SHARING"
            logger.debug("📊 SCENARIO 3: Instance Sharing")
            logger.debug(f"   {multi_agent_args.num_agents} agents, {multi_agent_args.num_instances} instances")
            logger.debug(f"   Sharing mode: {multi_agent_args.multi_agent_mode}")

            # Verify supported mode
            supported_modes = ['independent', 'round_robin', 'majority_vote']
            if multi_agent_args.multi_agent_mode not in supported_modes:
                if multi_agent_args.multi_agent_mode == 'genetic':
                    logger.error("Mode 'genetic' not yet implemented")
                    logger.error("   Available modes: independent, round_robin, majority_vote")
                    raise NotImplementedError("Genetic mode not implemented")
                else:
                    raise ValueError(f"Unknown mode: {multi_agent_args.multi_agent_mode}")

        # Validate allocation_map if manual mode
        if multi_agent_args.allocation_mode == 'manual':
            if multi_agent_args.allocation_map is None:
                raise ValueError("allocation_map required in manual mode")

        # Validate genetic mode parameters
        if multi_agent_args.multi_agent_mode == 'genetic':
            if multi_agent_args.genetic_generations < 1:
                raise ValueError("genetic_generations must be >= 1")
            if not (0.0 < multi_agent_args.genetic_elite_ratio < 1.0):
                raise ValueError("genetic_elite_ratio must be between 0 and 1")
            if not (0.0 <= multi_agent_args.genetic_mutation_rate <= 1.0):
                raise ValueError("genetic_mutation_rate must be between 0 and 1")

        # Validate block_size for round_robin
        if multi_agent_args.multi_agent_mode == 'round_robin':
            if multi_agent_args.block_size < 1:
                raise ValueError("block_size must be >= 1")

        return current_scenario

    def validate_weighted_params(args_to_validate):
        """
        Valide les paramètres weighted
        """
        if args_to_validate.allocation_mode != 'weighted':
            return True

        # Aucun paramètre obligatoire pour weighted, mais vérifier cohérence
        if args_to_validate.num_agents == 1:
            logger.warning("Mode weighted inutile avec 1 seul agent")

        return True

    # Autocorrect : if only --num-instances provided, set --num-agents to match
    if args.num_instances > 1 and args.num_agents == 1:
        logger.warning(f"--num-instances {args.num_instances} provided without --num-agents")
        logger.warning(f"Auto-setting --num-agents to {args.num_instances} (ONE_TO_ONE scenario)")
        args.num_agents = args.num_instances

    # Appeler validation
    scenario = validate_multi_agent_args(args)
    validate_genetic_params(args)
    validate_round_robin_params(args)
    validate_weighted_params(args)

    # INITIALISER allocation_result ICI (juste après validation)
    # Pour éviter "might be referenced before assignment"
    allocation_result = None  # Sera défini dans la création d'environnement

    def generate_example_allocation_map(num_agents: int, num_instances: int) -> str:
        """
        Génère un exemple d'allocation_map valide
        """
        if num_agents <= num_instances:
            # SCÉNARIO 1 ou 2 : chaque agent a des instances différentes
            instances_per_agent = num_instances // num_agents
            remainder = num_instances % num_agents

            mapping = []
            current_instance = 0

            for agent_identification in range(num_agents):
                count = instances_per_agent + (1 if agent_identification < remainder else 0)
                example_instances = list(range(current_instance, current_instance + count))
                mapping.append(f"{agent_identification}:{','.join(map(str, example_instances))}")
                current_instance += count

            return ';'.join(mapping)
        else:
            # SCÉNARIO 3 : agents partagent instances
            agents_per_instance = num_agents // num_instances

            mapping = []
            current_agent = 0

            for instance_id in range(num_instances):
                for _ in range(agents_per_instance):
                    if current_agent < num_agents:
                        mapping.append(f"{current_agent}:{instance_id}")
                        current_agent += 1

            # Restants sur dernière instance
            while current_agent < num_agents:
                mapping.append(f"{current_agent}:{num_instances - 1}")
                current_agent += 1

            return ';'.join(mapping)

    # ============================================================
    # MESSAGE INITIAL D'INFORMATION
    # ============================================================
    logger.info("🚀 LANCEMENT DE L'ENTRAÎNEMENT")

    # Afficher les arguments utilisés
    logger.info("📋 Arguments détectés :")
    logger.info(f"--timesteps : {args.timesteps:,}")
    logger.info(f"--name : {args.name or 'Auto-généré'}")
    logger.info(f"--resume : {args.resume or 'Nouveau'}")
    # Déterminer le mode d'entrée
    use_keyboard_mode = args.keyboard  # True si --keyboard fourni
    input_mode = "Clavier (pynput)" if use_keyboard_mode else "Manette virtuelle (vgamepad) - DÉFAUT"

    logger.info(f"--keyboard : {'Oui' if args.keyboard else 'Non'}")
    logger.info(f"--controller : {'Oui (explicite)' if args.controller else 'Auto'}")
    logger.info(f"Mode entrée : {input_mode}")
    logger.info(f"--grayscale : {'Oui, images analysées en niveau de gris' if args.grayscale else 'Non, images analysées en couleur'}")
    logger.info(f"--no-gui : {'Oui' if args.no_gui else 'Non'}")
    logger.info(f"--lr : {args.lr}")
    logger.info(f"--cpu : {'Oui' if args.cpu else 'Auto'}")
    logger.info(f"--env : {args.env}")
    env_mode_display = {
        'visual': 'Vision seule (CNN uniquement)',
        'memory': 'Mémoire seule (pas de vision)',
        'hybrid': 'Hybride (Vision + Mémoire)'
    }
    logger.info(f"Mode environnement : {env_mode_display[args.env]}")

    if args.debug_steps:
        logger.info(f"MODE DEBUG : --debug-steps {args.debug_steps}")
    if args.small_rollout:
        logger.info(f"MODE DEBUG : --small-rollout activé")

    logger.warning(f"💡 Pour voir tous les arguments disponibles :")
    logger.warning(f"python train.py --help")

    logger.info("" + "=" * 70)

    # Pause avant de continuer
    time.sleep(1.5)

    # ============================================================
    # RESTE DU CODE (nom expérience, logger, etc.)
    # ============================================================

    # Nom experience
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"mh_{timestamp}"
        logger.debug(f"Nouveau nom d'expérience auto-genere : {args.name}")
    else:
        # Nom fourni ou detecte depuis --resume
        logger.debug(f"📝 Nom d'expérience : {args.name}")

    # Configurer niveau global pour tous les modules
    set_global_log_level(args.log_level)
    logger.debug(f"Niveau de log global configuré : {args.log_level}")

    # Create logger(s) - one per agent in multi-agent mode
    # Initialize training_loggers to None for all cases
    training_loggers = None

    if args.num_agents > 1:
        # Multi-agent: create separate logger per agent
        training_loggers = []
        for agent_idx in range(args.num_agents):
            agent_logger = TrainingLogger(
                experiment_name=args.name,
                base_dir="./logs",
                console_log_level=args.log_level,
                agent_id=agent_idx  # Separate logs per agent
            )
            training_loggers.append(agent_logger)

        # Use first logger as main logger for compatibility
        training_logger = training_loggers[0]
        logger.info(f"📊 Multi-agent logging: {len(training_loggers)} separate log folders created")
    else:
        # Single-agent: normal logger
        training_logger = TrainingLogger(
            experiment_name=args.name,
            base_dir="./logs",
            console_log_level=args.log_level,
            agent_id=None  # No agent ID for single-agent
        )

    # Forcer la reconnexion de tous les loggers existants à console.log
    # Cela garantit que les loggers créés avant TrainingLogger sont captures
    console_level = getattr(logging, args.log_level.upper())
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith('mh_'):
            existing_logger = logging.getLogger(logger_name)

            # Mettre à jour le niveau du logger
            existing_logger.setLevel(console_level)

            # Mettre à jour le niveau de tous ses handlers
            for handler in existing_logger.handlers:
                handler.setLevel(console_level)

            # Récupérer le handler de advanced_logging
            advanced_handler = logging.getLogger('advanced_console_capture')
            if advanced_handler.handlers:
                for handler in advanced_handler.handlers:
                    if handler not in existing_logger.handlers:
                        existing_logger.addHandler(handler)
                        handler.setLevel(console_level)

    use_controller_mode = not args.keyboard

    # Déterminer use_vision et use_memory selon --env
    use_vision = args.env in ['visual', 'hybrid']
    use_memory = args.env in ['memory', 'hybrid']

    # Determiner configuration environnement pour logging
    env_config = {
        'mode': args.env,  # 'visual', 'memory', ou 'hybrid'
        'use_vision': use_vision,
        'use_memory': use_memory,
        'grayscale': args.grayscale,
        'frame_stack': 4,
        'frame_size': '84x84',
    }

    # Ajouter infos multi-instance si pertinent
    if args.num_instances > 1:
        env_config['multi_instance'] = True
        env_config['num_instances'] = args.num_instances
        env_config['num_agents'] = args.num_agents
    else:
        env_config['multi_instance'] = False

    # Logger la config
    training_logger.log_config({
        'timesteps': args.timesteps,
        'log_level': args.log_level,
        'learning_rate': args.lr,
        'grayscale': args.grayscale,
        'use_controller': use_controller_mode,  # Valeur effective
        'keyboard_mode': args.keyboard,         # Flag explicite
        'device': 'cpu' if args.cpu else 'auto',
        'resume': args.resume,
        'small_rollout': args.small_rollout,
        'debug_steps': args.debug_steps,
        'environment': env_config,
    })

    # Affichage config
    logger.info("🎮 ENTRAÎNEMENT MONSTER HUNTER TRI v2")
    logger.info(f"Expérience : {args.name}")
    logger.info(f"Timesteps : {args.timesteps:,}")

    if args.resume:
        logger.info(f"REPRISE depuis : {args.resume}")

    logger.info(f"Grayscale : {'Oui' if args.grayscale else 'Non'}")
    use_keyboard_mode = args.keyboard
    input_mode_display = "Clavier (pynput)" if use_keyboard_mode else "Manette virtuelle (vgamepad)"
    logger.info(f"Mode entrée : {input_mode_display}")

    if use_keyboard_mode:
        logger.warning("ATTENTION : Le mode clavier envoie les touches directement au système !")
        logger.warning("Assurez-vous que Dolphin est bien la fenêtre active")
    logger.info(f"Interface graphique : {'Non' if args.no_gui else 'Oui'}")

    # Device
    device = 'cpu'
    if not args.cpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"Device : GPU - {torch.cuda.get_device_name(0)}")
            else:
                logger.info(f"Device : CPU")
        except  (ImportError, AttributeError, OSError) as cuda_error:
            logger.info(f"Device : CPU (fallback, erreur : {cuda_error})")
    else:
        logger.info(f"Device : CPU (forcé)")

    # Dossiers
    models_dir = f"./models/{args.name}"
    logs_dir = f"./logs/{args.name}"

    import os
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    logger.info(f"Dossiers :")
    logger.info(f"Modèles : {models_dir}")
    logger.info(f"Logs : {logs_dir}")

    # Interface graphique
    gui = None
    if not args.no_gui:
        logger.info("INTERFACE GRAPHIQUE")

        try:
            gui = TrainingGUI(title=f"MH Training v2 - {args.name}")
            gui.start()
            logger.info("Interface graphique démarrée")
            logger.info("Stats temps réel")
            logger.info("Bouton '⏹️ Arrêter' pour stop propre")
            logger.info("Bouton '📈 Stats Étendues (Player)' pour détails joueur")
            logger.info("Bouton '💰 Reward Breakdown' pour analyse rewards")
        except Exception as gui_error:
            logger.error(f"Impossible de démarrer l'interface : {gui_error}")
            logger.error("Entraînement sans GUI")
            gui = None

    # ============================================================================
    # CRÉATION ENVIRONNEMENT(S) - MULTI-INSTANCE
    # ============================================================================
    logger.info("🎮 CRÉATION ENVIRONNEMENT")

    base_env = None
    env = None

    try:
        logger.info("Création de l'environnement...")

        # Déterminer le mode : clavier si --keyboard, sinon manette
        use_controller_mode = not args.keyboard

        # Déterminer use_vision et use_memory selon --env
        use_vision = args.env in ['visual', 'hybrid']
        use_memory = args.env in ['memory', 'hybrid']

        logger.info(f"Configuration environnement (--env={args.env}):")
        logger.info(f"   Vision : {'activée' if use_vision else 'désactivée'}")
        logger.info(f"   Mémoire : {'activée' if use_memory else 'désactivée'}")

        # ========================================================================
        # CAS 1 : INSTANCE UNIQUE (CODE ACTUEL)
        # ========================================================================
        if args.num_instances == 1:
            logger.info("Mode instance unique")

            # CRÉER L'ENVIRONNEMENT DE BASE (toujours)
            base_env = MonsterHunterEnv(
                use_vision=use_vision,
                use_memory=use_memory,
                grayscale=args.grayscale,
                frame_stack=4,
                use_controller=use_controller_mode,
                controller_debug=False,
                use_advanced_rewards=True,
                save_state_slot=args.save_state,
                rt_vision=args.rtvision and use_vision,
                rt_minimap=args.rtminimap and use_vision and use_memory,
                instance_id=0,
            )

            logger.info(f"Environnement créé")
            if isinstance(base_env.action_space, spaces.Discrete):
                num_actions = base_env.action_space.n
                logger.info(f"   Actions : {num_actions}")

                # VALIDATION
                if num_actions != 19:
                    logger.error(f"Attendu 19 actions, trouvé {num_actions}")
            else:
                logger.info(f"   Actions : {base_env.action_space}")
            logger.info(f"   Observation : {base_env.observation_space}")

            # ====================================================================
            # TEST VISION (SI ACTIVÉE)
            # ====================================================================
            if base_env.use_vision:
                logger.info("🔍 Test de la vision...")
                try:
                    test_frame = base_env.frame_capture.capture_frame()
                    logger.info(f"   Capture frame : {test_frame.shape}")

                    # Test preprocessing
                    processed = base_env.preprocessor.preprocess_frame(test_frame)
                    logger.info(f"   Preprocessing : {processed.shape}")

                    # Test stacking
                    stacked = base_env.preprocessor.process_and_stack(test_frame)
                    logger.info(f"   Frame stacking : {stacked.shape}")

                    # Save crop visualization to vision/debug/
                    debug_dir = os.path.join(".", "vision", "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    crop_viz_path = os.path.join(debug_dir, "crop_verification_training.png")
                    base_env.preprocessor.visualize_crop(
                        test_frame,
                        crop_viz_path
                    )
                    logger.info(f"   Crop visualisation: {crop_viz_path}")

                except Exception as vision_error:
                    training_logger.log_error(vision_error, context="Test vision")
                    logger.error(f"   Erreur test vision : {vision_error}")

            # ====================================================================
            # TEST EXPLORATION MAP (SI VISION + MEMORY)
            # ====================================================================
            if base_env.use_memory and base_env.state_fusion and base_env.memory is not None:
                logger.info("Exploration map test...")
                try:
                    # Simuler position
                    if base_env.memory:
                        test_state = base_env.memory.read_game_state()
                        test_x = test_state.get('player_x', 0.0)
                        test_y = test_state.get('player_y', 0.0)
                        test_z = test_state.get('player_z', 0.0)
                        test_zone = test_state.get('current_zone', 0) or 0

                        # Test création mini-carte
                        test_map = base_env.state_fusion.create_exploration_map_with_channels(
                            (test_x, test_y, test_z),
                            test_zone
                        )
                        logger.info(f"   Mini-carte : {test_map.shape}")

                        # Visualiser (optionnel)
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        axes[0].imshow(test_map[:, :, 0], cmap='viridis')
                        axes[0].set_title("Channel 0: Visites")
                        axes[1].imshow(test_map[:, :, 1], cmap='hot')
                        axes[1].set_title("Channel 1: Position joueur")
                        axes[2].imshow(test_map[:, :, 2], cmap='Blues')
                        axes[2].set_title("Channel 2: Cubes récents")
                        plt.tight_layout()
                        # Save to vision/debug/
                        debug_dir = os.path.join(".", "vision", "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        minimap_path = os.path.join(debug_dir, "minimap_test.png")
                        plt.savefig(minimap_path, dpi=100)
                        logger.info(f"   Mini-carte visualisation: {minimap_path}")
                        plt.close()

                except Exception as map_error:
                    training_logger.log_error(map_error, context="Test mini-carte")
                    logger.error(f"Erreur test mini-carte : {map_error}")

            # WRAPPER EN VECENV (toujours)
            env = DummyVecEnv([lambda: base_env])
            logger.info("Wrappé en DummyVecEnv")

        # ========================================================================
        # CAS 2 : MULTI-INSTANCES (VIA POWERSHELL)
        # ========================================================================
        else:
            logger.info(f"Mode multi-instances : {args.num_instances} instances")

            # Config file for persistence - store in config/ folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(script_dir, "config")
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, "dolphin_path_config.json")

            # Try to load saved Dolphin path
            if args.dolphin_path is None:
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            saved_config = json.load(f)
                            saved_path = saved_config.get('dolphin_path')

                            # Validate saved path still exists
                            if saved_path and os.path.exists(saved_path):
                                logger.info(f"Using saved Dolphin path: {saved_path}")
                                args.dolphin_path = saved_path
                            else:
                                logger.warning("Saved Dolphin path no longer valid")

                    except Exception as config_load_error:
                        logger.warning(f"Could not load saved config: {config_load_error}")

                # If still not found, auto-detect or prompt
                if args.dolphin_path is None:
                    logger.info("Dolphin path not provided, auto-detecting...")
                    args.dolphin_path = auto_detect_or_prompt_dolphin_path()

                    # Save the path for next time in config/user/
                    try:
                        with open(config_file, 'w') as f:
                            json.dump({'dolphin_path': args.dolphin_path}, f, indent=2)
                        logger.info(f"Dolphin path saved to: {config_file}")
                    except Exception as config_save_error:
                        logger.warning(f"Could not save config: {config_save_error}")

            # ====================================================================
            # VALIDATE DOLPHIN PATH BEFORE LAUNCH
            # ====================================================================
            logger.debug(f"Validating Dolphin path: {args.dolphin_path}")

            # Check if path exists
            if not os.path.exists(args.dolphin_path):
                error_msg = f"Dolphin path does not exist: {args.dolphin_path}"
                logger.error("=" * 70)
                logger.error("DOLPHIN PATH NOT FOUND")
                logger.error("=" * 70)
                logger.error(f"Provided path: {args.dolphin_path}")
                logger.error("")
                logger.error("SOLUTION:")
                logger.error("  Use --dolphin-path with valid path:")
                logger.error(f"     --dolphin-path 'C:\\Path\\To\\Dolphin-x64'")
                logger.error("=" * 70)

                # Cleanup and exit
                if gui:
                    gui.close()
                return

            logger.debug("Dolphin path validated successfully")

            # ====================================================================
            # INITIALIZE allocation_result
            # ====================================================================
            logger.debug("Initializing allocation_result structure...")

            # Determine scenario early
            if args.num_agents == args.num_instances:
                detected_scenario = 'ONE_TO_ONE'
            elif args.num_agents < args.num_instances:
                detected_scenario = 'AGENT_MULTIPLE_INSTANCES'
            else:
                detected_scenario = 'INSTANCE_SHARING'

            # Initialize allocation_result with empty allocation
            allocation_result = {
                'scenario': detected_scenario,
                'allocation': {},  # Will be filled later
                'num_agents': args.num_agents,
                'num_instances': args.num_instances,
                'dolphin_pids': []  # Will be filled after launch
            }

            logger.debug(f"Detected scenario: {detected_scenario}")
            logger.debug("allocation_result initialized successfully")

            # ====================================================================
            # CHECK AND CLOSE EXISTING DOLPHIN INSTANCES
            # ====================================================================
            logger.info("")
            logger.info("=" * 70)
            logger.info("CHECKING FOR EXISTING DOLPHIN INSTANCES")
            logger.info("=" * 70)

            try:
                # noinspection PyUnusedImports
                import psutil

                # Find all running Dolphin.exe processes
                existing_dolphin = []
                for proc in psutil.process_iter(['pid', 'name', 'exe']):
                    try:
                        if proc.info['name'] and 'dolphin' in proc.info['name'].lower():
                            # Verify it's actually Dolphin.exe (not dolphin-related files)
                            if proc.info['exe'] and 'dolphin.exe' in proc.info['exe'].lower():
                                existing_dolphin.append(proc.info['pid'])
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass

                if existing_dolphin:
                    logger.warning(f"Found {len(existing_dolphin)} existing Dolphin instance(s)")
                    logger.warning(f"PIDs: {existing_dolphin}")
                    logger.warning("")
                    logger.warning("These instances will interfere with memory hook!")
                    logger.warning("Closing them automatically...")
                    logger.warning("")

                    # Close all existing instances
                    cleanup_dolphin_processes(existing_dolphin, emergency=False)

                    # Verify cleanup succeeded
                    time.sleep(1.0)
                    still_running = [pid for pid in existing_dolphin if psutil.pid_exists(pid)]

                    if still_running:
                        logger.error("=" * 70)
                        logger.error("FAILED TO CLOSE EXISTING DOLPHIN INSTANCES")
                        logger.error("=" * 70)
                        logger.error(f"Still running: {still_running}")
                        logger.error("")
                        logger.error("SOLUTION:")
                        logger.error("  1. Close Dolphin manually via Task Manager")
                        logger.error("  2. Run training script again")
                        logger.error("=" * 70)

                        if gui:
                            gui.close()

                        _cleanup_done = True
                        return
                    else:
                        logger.info("All existing Dolphin instances closed successfully")
                else:
                    logger.info("No existing Dolphin instances found")

            except ImportError:
                logger.warning("psutil not available. Cannot check for existing Dolphin")
                logger.warning("Install with: pip install psutil")
            except Exception as check_error:
                logger.error(f"Error checking existing Dolphin: {check_error}")
                import traceback
                traceback.print_exc()

            logger.info("=" * 70)
            logger.info("")

            # ====================================================================
            # STEP 1: LAUNCH DOLPHIN VIA POWERSHELL
            # ====================================================================
            success = launch_dolphin_instances_via_powershell(
                num_instances=args.num_instances,
                dolphin_path=args.dolphin_path,
                minimize_dolphin=True,  # Réduire menus Dolphin
                minimize_game=False  # Garder fenêtres jeu visibles
            )

            # ====================================================================
            # RÉCUPÉRATION DES PIDs DEPUIS FICHIERS TEMPORAIRES
            # ====================================================================
            logger.info("")
            logger.info("=" * 70)
            logger.info("RÉCUPÉRATION PIDs DOLPHIN")
            logger.info("=" * 70)

            dolphin_pids = []
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Read PIDs from vision/temp/
            vision_dir = os.path.join(script_dir, "vision")
            temp_dir = os.path.join(vision_dir, "temp")

            for i in range(args.num_instances):
                pid_file = os.path.join(temp_dir, f"dolphin_pid_{i}.tmp")
                try:
                    if os.path.exists(pid_file):
                        with open(pid_file, 'r') as f:
                            pid = int(f.read().strip())
                            dolphin_pids.append(pid)
                            logger.info(f"Instance {i}: PID {pid}")

                        # Keep PID file for emergency cleanup
                        # Don't remove it yet, will be cleaned up after training starts
                        logger.debug(f"Keeping PID file for emergency cleanup: {pid_file}")
                    else:
                        logger.warning(f"Instance {i}: PID file not found ({pid_file})")
                        dolphin_pids.append(None)

                except Exception as pid_read_error:
                    logger.error(f"Instance {i}: Error reading PID: {pid_read_error}")
                    dolphin_pids.append(None)

            # Check if we retrieved all PIDs
            pids_found = sum(1 for pid in dolphin_pids if pid is not None)

            if pids_found == 0:
                logger.error("No Dolphin PID retrieved")
                logger.error("PowerShell did not create the temporary PID files")
                logger.error("Check that the PowerShell script is working correctly")

                # CRITICAL: Try to find and close any Dolphin instances manually
                logger.warning("")
                logger.warning("=" * 70)
                logger.warning("ATTEMPTING MANUAL DOLPHIN CLEANUP")
                logger.warning("=" * 70)

                try:
                    # noinspection PyUnusedImports
                    import psutil

                    # Find all Dolphin.exe processes
                    orphan_pids = []
                    for proc in psutil.process_iter(['pid', 'name', 'create_time']):
                        try:
                            if proc.info['name'] and 'dolphin.exe' in proc.info['name'].lower():
                                # Only close recently created processes (within last 60 seconds)
                                proc_age = time.time() - proc.info['create_time']
                                if proc_age < 60:
                                    orphan_pids.append(proc.info['pid'])
                                    logger.warning(
                                        f"Found recent Dolphin process: PID {proc.info['pid']} (age: {proc_age:.1f}s)")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    if orphan_pids:
                        logger.warning(f"Closing {len(orphan_pids)} orphan Dolphin process(es)...")
                        cleanup_dolphin_processes(orphan_pids, emergency=True)
                        logger.info("Orphan Dolphin processes closed")
                    else:
                        logger.warning("No recent Dolphin processes found")

                except ImportError:
                    logger.error("psutil not available - cannot auto-close Dolphin")
                    logger.error("Please close Dolphin windows manually")
                except Exception as manual_cleanup_error:
                    logger.error(f"Manual cleanup failed: {manual_cleanup_error}")

                logger.warning("=" * 70)
                logger.warning("")

                if gui:
                    gui.close()

                _cleanup_done = True
                return

            elif pids_found < args.num_instances:
                logger.warning(f"Partial PIDs: {pids_found}/{args.num_instances}")
                logger.warning("Some instances were not detected")
            else:
                logger.info(f"All PIDs retrieved ({pids_found}/{args.num_instances})")

            logger.info("=" * 70)
            logger.info("")

            # Store PIDs in pre-initialized allocation_result
            allocation_result['dolphin_pids'] = dolphin_pids

            # Also store in global variable for signal handler
            global _global_dolphin_pids
            _global_dolphin_pids = dolphin_pids
            logger.debug(f"PIDs stored globally for signal handler: {dolphin_pids}")

            # Verify PIDs were stored correctly
            logger.debug(f"Stored {len(dolphin_pids)} PIDs in allocation_result")
            logger.debug(f"PIDs: {dolphin_pids}")

            # Register signal handlers for Ctrl+C and termination
            # This ensures Dolphin is closed even when blocking (cv2.waitKey, input(), etc.)
            signal.signal(signal.SIGINT, emergency_signal_handler)
            signal.signal(signal.SIGTERM, emergency_signal_handler)
            logger.info("Signal handlers registered (SIGINT, SIGTERM)")

            # SIGINT (Ctrl+C) and SIGTERM are handled separately as normal shutdown
            def emergency_cleanup_handler():
                # Declare global at the very beginning of function
                global _cleanup_done

                # Skip if cleanup already done in finally block
                if _cleanup_done:
                    logger.debug("Cleanup already done, skipping atexit handler")
                    return

                logger.warning("Emergency cleanup triggered (unexpected termination)")

                # Access dolphin_pids from allocation_result if available
                pids_to_cleanup = []
                try:
                    if 'allocation_result' in globals() and allocation_result is not None:
                        pids_to_cleanup = allocation_result.get('dolphin_pids', [])

                    if pids_to_cleanup:
                        cleanup_dolphin_processes(pids_to_cleanup, emergency=True)
                    else:
                        logger.debug("No Dolphin PIDs found for emergency cleanup")
                except Exception as emergency_error:
                    logger.error(f"Emergency cleanup error: {emergency_error}")

                _cleanup_done = True

            # Register for unexpected exits (crashes)
            atexit.register(emergency_cleanup_handler)

            # Don't register SIGINT/SIGTERM here, they're handled in the finally block as normal cleanup
            logger.info("Emergency cleanup handler registered")

            if not success:
                logger.error("Failed to launch Dolphin instances")
                cleanup_dolphin_processes(dolphin_pids, emergency=True)  # Explicit cleanup on failure
                if gui:
                    gui.close()
                _cleanup_done = True
                return

            # ====================================================================
            # VALIDATE BACKGROUND INPUT CONFIGURATION
            # ====================================================================
            logger.info("")
            logger.info("=" * 70)
            logger.info("VALIDATING BACKGROUND INPUT")
            logger.info("=" * 70)

            try:
                import configparser

                dolphin_dir = os.path.dirname(args.dolphin_path)
                validation_passed = True

                for instance_id in range(args.num_instances):
                    # Instance 0 uses "User", others use "User1", "User2", etc.
                    user_folder_name = "User" if instance_id == 0 else f"User{instance_id}"
                    user_folder = os.path.join(dolphin_dir, user_folder_name)
                    dolphin_ini = os.path.join(user_folder, "Config", "Dolphin.ini")

                    if not os.path.exists(dolphin_ini):
                        logger.warning(f"Instance {instance_id}: Dolphin.ini not found")
                        logger.warning(f"Expected: {dolphin_ini}")
                        validation_passed = False
                        continue

                    # Parse INI file
                    config = configparser.ConfigParser()
                    config.read(dolphin_ini)

                    # Check BackgroundInput setting
                    if config.has_option('Input', 'BackgroundInput'):
                        value = config.get('Input', 'BackgroundInput')
                        if value.lower() in ['true', '1', 'yes']:
                            logger.info(f"Instance {instance_id}: Background Input ENABLED")
                        else:
                            logger.warning(f"Instance {instance_id}: Background Input DISABLED")
                            logger.warning(f"   Current value: {value}")
                            validation_passed = False
                    else:
                        logger.warning(f"Instance {instance_id}: BackgroundInput setting not found")
                        validation_passed = False

                if not validation_passed:
                    logger.error("")
                    logger.error("BACKGROUND INPUT NOT PROPERLY CONFIGURED")
                    logger.error("=" * 70)
                    logger.error("Multi-instance training requires Background Input enabled")
                    logger.error("=" * 70)

                    try:
                        user_input = input("\nContinue anyway? (y/N): ")
                        if user_input.lower() != 'y':
                            logger.info("Training cancelled by user")

                            # Cleanup Dolphin instances
                            logger.warning("Closing Dolphin instances...")
                            cleanup_dolphin_processes(dolphin_pids, emergency=False)

                            if gui:
                                gui.close()

                            _cleanup_done = True
                            return

                    except KeyboardInterrupt:
                        logger.warning("")
                        logger.warning("=" * 70)
                        logger.warning("CANCELLED BY USER (Ctrl+C)")
                        logger.warning("=" * 70)
                        logger.warning("Closing Dolphin instances...")

                        cleanup_dolphin_processes(dolphin_pids, emergency=False)
                        _cleanup_done = True

                        if gui:
                            gui.close()

                        _cleanup_done = True
                        logger.warning("Cleanup complete")
                        logger.warning("=" * 70)
                        return
                else:
                    logger.info("")
                    logger.info("All instances have Background Input enabled")

            except Exception as validation_error:
                logger.error(f"Background Input validation failed: {validation_error}")
                logger.warning("Continuing without validation...")

            logger.info("=" * 70)
            logger.info("")

            # ====================================================================
            # STEP 2: WAIT FOR WINDOW DETECTION (POLLING 10S)
            # ====================================================================
            windows_ready = wait_for_dolphin_windows(
                num_instances=args.num_instances,
                timeout=args.dolphin_timeout,  # Timeout 60s
                check_interval=10  # Check toutes les 10s
            )

            if not windows_ready:
                logger.error("Toutes les fenêtres n'ont pas été détectées")
                logger.error("💡 Les instances Dolphin sont probablement lancées mais pas encore prêtes")
                logger.error("Tu peux continuer manuellement ou relancer avec un timeout plus long")

                if gui:
                    gui.close()
                return

            # ====================================================================
            # ÉTAPE 3 : CALCULER RÉPARTITION AGENTS/INSTANCES
            # ====================================================================
            allocation_result = calculate_agent_allocation(
                num_agents=args.num_agents,
                num_instances=args.num_instances,
                allocation_mode=args.allocation_mode,
                allocation_map=args.allocation_map,
                multi_agent_mode=args.multi_agent_mode,
            )

            scenario = allocation_result['scenario']
            allocation = allocation_result['allocation']

            # ====================================================================
            # ÉTAPE 4 : CRÉER ENVIRONNEMENTS SELON SCÉNARIO
            # ====================================================================

            if scenario == "ONE_TO_ONE" or scenario == "AGENT_MULTIPLE_INSTANCES":
                # ================================================================
                # CAS 1 : Agent PPO, N Environnements
                # ================================================================
                logger.info("")
                logger.info("=" * 70)
                logger.info("🎯 ARCHITECTURE : 1 Agent PPO, N Environnements")
                logger.info("=" * 70)
                logger.info(f"   Agent unique contrôle {args.num_instances} environnements")
                logger.info(f"   Buffer partagé : ✅")
                logger.info(f"   Parallélisation : ✅")
                logger.info("=" * 70)
                logger.info("")

                def make_env(env_idx: int):
                    """Factory pour créer un environnement par instance Dolphin"""

                    def _init():
                        return MonsterHunterEnv(
                            use_vision=use_vision,
                            use_memory=use_memory,
                            grayscale=args.grayscale,
                            frame_stack=4,
                            use_controller=use_controller_mode,
                            controller_debug=False,
                            use_advanced_rewards=True,
                            save_state_slot=args.save_state,
                            # Vision temps réel UNIQUEMENT pour instance 0
                            rt_vision=(args.rtvision and use_vision and env_idx == 0),
                            rt_minimap=(args.rtminimap and use_vision and use_memory and env_idx == 0),
                            instance_id=env_idx,
                        )

                    return _init

                # Create vectorized environments
                # NOTE: Using DummyVecEnv instead of SubprocVecEnv to avoid pickling issues
                # with MemoryReader's ctypes callbacks
                logger.info("Creating vectorized environments...")

                # Use DummyVecEnv for compatibility with MemoryReader
                env = DummyVecEnv([make_env(i) for i in range(args.num_instances)])
                logger.info(f"{args.num_instances} environments created (DummyVecEnv)")

                # Warn about performance
                if args.num_instances > 3:
                    logger.warning("WARNING: Using DummyVecEnv with many instances")
                    logger.warning("  - Environments run sequentially, not in parallel")
                    logger.warning("  - Consider using 1-3 instances for better performance")
                    logger.warning("  - Or fix pickling issue in MemoryReader for true parallelism")

                # ====================================================================
                # CONFIGURATION HIDHIDE (ISOLATION MANETTES VIRTUELLES)
                # ====================================================================
                if args.num_instances > 1 and use_controller_mode:
                    logger.info("")
                    logger.info("=" * 70)
                    logger.info("CONFIGURATION HIDHIDE")
                    logger.info("=" * 70)

                    # Continuer seulement si import réussi
                    if HIDHIDE_AVAILABLE:
                        try:
                            if not is_admin():
                                logger.warning("Script non lancé en administrateur")
                                logger.warning("HidHide nécessite droits élevés pour fonctionner")
                                logger.warning("Isolation manettes désactivée")
                            else:
                                logger.info("Initialisation HidHide...")
                                hidhide = HidHideManager()

                                logger.info("Waiting for virtual controllers creation and driver registration...")
                                time.sleep(5.0)  # 5s for ViGEm driver registration

                                # Get controllers from environments
                                controllers = env.env_method('get_controller')

                                logger.info(f"Retrieved {len(controllers)} controllers...")

                                # Additional verification: check if gamepads are actually created
                                controllers_ready = 0
                                for controller in controllers:
                                    if controller and hasattr(controller, 'gamepad') and controller.gamepad is not None:
                                        controllers_ready += 1

                                logger.info(f"Controllers with active gamepads: {controllers_ready}/{len(controllers)}")

                                if controllers_ready == 0:
                                    logger.warning("No active gamepads detected after 5s")
                                    logger.warning("HidHide isolation will be skipped")
                                    logger.warning("Possible causes:")
                                    logger.warning("  1. ViGEmBus driver not loaded")
                                    logger.warning("  2. Gamepads not created yet (increase delay)")
                                    logger.warning("  3. Permission issues (run as admin)")
                                else:
                                    logger.info(
                                        f"Proceeding with HidHide configuration for {controllers_ready} controllers...")

                                # Configurer HidHide pour chaque instance
                                for instance_idx, controller in enumerate(controllers):
                                    if controller is None:
                                        logger.warning(f"Instance {instance_idx} : contrôleur non disponible")
                                        continue

                                    if not controller.use_controller:
                                        logger.info(f"Instance {instance_idx} : mode clavier, HidHide non nécessaire")
                                        continue

                                    logger.info(f"Configuration instance {instance_idx}...")

                                    # Obtenir device path ViGEm
                                    device_path = hidhide.get_vgamepad_device_path(instance_idx)

                                    if device_path is None:
                                        logger.error(f"Device ViGEm #{instance_idx} non trouvé")
                                        continue

                                    # Note : Utilise le chemin Dolphin global pour toutes les instances
                                    # En multi-instance, toutes les fenêtres Dolphin proviennent du même .exe
                                    # HidHide isole par device ID, pas par processus
                                    dolphin_exe = args.dolphin_path

                                    # Configurer isolation
                                    hidhide.configure_device_for_exe(
                                        device_instance_path=device_path,
                                        allowed_exe=dolphin_exe
                                    )

                                    logger.info(f"Instance {instance_idx} configurée avec HidHide")

                                logger.info("Configuration HidHide terminée")

                        except Exception as hidhide_config_error:
                            logger.error(f"Erreur configuration HidHide : {hidhide_config_error}")
                            logger.error("Isolation manettes désactivée")

                    logger.info("=" * 70)
                    logger.info("")

            elif scenario == "INSTANCE_SHARING":
                # ================================================================
                # CAS 3 : Multi-Agents Partageant Instances
                # ================================================================
                logger.info("")
                logger.info("=" * 70)
                logger.info("🎯 SCÉNARIO 3 : PARTAGE D'INSTANCES (IMPLÉMENTATION PARTIELLE)")
                logger.info("=" * 70)
                logger.info(f"   {args.num_agents} agents partagent {args.num_instances} instances")
                logger.info(f"   Mode : {args.multi_agent_mode}")
                logger.info("")

                # Créer les environnements (comme SCÉNARIO 2)
                def make_env(env_index: int):
                    """Factory pour créer un environnement par instance Dolphin"""

                    def _init():
                        return MonsterHunterEnv(
                            use_vision=use_vision,
                            use_memory=use_memory,
                            grayscale=args.grayscale,
                            frame_stack=4,
                            use_controller=use_controller_mode,
                            controller_debug=False,
                            use_advanced_rewards=True,
                            save_state_slot=args.save_state,
                            rt_vision=(args.rtvision and use_vision and env_index == 0),
                            rt_minimap=(args.rtminimap and use_vision and use_memory and env_index == 0),
                            instance_id=env_index,
                        )

                    return _init

                from stable_baselines3.common.vec_env import SubprocVecEnv

                logger.info(f"{args.num_instances} environnements créés (SubprocVecEnv)")
                logger.info("")
                logger.info("SCÉNARIO 3 : Partage d'instances")
                logger.info(f"   {args.num_agents} agents partageront {args.num_instances} environnements")
                logger.info(f"   Mode de gestion : {args.multi_agent_mode}")
                logger.info(f"   Scheduler : Tour par tour ou vote selon mode")
                logger.info("")

            # ====================================================================
            # ÉTAPE 5 : TESTS DES ENVIRONNEMENTS
            # ====================================================================
            logger.info("")
            logger.info("=" * 70)
            logger.info("🔍 TESTS DES ENVIRONNEMENTS")
            logger.info("=" * 70)

            try:
                # Test 1 : Reset
                logger.info("Test 1/3 : Reset global...")

                # VecNormalize might return only obs (legacy) or (obs, info)
                reset_result = env.reset()

                if isinstance(reset_result, tuple):
                    obs, reset_info = reset_result
                    logger.debug("Reset returned (obs, info) tuple")
                else:
                    obs = reset_result
                    reset_info = {}
                    logger.debug("Reset returned obs only (legacy format)")

                # DEBUG: Show what we actually received
                logger.debug(f"Type of obs: {type(obs)}")
                if isinstance(obs, dict):
                    logger.debug(f"Obs is dict with keys: {list(obs.keys())}")
                elif isinstance(obs, (list, tuple)):
                    logger.debug(f"Obs is sequence with {len(obs)} elements")

                # Verify structure based on actual type
                if isinstance(obs, dict):
                    # Dict observation space (multi-modal)
                    logger.info(f"Dict observation with {len(obs)} modalities")
                    logger.info(f"   Keys : {list(obs.keys())}")

                    for key, value in obs.items():
                        if isinstance(value, np.ndarray):
                            # VecEnv returns stacked arrays: (num_envs, *shape)
                            logger.info(f"      {key}: {value.shape} (num_envs={env.num_envs})")
                        else:
                            logger.info(f"      {key}: {type(value)}")

                elif isinstance(obs, np.ndarray):
                    # Single Box observation space
                    logger.info(f"Box observation: {obs.shape}")
                else:
                    logger.warning(f"Unexpected obs type: {type(obs)}")

                # Test 2 : Step
                logger.info("Test 2/3 : Step (neutral action)...")
                actions = np.array([0] * env.num_envs, dtype=np.int64)

                try:
                    # Simple step call with basic unpacking
                    step_result = env.step(actions)

                    # Detect format by checking result length
                    result_length = len(step_result)

                    if result_length == 5:
                        # Gymnasium format: obs, reward, terminated, truncated, info
                        # noinspection PyTypeChecker,PyTupleAssignmentBalance
                        obs, rewards, terminated, truncated, infos = step_result
                        # Convert terminated/truncated to dones for compatibility
                        dones = np.logical_or(terminated, truncated)
                        logger.info("Step successful (Gymnasium format: 5 values)")
                    elif result_length == 4:
                        # Legacy format: obs, reward, done, info
                        obs, rewards, dones, infos = step_result
                        # Create truncated array for compatibility
                        truncated = np.array([False] * env.num_envs, dtype=bool)
                        logger.info("Step successful (legacy format: 4 values)")
                    else:
                        # Unexpected format - log detailed error
                        logger.error(f"Unexpected step() return format: {result_length} values")
                        logger.error(f"Expected 4 (legacy) or 5 (Gymnasium) values")
                        logger.error(f"step_result types: {[type(x).__name__ for x in step_result]}")
                        raise ValueError(
                            f"env.step() returned {result_length} values, expected 4 or 5. "
                            f"Check your VecEnv configuration."
                        )

                except Exception as step_test_error:
                    logger.error(f"Step test error: {step_test_error}")
                    import traceback
                    traceback.print_exc()
                    if gui:
                        gui.close()
                    return

                # Test 3 : Vérifier titres fenêtres
                logger.info("Test 3/3 : Vérification fenêtres capturées...")
                try:
                    # Méthode custom pour récupérer titre fenêtre
                    window_titles = env.env_method('get_window_title')
                    for i, title in enumerate(window_titles):
                        status = "true" if title else "false"
                        logger.info(f"   {status} Instance {i} : '{title}'")
                except AttributeError:
                    logger.warning("Méthode get_window_title() non disponible")

                logger.info("Tous les tests ont réussi")

            except Exception as test_error:
                import traceback
                logger.error(f"Erreur durant les tests : {test_error}")
                traceback.print_exc()
                logger.error("")
                logger.error("💡 Les environnements ne sont pas correctement initialisés")
                logger.error("   Vérifie que Dolphin tourne et que MH Tri est chargé")

                if gui:
                    gui.close()
                return

            logger.info("=" * 70)
            logger.info("")

            # ====================================================================
            # ÉTAPE 6 : RÉSUMÉ CONFIGURATION
            # ====================================================================
            logger.info("")
            logger.info("=" * 70)
            logger.info("📊 RÉSUMÉ CONFIGURATION MULTI-INSTANCE")
            logger.info("=" * 70)
            logger.info(f"Scénario           : {scenario}")
            logger.info(f"Instances Dolphin  : {args.num_instances}")
            logger.info(f"Environnements     : {env.num_envs}")
            logger.info(f"Agents PPO         : {args.num_agents}")
            logger.info(f"Mode allocation    : {args.allocation_mode}")
            logger.info("")
            logger.info("Répartition :")
            for agent_id, instances in sorted(allocation.items()):
                instances_str = ", ".join(map(str, instances))
                logger.info(f"   Agent {agent_id} → Instances [{instances_str}]")
            logger.info("")
            logger.info(f"Buffer partagé     : (toutes expériences)")
            logger.info(f"Collection         : ~{args.num_instances}× plus rapide")
            logger.info("=" * 70)
            logger.info("")

        # ========================================================================
        # VECNORMALIZE (COMMUN À TOUS LES CAS)
        # ========================================================================
        vec_normalize_path = os.path.join(logs_dir, "vec_normalize.pkl")

        if args.resume and os.path.exists(vec_normalize_path) and not args.force_new_vecnormalize:
            # REPRISE + .pkl existe
            logger.info(f"📦 Reprise - Tentative chargement VecNormalize...")
            logger.info(f"   Fichier : {vec_normalize_path}")
            try:
                env = VecNormalize.load(vec_normalize_path, env)
                logger.info("VecNormalize chargé depuis checkpoint")
            except Exception as VecNormalize_load_error:
                training_logger.log_error(VecNormalize_load_error, context="Chargement VecNormalize")
                logger.error(f"Erreur chargement VecNormalize: {VecNormalize_load_error}")
                logger.warning("Création d'un NOUVEAU VecNormalize")

                env = VecNormalize(
                    env,
                    norm_obs=True,
                    norm_reward=False,
                    clip_obs=10.0,
                    clip_reward=100.0,
                    gamma=0.997
                )
        else:
            # NOUVEAU ou FORCE-NEW
            if args.force_new_vecnormalize:
                logger.info("🔄 Option --force-new-vecnormalize activée")

            logger.info("📦 Création d'un nouveau VecNormalize...")
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
                clip_reward=100.0,
                gamma=0.997
            )
            logger.info("VecNormalize créé")

        logger.info("🔧 VecNormalize actif:")
        logger.info("   Observations normalisées et clippées")
        logger.info("   Rewards normalisées et clippées")

    except (RuntimeError, ValueError, ImportError, OSError) as env_error:
        # LOG THE ERROR
        training_logger.log_error(env_error, context="Création environnement")

        logger.error(f"ERROR environment creation :")
        logger.error(f"{env_error}")
        import traceback
        traceback.print_exc()

        # CLEANUP DOLPHIN INSTANCES BEFORE EXITING - FORCE CLEANUP
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning("ENVIRONMENT CREATION FAILED - CLEANING UP DOLPHIN")
        logger.warning("=" * 70)

        try:
            # Check if allocation_result exists in the correct scope
            # allocation_result is initialized earlier but might not have PIDs yet
            dolphin_pids = []

            # Try to get PIDs from allocation_result if it exists
            if 'allocation_result' in locals() and allocation_result is not None:
                dolphin_pids = allocation_result.get('dolphin_pids', [])
                logger.debug(f"Found {len(dolphin_pids)} PIDs in allocation_result")

            # If allocation_result has no PIDs, try reading from PID files
            # This handles the case where Dolphin launched but env creation failed
            if not dolphin_pids and args.num_instances > 1:
                logger.warning("No PIDs in allocation_result, attempting to read from PID files...")
                script_dir = os.path.dirname(os.path.abspath(__file__))

                for i in range(args.num_instances):
                    pid_file = os.path.join(script_dir, f"dolphin_pid_{i}.tmp")
                    try:
                        if os.path.exists(pid_file):
                            with open(pid_file, 'r') as f:
                                pid = int(f.read().strip())
                                if pid > 0:
                                    dolphin_pids.append(pid)
                                    logger.debug(f"Recovered PID {pid} from file {pid_file}")
                            # Clean up PID file
                            os.remove(pid_file)
                    except Exception as pid_recovery_error:
                        logger.debug(f"Could not recover PID from {pid_file}: {pid_recovery_error}")

            # Filter valid PIDs
            valid_pids = [pid for pid in dolphin_pids if pid is not None and pid > 0]

            if valid_pids:
                logger.warning(f"Closing {len(valid_pids)} Dolphin instance(s)...")
                cleanup_dolphin_processes(valid_pids, emergency=True)

                # Verify cleanup with detailed status
                import psutil
                time.sleep(1.0)  # Wait for processes to fully terminate

                still_running = []
                for pid in valid_pids:
                    if psutil.pid_exists(pid):
                        still_running.append(pid)
                        logger.error(f"  PID {pid} still running - attempting force kill...")
                        try:
                            process = psutil.Process(pid)
                            process.kill()
                            process.wait(timeout=2)
                            logger.warning(f"  PID {pid} force killed")
                        except Exception as force_kill_error:
                            logger.error(f"  Failed to force kill PID {pid}: {force_kill_error}")

                if not still_running:
                    logger.info("All Dolphin instances closed successfully")
                else:
                    logger.error(f"WARNING: {len(still_running)} Dolphin process(es) still running: {still_running}")
                    logger.error("You may need to close them manually via Task Manager")
            else:
                logger.warning("No valid Dolphin PIDs found for cleanup")
                logger.warning("If Dolphin instances are running, close them manually")

        except Exception as cleanup_error:
            logger.error(f"Error during emergency Dolphin cleanup: {cleanup_error}")
            import traceback
            traceback.print_exc()
            logger.error("")
            logger.error("DOLPHIN CLEANUP FAILED - PLEASE CLOSE MANUALLY")
            logger.error("Open Task Manager (Ctrl+Shift+Esc) and end Dolphin.exe processes")

        logger.warning("=" * 70)
        logger.warning("")

        if gui:
            gui.close()

        # Mark cleanup done to prevent atexit handler
        _cleanup_done = True

        return

    # ====================================================================
    # VALIDATION MULTI-INSTANCE
    # ====================================================================
    if args.num_instances > 1:
        logger.info("")
        logger.info("=" * 70)
        logger.info("VALIDATION MULTI-INSTANCE")
        logger.info("=" * 70)

        try:
            # Vérifier fenêtres capturées
            window_titles = env.env_method('get_window_title')

            logger.info(f"Fenêtres capturées par les {len(window_titles)} instances :")

            collision_detected = False
            for i, title in enumerate(window_titles):
                expected = f"MHTri-{i}"
                match_status = "CORRECT" if expected in title else "ERREUR"

                logger.info(f"  Instance {i} : '{title}' [{match_status}]")

                if match_status == "ERREUR":
                    collision_detected = True

            if collision_detected:
                logger.error("")
                logger.error("COLLISION DÉ TECTÉ E : Instances capturent mauvaises fenêtres")
                logger.error("Solutions :")
                logger.error("  1. Vérifie script PowerShell de lancement")
                logger.error("  2. Vérifie renommage fenêtres (MHTri-1, MHTri-2, etc.)")
                logger.error("  3. Relance avec moins d'instances pour tester")
                logger.error("")

                if not args.debug_steps:
                    logger.error("Arrêt pour éviter données corrompues")
                    logger.error("Utilise --debug-steps 100 pour forcer continuation")

                    if gui:
                        gui.close()
                    env.close()
                    return
            else:
                logger.info("")
                logger.info("Toutes les instances capturent les bonnes fenêtres")

        except Exception as validation_error:
            logger.error(f"Erreur validation : {validation_error}")
            logger.warning("Validation ignorée, continue entraînement...")

        logger.info("=" * 70)
        logger.info("")

    # ============================================================================
    # Agent(s) (création ou chargement)
    # ============================================================================
    logger.info("🤖 AGENT(S) PPO")

    # Initialiser
    agents = []
    scheduler = None

    # S'assurer que allocation_result existe
    if allocation_result is None:
        allocation_result = {
            'scenario': 'ONE_TO_ONE',
            'allocation': {0: [0]},
            'num_agents': 1,
            'num_instances': 1
        }

    try:
        scenario = allocation_result['scenario']

        # ====================================================================
        # CAS 1 & 2 : 1 seul agent PPO
        # ====================================================================
        if args.num_agents == 1 or scenario in ["ONE_TO_ONE", "AGENT_MULTIPLE_INSTANCES"]:
            logger.info("📋 Mode 1 agent PPO")

            # Tentative de reprise
            agent, previous_timesteps = load_model_if_resume(args, env, device, training_logger)

            if agent is None:
                # Créer nouveau modèle
                logger.info("Création d'un NOUVEAU modèle...")

                # Adapter les hyperparamètres selon le mode debug
                n_steps = 512 if args.small_rollout else 2048
                batch_size = 32 if args.small_rollout else 64

                agent = create_ppo_agent(
                    environment_new=env,
                    learning_rate=args.lr,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    features_dim=256,
                    device=device,
                    verbose=1,
                    tensorboard_log=logs_dir
                )

                previous_timesteps = 0
                logger.info(f"Agent créé")

                # TEST CNN
                if (base_env is not None
                        and base_env.use_vision
                        and isinstance(base_env.observation_space, spaces.Dict)):
                    logger.info("🧠 Test extraction features CNN...")
                    try:
                        import torch

                        with torch.no_grad():
                            test_obs = base_env.reset()[0]

                            logger.debug(f"   Observation de test :")
                            for key, value in test_obs.items():
                                logger.debug(f"      {key}: shape={value.shape}, dtype={value.dtype}")

                            # Préparer obs pour le modèle
                            obs_tensor = {}
                            for key, value in test_obs.items():
                                if isinstance(value, dict):
                                    obs_tensor[key] = {
                                        k: torch.FloatTensor(v).unsqueeze(0).to(device)
                                        for k, v in value.items()
                                    }
                                else:
                                    obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(device)

                            # Extraire features
                            features = agent.policy.extract_features(obs_tensor, agent.policy.features_extractor)
                            logger.info(f"Features extraites : shape={features.shape}")

                    except Exception as cnn_error:
                        training_logger.log_error(cnn_error, context="Test CNN")
                        logger.error(f"Erreur test CNN : {cnn_error}")
                        import traceback
                        traceback.print_exc()
                elif base_env is None:
                    logger.info("ℹ️  Test CNN ignoré (mode multi-instance, pas d'accès direct à base_env)")
            else:
                logger.info(f"Agent chargé depuis {args.resume}")
                logger.info(f"Timesteps précédents : {previous_timesteps:,}")

            # Ajouter à la liste (compatibilité avec la suite)
            agents = [agent]

        # ====================================================================
        # CAS 3 : Plusieurs agents PPO (SCÉNARIO 3 - Instance Sharing)
        # ====================================================================
        else:
            logger.info(f"📋 Mode {args.num_agents} agents PPO")
            logger.info(f"Scénario détecté : {scenario}")

            # Utiliser raise au lieu de sys.exit(1)
            if args.resume:
                error_msg = "Reprise multi-agents pas encore implémentée"
                logger.error(error_msg)
                logger.error("Démarre un nouvel entraînement sans --resume")

                # Créer l'exception et la logger
                resume_error = NotImplementedError(error_msg)
                training_logger.log_error(resume_error, context="Multi-agent resume")
                raise resume_error

            logger.info(f"Création de {args.num_agents} agents...")

            n_steps = 512 if args.small_rollout else 2048
            batch_size = 32 if args.small_rollout else 64

            for agent_id in range(args.num_agents):
                logger.info(f"   Agent {agent_id + 1}/{args.num_agents}...")

                agent = create_ppo_agent(
                    environment_new=env,
                    learning_rate=args.lr,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    features_dim=256,
                    device=device,
                    verbose=0,  # Moins verbeux pour multi-agents
                    tensorboard_log=f"{logs_dir}/agent_{agent_id}"
                )

                agents.append(agent)

            logger.info(f"{len(agents)} agents créés")

            # Utiliser raise au lieu de sys.exit(1)
            if not MULTI_AGENT_AVAILABLE:
                error_msg = "MultiAgentScheduler non disponible"
                logger.error(error_msg)
                logger.error("Fichier manquant : utils/multi_agent_scheduler.py")

                # Créer l'exception et la logger
                import_error = ImportError(error_msg)
                training_logger.log_error(import_error, context="Multi-agent import")
                raise import_error

            allocation = allocation_result['allocation']

            scheduler = MultiAgentScheduler(
                agents=agents,
                allocation=allocation,
                mode=args.multi_agent_mode,
                block_size=args.block_size,
                weighted_eval_freq=getattr(args, 'weighted_eval_freq', 100),
            )

            logger.info(f"Scheduler créé (mode={args.multi_agent_mode})")

            previous_timesteps = 0

    except (NotImplementedError, ImportError) as agent_error:
        # Gérer les erreurs proprement
        logger.error(f"❌ Erreur création agent(s) : {agent_error}")
        training_logger.log_error(agent_error, context="Création agents")

        if gui:
            gui.close()

        # Cleanup environnement
        if env is not None:
            try:
                env.close()
            except Exception as cleanup_error:
                logger.debug(f"Erreur nettoyage env: {cleanup_error}")

        return  # Return au lieu de sys.exit(1)

    except Exception as agent_error:
        training_logger.log_error(agent_error, context="Création agents")
        logger.error(f"Erreur création agent(s) : {agent_error}")
        import traceback
        traceback.print_exc()

        if gui:
            gui.close()

        # Cleanup environnement
        if env is not None:
            try:
                env.close()
            except Exception as env_cleanup_error:
                logger.debug(f"Erreur fermeture env: {env_cleanup_error}")

        return  # Return au lieu de sys.exit(1)

    # ============================================================================
    # Callbacks
    # ============================================================================
    callbacks = []

    # Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="checkpoint"
    )
    callbacks.append(checkpoint_callback)

    # Logging callback(s) - one per agent in multi-agent mode
    # Check if training_loggers is not None instead of using locals()
    if args.num_agents > 1 and training_loggers is not None:
        # Multi-agent: create callback per agent
        for agent_idx, agent_logger in enumerate(training_loggers):
            logging_callback = LoggingCallback(agent_logger)
            callbacks.append(logging_callback)
        logger.info(f"📊 {len(training_loggers)} logging callbacks created (one per agent)")
    else:
        # Single-agent: normal callback
        logging_callback = LoggingCallback(training_logger)
        callbacks.append(logging_callback)

    logger.info(f"{len(callbacks)} callbacks configurés")

    # Checkpoint carte d'exploration
    class ExplorationCheckpoint(BaseCallback):
        """
        Sauvegarde périodique de la carte d'exploration
        """

        def __init__(self, save_path: str, save_freq: int = 50000, advanced_logger: TrainingLogger = None):
            super().__init__()
            self.save_path = save_path
            self.save_freq = save_freq
            self.advanced_logger = advanced_logger

        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                filepath = None  # Initialiser avant le try
                try:
                    import json
                    # Récupérer le tracker depuis l'env
                    if hasattr(self.model.env, 'get_attr'):
                        env_reward_calc = self.model.env.get_attr('reward_calc')[0]
                        if env_reward_calc and hasattr(env_reward_calc, 'exploration_tracker'):
                            stats = env_reward_calc.exploration_tracker.get_stats()

                            # Afficher les stats avant nettoyage
                            logger.debug(f"Stats brutes récupérées:")
                            logger.debug(f"Type: {type(stats)}")
                            logger.debug(f"Clés: {list(stats.keys()) if isinstance(stats, dict) else 'N/A'}")
                            logger.debug(f"Contenu: {stats}")

                            # Nettoyer les stats pour JSON (notamment marker)
                            stats = self._make_json_serializable(stats)

                            # Afficher les stats APRÈS nettoyage
                            logger.debug(f"Stats après nettoyage:")
                            logger.debug(f"Type: {type(stats)}")
                            logger.debug(f"Contenu: {stats}")

                            filepath = os.path.join(self.save_path, f'exploration_{self.n_calls}.json')
                            with open(filepath, 'w') as m:
                                json.dump(stats, m, indent=2, ensure_ascii=False)

                            logger.info(f"💾 Carte exploration sauvegardée: {filepath}")
                            logger.info(f"Cubes: {stats['total_cubes']}, Zones: {stats['zones_discovered']}")

                            # LOGGER LE CHECKPOINT
                    if self.advanced_logger and filepath:
                        self.advanced_logger.log_checkpoint(filepath, self.n_calls)

                except (OSError, IOError, AttributeError, TypeError, Exception) as save_exploration_error:
                    # Logger l'erreur
                    if self.advanced_logger:
                        self.advanced_logger.log_error(save_exploration_error,
                                              context="Sauvegarde exploration")

                    logger.error(f"Erreur sauvegarde exploration: {save_exploration_error}")

                    # DEBUG : Afficher le traceback complet
                    traceback.print_exc()

            return True

        def _make_json_serializable(self, obj):
            """
            Convertit récursivement un objet en format JSON valide.

            Gère les cas spéciaux comme MarkerType, numpy arrays, etc.:
            - Enums (dont MarkerType) = string via .name
            - numpy types = types Python natifs
            - Dicts avec clés non-string = conversion clés en string
            - Listes/tuples = récursion
            """
            from environment.cube_markers import MarkerType

            # CAS 1 : Types de base déjà sérialisables
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj

            # Gestion explicite de MarkerType AVANT les autres enums
            # RAISON : C'est la cause principale de l'erreur "keys must be str, not MarkerType"
            # Cas 2 : On convertit les enums MarkerType en leur valeur string
            if isinstance(obj, MarkerType):
                return obj.value  # Retourne 'zone_transition', 'monster_location', etc.

            # CAS 3 : Autres Enums génériques (fallback)
            if hasattr(obj, '__class__') and 'Enum' in str(type(obj).__mro__):
                return obj.value  # Retourner le nom de l'enum (string)

            # CAS 4 : Types numpy
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()

            # Gestion des clés de dictionnaire MarkerType
            # RAISON : Les stats contiennent des dicts avec MarkerType comme clés
            #          JSON n'accepte que str/int/float/bool/None comme clés
            # CAS 5
            if isinstance(obj, dict):
                cleaned_dict = {}
                for json_key, json_value in obj.items():  # RENOMMER : value -> json_value
                    # Convertir la clé si nécessaire
                    if isinstance(json_key, MarkerType):
                        # Conversion enum = string pour clé JSON
                        cleaned_key = json_key.value
                    elif isinstance(json_key, (str, int, float, bool)) or json_key is None:
                        # Clé déjà valide pour JSON
                        cleaned_key = json_key
                    else:
                        # Autre type : convertir en string
                        cleaned_key = str(json_key)

                    # Convertir la valeur récursivement
                    cleaned_dict[cleaned_key] = self._make_json_serializable(json_value)

                return cleaned_dict

            # CAS 6 : Listes, tuples ou set
            elif isinstance(obj, (list, tuple, set)):
                return [self._make_json_serializable(item) for item in obj]

            # CAS 7 : Autres objets = convertir en string
            else:
                try:
                    return str(obj)
                except Exception as make_json_serializable_error:
                    return f"<non-serializable: {type(obj).__name__}: {str(make_json_serializable_error)}>"

    exploration_checkpoint = ExplorationCheckpoint(
        save_path=models_dir,
        save_freq=50000,
        advanced_logger = training_logger,
    )
    callbacks.append(exploration_checkpoint)

    # GUI
    if gui:
        gui_callback = GUIUpdateCallback(gui, verbose=1)
        callbacks.append(gui_callback)

    # Entraînement
    logger.info("ENTRAÎNEMENT")

    if gui:
        logger.info(f"🖥️ Interface graphique :")
        logger.info(f"   - Stats temps réel visibles")
        logger.info(f"   - Clique 'Arrêter' pour stop propre")
        logger.info(f"   - Ouvre 'Stats Étendues' et 'Reward Breakdown' pour détails")

    logger.info(f"📊 TensorBoard : tensorboard --logdir {logs_dir}")

    if previous_timesteps > 0:
        logger.info(f"REPRISE : Continuera à partir de {previous_timesteps:,} steps")
        logger.info(f"Objectif : {args.timesteps:,} steps supplémentaires")
        total_target = previous_timesteps + args.timesteps
        logger.info(f"Total visé : {total_target:,} steps")

    # ============================================================
    # ATTENTE AVANT DÉMARRAGE (avec gestion GUI et no-GUI)
    # ============================================================
    logger.info("PRÊT À DÉMARRER")

    # Initialiser avant le try
    shutdown_flag = None
    input_thread = None

    try:
        if gui:
            # MODE AVEC GUI : Utiliser un thread pour input() non-bloquant
            logger.info("⏸️ Appuie sur ENTRÉE pour démarrer l'entraînement...")
            logger.info("(ou attends 10 secondes pour démarrage automatique)")
            logger.info("(Ctrl+C pour annuler)")

            # Variable partagée entre threads
            start_requested = {'value': False, 'by_user': False}
            shutdown_flag = {'value': False}  # Flag pour arrêter le thread proprement

            def wait_for_enter():
                """
                Thread waiting for ENTER without blocking the GUI
                """
                try:
                    # La fonction input() met le thread en pause jusqu'à ce que l'utilisateur appuie sur la touche ENTRÉE.
                    # Cette méthode est utilisée ici dans un thread séparé pour ne pas bloquer le GUI.
                    input()
                except UnicodeDecodeError:
                    # --- Erreur d'encodage de saisie ---
                    # Peut survenir si le flux standard d'entrée (stdin) reçoit des caractères
                    # non valides selon l'encodage attendu (par exemple, un terminal exotique).
                    # Dans ce cas, on ignore l’erreur et on quitte proprement la fonction.
                    return
                except (EOFError, KeyboardInterrupt):
                    # --- Flux d'entrée fermé ou interruption clavier ---
                    # EOFError : le flux stdin a été fermé (par ex. fin de fichier, fermeture de terminal).
                    # KeyboardInterrupt : l'utilisateur a interrompu le programme avec Ctrl+C.
                    # Si le programme n’est pas déjà en cours d’arrêt, on interprète cette
                    # interruption comme une demande de "start" (par cohérence avec le reste du code).
                    # Don't mark as started if interrupted
                    # Let the main thread handle cleanup
                    if not shutdown_flag['value']:
                        logger.debug("Input thread interrupted")
                    return
                except Exception as input_error:
                    logger.debug(f"Input thread error: {input_error}")
                    return
                else:
                    # --- Cas normal : l’utilisateur a appuyé sur ENTRÉE ---
                    # Aucun problème d’encodage ni d’interruption ; l’entrée s’est déroulée normalement.
                    # On considère que l’utilisateur souhaite lancer le processus principal.
                    # Only mark as started if clean input received
                    if not shutdown_flag['value']:
                        start_requested['value'] = True  # Active le signal de démarrage
                        start_requested['by_user'] = True  # Indique que la demande provient de l’utilisateur

            # Lancer le thread d'attente
            input_thread = threading.Thread(target=wait_for_enter, daemon=True)
            input_thread.start()

            # Countdown avec vérification périodique
            countdown = 10
            cancelled = False

            while countdown > 0 and not start_requested['value']:
                # Vérifier si bouton Stop cliqué dans le GUI
                if gui.should_stop():
                    logger.warning("Arrêt demandé via l'interface avant le démarrage")
                    cancelled = True
                    break

                # Afficher countdown (écrase la ligne)
                print(f"\rDémarrage auto dans {countdown}s... ", end='', flush=True)
                time.sleep(1)
                countdown -= 1

            if cancelled:
                logger.info("Nettoyage...")

                # Attendre que le thread se termine proprement
                shutdown_flag['value'] = True
                input_thread.join(timeout=1.0)  # Attendre max 1 seconde

                env.close()
                gui.close()
                logger.info("Entrainement terminé")
                return

            # Message selon le mode de démarrage
            if start_requested['by_user']:
                logger.warning("Démarrage manuel (ENTRÉE pressée) !")
            else:
                logger.warning("⏰ Démarrage automatique (timeout) !")

            logger.warning("🚀 Lancement de l'entraînement...")

            # Clean up PID files now that training has started successfully
            if args.num_instances > 1:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Clean PIDs from vision/temp/
                vision_dir = os.path.join(script_dir, "vision")
                temp_dir = os.path.join(vision_dir, "temp")
                for i in range(args.num_instances):
                    pid_file = os.path.join(temp_dir, f"dolphin_pid_{i}.tmp")
                    try:
                        if os.path.exists(pid_file):
                            os.remove(pid_file)
                            logger.debug(f"Cleaned up PID file: {pid_file}")
                    except Exception as pid_cleanup_error:
                        logger.debug(f"Could not remove PID file {pid_file}: {pid_cleanup_error}")

            # Activer le bouton Stop de la GUI
            if hasattr(gui, 'stop_button'):
                gui.stop_button.config(state=tk.NORMAL)

        else:
            # MODE SANS GUI : input() classique et direct
            logger.warning("⏸️ Appuie sur ENTRÉE pour démarrer l'entraînement (Ctrl+C pour annuler)...")
            input()
            logger.warning("🚀 Démarrage de l'entraînement...")

    except KeyboardInterrupt:
        logger.info("CANCELLED BEFORE TRAINING START (Ctrl+C)")
        logger.info("Cleaning up...")

        # Close environment first
        try:
            if 'env' in locals() and env is not None:
                logger.info("Closing environment...")
                env.close()
        except Exception as env_close_error:
            logger.error(f"Error closing environment: {env_close_error}")

        # Close GUI
        try:
            if 'gui' in locals() and gui is not None:
                logger.info("Closing GUI...")
                gui.close()
        except Exception as gui_close_error:
            logger.error(f"Error closing GUI: {gui_close_error}")

        # CLEANUP DOLPHIN INSTANCES
        try:
            # Comprehensive PID recovery for all cancellation scenarios
            dolphin_pids = []

            # Method 1 : If no PIDs yet, try reading from PID files
            # At this stage, PIDs exist in temp files but may not be in allocation_result yet
            # This handles early cancellation before PIDs were stored in allocation_result
            if args.num_instances > 1:
                logger.debug("Reading PIDs from temporary files...")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                vision_dir = os.path.join(script_dir, "vision")
                temp_dir = os.path.join(vision_dir, "temp")

                for i in range(args.num_instances):
                    pid_file = os.path.join(temp_dir, f"dolphin_pid_{i}.tmp")
                    try:
                        if os.path.exists(pid_file):
                            with open(pid_file, 'r') as f:
                                pid_str = f.read().strip()
                                if pid_str and pid_str != "-1":
                                    pid = int(pid_str)
                                    if pid > 0 and pid not in dolphin_pids:
                                        dolphin_pids.append(pid)
                                        logger.debug(f"Recovered PID {pid} from file {pid_file}")
                            # Clean up PID file
                            os.remove(pid_file)
                            logger.debug(f"Removed PID file: {pid_file}")
                    except Exception as pid_recovery_error:
                        logger.debug(f"Could not recover PID from {pid_file}: {pid_recovery_error}")

            # Fallbacks : Try allocation_result (may be empty at this stage)
            if 'allocation_result' in locals() and allocation_result is not None:
                pids_from_alloc = allocation_result.get('dolphin_pids', [])
                if pids_from_alloc:
                    dolphin_pids.extend(pids_from_alloc)
                    logger.debug(f"Found {len(pids_from_alloc)} PIDs from allocation_result")

            # Filter valid PIDs and remove duplicates
            valid_pids = list(set([pid for pid in dolphin_pids if pid is not None and pid > 0]))

            if valid_pids:
                logger.warning(f"Closing {len(valid_pids)} Dolphin instance(s)...")
                cleanup_dolphin_processes(valid_pids, emergency=False)

                # Verify cleanup
                import psutil
                time.sleep(0.5)
                still_running = [pid for pid in valid_pids if psutil.pid_exists(pid)]

                if not still_running:
                    logger.info("All Dolphin instances closed")
                else:
                    logger.warning(f"{len(still_running)} instance(s) still running")
                    logger.warning("Please close them manually if needed")
            else:
                logger.debug("No Dolphin instances to cleanup")
                logger.debug("If Dolphin windows are open, they may need manual closure")

        except Exception as dolphin_cleanup_error:
            logger.error(f"Error cleaning up Dolphin: {dolphin_cleanup_error}")
            import traceback
            traceback.print_exc()

        logger.info("Cleanup complete")

        # Mark cleanup done
        _cleanup_done = True

        return

    try:
        logger.warning("🚀 Démarrage entraînement...")

        # Utiliser --debug-steps si fourni
        timesteps_to_train = args.debug_steps if args.debug_steps else args.timesteps

        if args.debug_steps:
            logger.info(f"MODE DEBUG : Entraînement limité à {args.debug_steps} steps")

        # ====================================================================
        # CAS 1 & 2 : 1 seul agent
        # ====================================================================
        if len(agents) == 1:
            agents[0].learn(
                total_timesteps=timesteps_to_train,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False
            )

        # ====================================================================
        # CAS 3 : Multi-agents avec scheduler
        # ====================================================================
        else:
            logger.info("Mode multi-agents : Boucle custom")

            # Vérifier disponibilité
            if not MULTI_AGENT_AVAILABLE:
                logger.error("Multi-agent non disponible")
                if gui:
                    gui.close()
                env.close()
                return

            # DISTINCTION GENETIC vs AUTRES MODES
            if args.multi_agent_mode == 'genetic':
                logger.info("")
                logger.info("=" * 70)
                logger.info("MODE GÉNÉTIQUE ACTIVÉ")
                logger.info("=" * 70)

                if GeneticTrainer is None:
                    logger.error("GeneticTrainer non disponible")
                    if gui:
                        gui.close()
                    env.close()
                    return

                # Créer trainer génétique
                genetic_trainer = GeneticTrainer(
                    agents=agents,
                    env=env,
                    elite_ratio=args.genetic_elite_ratio,
                    mutation_rate=args.genetic_mutation_rate,
                    episodes_per_eval=10,
                )

                # Lancer entraînement
                results = genetic_trainer.train(
                    num_generations=args.genetic_generations,
                    progress_bar=True
                )

                # Les meilleurs agents sont dans results['final_agents']
                agents = results['final_agents']

            else:
                # Modes : independent, round_robin, majority_vote
                try:
                    from utils.multi_agent_trainer import MultiAgentTrainer

                    # Creer trainer
                    trainer = MultiAgentTrainer(
                        agents=agents,
                        env=env,
                        scheduler=scheduler,
                        steps_per_agent=args.steps_per_agent,
                        callback=callbacks[0] if callbacks else None, # Premier callbacks
                        scenario=str(scenario),
                        allocation=allocation_result['allocation'],
                    )

                    # Log demarrage entraînement
                    logger.info("")
                    logger.info("🚀 Démarrage entraînement multi-agents")
                    logger.info(f"Mode : {args.multi_agent_mode}")
                    logger.info(f"Steps total : {timesteps_to_train:,}")
                    logger.info("")

                    # Lancer entraînement
                    trainer.train(
                        total_timesteps=timesteps_to_train,
                        progress_bar=True
                    )

                except ImportError as import_trainer_error:
                    logger.error(f"MultiAgentTrainer non disponible : {import_trainer_error}")
                    logger.error("Utilise 1 agent pour l'instant")

                    # Fallback : entraîner seulement le premier agent
                    logger.warning("Fallback : Entraînement de l'agent 0 uniquement")
                    agents[0].learn(
                        total_timesteps=timesteps_to_train,
                        callback=callbacks,
                        progress_bar=True,
                        reset_num_timesteps=False
                    )

                except Exception as trainer_error:
                    logger.error(f"Erreur trainer multi-agents : {trainer_error}")
                    import traceback
                    traceback.print_exc()

                    # Logger l'erreur
                    training_logger.log_error(trainer_error, context="Multi-agent training")
                    raise

        # Afficher la carte finale
        if hasattr(env, 'env_method'):
            # Si VecEnv
            try:
                if hasattr(env, 'get_attr'):
                    reward_calc = env.get_attr('reward_calc')[0]
                    if reward_calc and hasattr(reward_calc, 'exploration_tracker'):
                        logger.info(reward_calc.exploration_tracker.get_detailed_map_info())
            except (AttributeError, KeyError, IndexError, TypeError):
                pass  # Environnement non compatible ou pas de tracker

        # ====================================================================
        # SAUVEGARDE FINALE
        # ====================================================================
        if len(agents) == 1:
            # 1 seul agent : sauvegarde simple
            final_path = os.path.join(models_dir, "final_model")
            agent = agents[0]
            agent.save(final_path)
        else:
            # Plusieurs agents : sauvegarder chacun
            logger.warning(f"💾 Sauvegarde de {len(agents)} agents...")
            for agent_id, agent in enumerate(agents):  # ← CHANGEMENT ICI
                final_path = os.path.join(models_dir, f"final_model_agent_{agent_id}")
                agent.save(final_path)
                logger.warning(f"   Agent {agent_id} : {final_path}.zip ({agent.num_timesteps:,} steps)")

            # Définir pour compatibilité avec le finally
            final_path = models_dir
            agent = agents[0] if agents else None

        # Sauvegarder aussi le VecNormalize
        try:
            vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")
            env.save(vec_normalize_path)
            logger.info(f"VecNormalize sauvegardé : {vec_normalize_path}")
        except (OSError, AttributeError) as VecNormalize_save_error:
            training_logger.log_error(VecNormalize_save_error, context="Sauvegarde VecNormalize final")
            logger.error(f"Impossible de sauvegarder VecNormalize : {VecNormalize_save_error}")

        logger.warning("=" * 70)
        logger.warning("ENTRAÎNEMENT TERMINÉ")
        logger.warning("=" * 70)
        logger.warning(f"Modèle sauvegardé : {final_path}.zip")
        logger.warning(f"Total timesteps : {agent.num_timesteps:,}")


    except KeyboardInterrupt:
        # LOGGER L'INTERRUPTION
        training_logger.log_warning("Entraînement interrompu par l'utilisateur (Ctrl+C)")

        logger.warning("Interruption (Ctrl+C)")

        # Sauvegarde d'urgence - vérifier que l'agent existe
        if len(agents) > 0:
            try:
                if len(agents) == 1:
                    # 1 seul agent
                    agent = agents[0]
                    interrupt_path = os.path.join(models_dir, "interrupted")
                    agent.save(interrupt_path)
                    logger.info(f"💾 Sauvegarde : {interrupt_path}.zip")
                    logger.info(f"Timesteps : {agent.num_timesteps:,}")
                    logger.info(f"Pour reprendre : --resume {interrupt_path}.zip")
                else:
                    # Multi-agents
                    logger.info(f"💾 Sauvegarde de {len(agents)} agents interrompus...")
                    for agent_id, agent in enumerate(agents):
                        interrupt_path = os.path.join(models_dir, f"interrupted_agent_{agent_id}")
                        agent.save(interrupt_path)
                        logger.info(f"   Agent {agent_id} : {interrupt_path}.zip ({agent.num_timesteps:,} steps)")
            except Exception as save_interrupt_error:
                logger.error(f"Erreur sauvegarde interruption : {save_interrupt_error}")
                training_logger.log_error(save_interrupt_error, context="Sauvegarde interruption")
        else:
            logger.warning("Aucun agent à sauvegarder (interruption avant création)")

        # Sauvegarder VecNormalize
        try:
            vec_normalize_path = os.path.join(models_dir, "interrupted_vec_normalize.pkl")
            env.save(vec_normalize_path)
            logger.info(f"VecNormalize : {vec_normalize_path}")
        except (OSError, AttributeError) as VecNormalize_interrupt_error:
            training_logger.log_error(VecNormalize_interrupt_error, context="Sauvegarde VecNormalize interrupted")
            logger.error(f"Erreur sauvegarde VecNormalize : {VecNormalize_interrupt_error}")

    except Exception as training_error:
        # Logger l'erreur
        training_logger.log_error(training_error, context="Boucle d'entraînement")

        logger.error(f"ERREUR entraînement : {training_error}")
        import traceback
        traceback.print_exc()

        # Sauvegarde d'urgence - vérifier que l'agent existe
        if len(agents) > 0:
            try:
                if len(agents) == 1:
                    # 1 seul agent
                    agent = agents[0]
                    error_path = os.path.join(models_dir, "error")
                    agent.save(error_path)
                    logger.error(f"Sauvegarde urgence : {error_path}.zip")
                    logger.error(f"Timesteps : {agent.num_timesteps:,}")
                else:
                    # Multi-agents
                    logger.error(f"💾 Sauvegarde urgence de {len(agents)} agents...")
                    for agent_id, agent in enumerate(agents):
                        error_path = os.path.join(models_dir, f"error_agent_{agent_id}")
                        agent.save(error_path)
                        logger.error(f"   Agent {agent_id} : {error_path}.zip")
            except Exception as emergency_save_error:
                logger.error(f"Impossible de sauvegarder : {emergency_save_error}")
                training_logger.log_error(emergency_save_error, context="Sauvegarde d'urgence")
        else:
            logger.warning("Aucun agent à sauvegarder (erreur avant création)")

    finally:
        # Mark cleanup as done to prevent atexit handler from running
        logger.info("Cleanup started...")

        # PRIORITY 0: Stop input thread FIRST to avoid stdin blocking
        try:
            if 'shutdown_flag' in locals() and shutdown_flag is not None:
                logger.debug("Signaling input thread to stop...")
                shutdown_flag['value'] = True

            if 'input_thread' in locals() and input_thread is not None:
                if input_thread.is_alive():
                    logger.debug("Waiting for input thread to finish...")
                    # Give thread time to exit its loop (checks flag every 0.1s)
                    time.sleep(0.2)
                    input_thread.join(timeout=2.0)

                    if input_thread.is_alive():
                        logger.warning("Input thread still alive after timeout - forcing cleanup")
                    else:
                        logger.debug("Input thread terminated cleanly")
        except Exception as thread_cleanup_error:
            logger.debug(f"Error stopping input thread: {thread_cleanup_error}")

        # PRIORITY 1: Close OpenCV windows (rtvision) to unblock waitKey
        try:
            if args.rtvision:
                logger.info("Closing OpenCV windows (rtvision)...")
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Process window close events
                logger.debug("OpenCV windows closed")
        except Exception as cv2_close_error:
            logger.debug(f"Error closing OpenCV windows: {cv2_close_error}")

        # Track if Dolphin instances were launched and cleanup status
        had_dolphin_instances = False
        dolphin_cleanup_successful = False

        # PRIORITY 2: Stop frame capture reconnection attempts before closing Dolphin
        #             to avoid log error spam
        try:
            if 'env' in locals() and env is not None:
                # Signal all frame captures to stop trying to reconnect
                if hasattr(env, 'env_method'):
                    # VecEnv case (multi-instance)
                    try:
                        frame_captures = env.env_method('get_frame_capture')
                        shutdown_count = 0
                        for fc in frame_captures:
                            if fc and hasattr(fc, 'shutdown'):
                                fc.shutdown()
                                shutdown_count += 1
                        if shutdown_count > 0:
                            logger.debug(f"Signaled {shutdown_count} frame captures to stop reconnection")
                    except (AttributeError, IndexError, TypeError) as fc_method_error:
                        logger.debug(f"Could not call env_method for frame capture: {fc_method_error}")
                elif hasattr(env, 'frame_capture'):
                    # Single env case
                    if hasattr(env.frame_capture, 'shutdown'):
                        env.frame_capture.shutdown()
                        logger.debug("Signaled single frame capture to stop reconnection")
        except Exception as fc_shutdown_error:
            logger.debug(f"Error signaling frame capture shutdown: {fc_shutdown_error}")

        # PRIORITY 3: Close Dolphin instances (after stopping frame capture)
        try:
            # Check if allocation_result exists and has PIDs
            if 'allocation_result' in locals() and allocation_result is not None:
                dolphin_pids = allocation_result.get('dolphin_pids', [])

                # Filter out None and invalid PIDs
                valid_pids = [pid for pid in dolphin_pids if pid is not None and pid > 0]

                if valid_pids:
                    had_dolphin_instances = True
                    logger.info(f"Closing {len(valid_pids)} Dolphin instance(s)...")

                    # Call cleanup function
                    cleanup_dolphin_processes(valid_pids, emergency=False)

                    # Verify cleanup success by checking if processes still exist
                    import psutil
                    still_running = []
                    for pid in valid_pids:
                        try:
                            if psutil.pid_exists(pid):
                                still_running.append(pid)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as pid_check_error:
                            # NoSuchProcess: Process terminated between check and access
                            # AccessDenied: Insufficient permissions (shouldn't happen for pid_exists)
                            # OSError: System-level error (rare)
                            logger.debug(f"Could not verify PID {pid}: {pid_check_error}")
                            pass

                    if not still_running:
                        dolphin_cleanup_successful = True
                        logger.info("All Dolphin instances closed successfully")
                    else:
                        dolphin_cleanup_successful = False
                        logger.warning(f"{len(still_running)} Dolphin instance(s) still running: {still_running}")
                else:
                    logger.debug("No valid Dolphin PIDs to cleanup")
                    # No instances to clean = success by default
                    dolphin_cleanup_successful = True
            else:
                logger.debug("allocation_result not found - no Dolphin instances launched")
                # No allocation_result = single instance mode or no instances
                dolphin_cleanup_successful = True

        except Exception as dolphin_cleanup_error:
            logger.error(f"Error cleaning up Dolphin: {dolphin_cleanup_error}")
            import traceback
            traceback.print_exc()
            dolphin_cleanup_successful = False

        # Clean up remaining PID files in temp/ directory
        try:
            if args.num_instances > 1:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                vision_dir = os.path.join(script_dir, "vision")
                temp_dir = os.path.join(vision_dir, "temp")

                if os.path.exists(temp_dir):
                    pid_files_cleaned = 0
                    for i in range(args.num_instances):
                        pid_file = os.path.join(temp_dir, f"dolphin_pid_{i}.tmp")
                        if os.path.exists(pid_file):
                            try:
                                os.remove(pid_file)
                                pid_files_cleaned += 1
                            except Exception as pid_remove_error:
                                logger.debug(f"Could not remove {pid_file}: {pid_remove_error}")

                    if pid_files_cleaned > 0:
                        logger.debug(f"Cleaned up {pid_files_cleaned} PID file(s) from temp/")

                    # Try to remove temp directory if empty
                    try:
                        if not os.listdir(temp_dir):
                            os.rmdir(temp_dir)
                            logger.debug("Removed empty temp/ directory")
                    except OSError:
                        pass  # Directory not empty or other issue, leave it
        except Exception as temp_cleanup_error:
            logger.debug(f"Error cleaning temp directory: {temp_cleanup_error}")

        # Warn user ONLY if:
        # 1. Dolphin instances were actually launched (had_dolphin_instances = True)
        # 2. AND cleanup failed (dolphin_cleanup_successful = False)
        if had_dolphin_instances and not dolphin_cleanup_successful:
            logger.error("")
            logger.error("=" * 70)
            logger.error("DOLPHIN CLEANUP FAILED")
            logger.error("=" * 70)
            logger.error("Some Dolphin instances may still be running")
            logger.error("Please close them manually:")
            logger.error("  1. Open Task Manager (Ctrl+Shift+Esc)")
            logger.error("  2. Find 'Dolphin.exe' processes")
            logger.error("  3. End task for each instance (sorry)")
            logger.error("=" * 70)
            logger.error("")

        # Clean up controller first
        try:
            if 'env' in locals():
                controller = None

                # Method 1: Via get_attr (VecEnv)
                if hasattr(env, 'get_attr'):
                    try:
                        controllers = env.get_attr('controller')
                        # Unfold nested lists
                        while isinstance(controllers, list) and len(controllers) > 0:
                            controllers = controllers[0]
                        controller = controllers
                    except (AttributeError, IndexError):
                        pass

                # Method 2: Direct access (fallback)
                if controller is None and hasattr(env, 'envs'):
                    try:
                        controller = env.envs[0].controller
                    except (AttributeError, IndexError):
                        pass

                # Cleanup if found
                if controller and hasattr(controller, 'cleanup'):
                    logger.info("🎮 Cleaning up controller...")
                    controller.cleanup()
                    logger.info("Controller cleaned up")
                else:
                    logger.debug("Controller not found for cleanup")

        except Exception as cleanup_error:
            training_logger.log_error(cleanup_error, context="Controller cleanup")
            logger.error(f"Error cleaning up controller: {cleanup_error}")

        except (AttributeError, IndexError, TypeError) as cleanup_error:
            training_logger.log_error(cleanup_error, context="Controller cleanup")
            logger.error(f"Unable to clean up controller: {cleanup_error}")

        # CLOSE LOGGER
        try:
            if 'training_logger' in locals() and training_logger is not None:
                training_logger.close()
        except Exception as logger_error:
            logger.error(f"Error closing logger: {logger_error}")

        # CLOSE ENVIRONMENT
        try:
            if 'env' in locals() and env is not None:
                logger.info("🌍 Closing environment...")
                env.close()
        except Exception as env_error:
            logger.error(f"Error closing env: {env_error}")

        # CLOSE GUI
        try:
            if 'gui' in locals() and gui is not None:
                logger.warning("Closing interface...")
                gui.close()
        except Exception as gui_error:
            logger.error(f"Error closing GUI: {gui_error}")

        _cleanup_done = True

        # Also mark global cleanup as done
        global _global_cleanup_done
        _global_cleanup_done = True

        logger.warning("Terminated")

# ============================================================
# LANCER ENTRAINEMENT
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Gérer Ctrl+C proprement
        logger.warning("\n\nInterruption (Ctrl+C)")
        logger.warning("Programme arrêté proprement")
        sys.exit(0)  # Exit code 0 = succès
    except Exception as critical_error:
        logger.error(f"\n💥 ERREUR CRITIQUE : {critical_error}")
        traceback.print_exc()
        sys.exit(1)
