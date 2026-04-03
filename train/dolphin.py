"""
dolphin.py — Dolphin emulator process management.

Handles launching instances via PowerShell, game-window detection,
path resolution, PID recovery, process cleanup, and emergency signal handling.

Exports:
    resolve_dolphin_path(cli_path)           -> str
    auto_detect_or_prompt_dolphin_path()     -> str
    launch_dolphin_instances_via_powershell() -> bool
    wait_for_dolphin_windows()               -> bool
    is_mhtri_window_open()                   -> bool
    read_pids_from_temp(n)                   -> list
    clean_pid_files(n)
    close_existing_dolphin_instances()       -> bool
    cleanup_dolphin_processes(pids, emergency)
    register_signal_handlers()
    emergency_signal_handler(signum, frame)

Module-level state:
    global_dolphin_pids  — PID list shared with the signal handler
    global_cleanup_done  — flag to prevent double cleanup
"""

import os
import sys
import json
import time
import signal
import subprocess
import psutil

import win32process
import win32gui

from info.module_logger import get_module_logger

logger = get_module_logger('train.dolphin')

# ======================================================================
#  MODULE STATE
# ======================================================================

global_dolphin_pids: list = []
global_cleanup_done: bool = False

# Relative path (from project root) to the persisted Dolphin config file
_CONFIG_FILE_REL = os.path.join("config", "dolphin_path_config.json")
# Relative path (from project root) to the persisted ROM config file
_ROM_CONFIG_FILE_REL = os.path.join("config", "rom_path_config.json")


# ======================================================================
#  PATH RESOLUTION
# ======================================================================

def _project_root() -> str:
    """Return the project root (one level above this package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_saved_dolphin_path() -> str | None:
    """Load the previously saved Dolphin path from the config file."""
    config_file = os.path.join(_project_root(), _CONFIG_FILE_REL)
    if not os.path.exists(config_file):
        return None
    try:
        with open(config_file, 'r') as f:
            saved = json.load(f)
            path = saved.get('dolphin_path')
            if path and os.path.exists(path):
                return path
    except (json.JSONDecodeError, OSError, KeyError) as dolphin_config_load_error:
        logger.warning(f"Could not load saved Dolphin config: {dolphin_config_load_error}")
    return None


def _save_dolphin_path(path: str) -> None:
    """Persist the Dolphin path for the next session."""
    config_dir = os.path.join(_project_root(), "config")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "dolphin_path_config.json")
    try:
        with open(config_file, 'w') as f:
            json.dump({'dolphin_path': path}, f, indent=2)
        logger.info(f"Dolphin path saved to {config_file}")
    except (OSError, TypeError) as dolphin_config_save_error:
        logger.warning(f"Could not save Dolphin config: {dolphin_config_save_error}")


def resolve_dolphin_path(cli_path: str | None) -> str:
    """
    Resolve the Dolphin path using this priority order:
      1. Explicit CLI argument
      2. Previously saved config file
      3. Auto-detection / interactive prompt

    Returns a valid path to the Dolphin folder (containing Dolphin.exe).
    """
    if cli_path is not None:
        return cli_path

    saved = _load_saved_dolphin_path()
    if saved is not None:
        logger.info(f"Using saved Dolphin path: {saved}")
        return saved

    logger.info("Dolphin path not provided — auto-detecting...")
    detected = auto_detect_or_prompt_dolphin_path()
    _save_dolphin_path(detected)
    return detected


def auto_detect_or_prompt_dolphin_path() -> str:
    """
    Search common installation locations for Dolphin.
    Falls back to an interactive prompt if nothing is found.

    Returns a valid folder path containing Dolphin.exe, or exits.
    """
    logger.info("Auto-detecting Dolphin path...")

    common_paths = [
        "./Dolphin-x64", "./Dolphin",
        "../Dolphin-x64", "../Dolphin",
        os.path.join(os.path.expanduser("~"), "Documents", "Dolphin-x64"),
        os.path.join(os.path.expanduser("~"), "Documents", "Dolphin"),
        "C:/Program Files/Dolphin-x64",
        "C:/Program Files (x86)/Dolphin-x64",
        os.path.join(os.path.expanduser("~"), "Desktop", "Dolphin-x64"),
        os.path.join(os.path.expanduser("~"), "Desktop", "Dolphin"),
    ]

    for path in common_paths:
        if os.path.isdir(path):
            exe = os.path.join(path, "Dolphin.exe")
            if os.path.isfile(exe):
                logger.info(f"Found Dolphin at: {path}")
                return path
        elif os.path.isfile(path) and path.endswith("Dolphin.exe"):
            logger.info(f"Found Dolphin.exe at: {path}")
            return os.path.dirname(path)

    # Interactive prompt
    logger.warning("Dolphin not found automatically")
    logger.warning("Please provide the path to Dolphin (folder or .exe):")

    while True:
        try:
            user_path = input("Dolphin path: ").strip().strip('"').strip("'")
            if not user_path:
                continue

            user_path = os.path.abspath(user_path)

            if os.path.isdir(user_path):
                if os.path.isfile(os.path.join(user_path, "Dolphin.exe")):
                    return user_path
                logger.error(f"Folder does not contain Dolphin.exe: {user_path}")
            elif os.path.isfile(user_path) and user_path.endswith("Dolphin.exe"):
                return os.path.dirname(user_path)
            else:
                logger.error(f"Invalid path: {user_path}")
        except KeyboardInterrupt:
            logger.warning("Cancelled by user (Ctrl+C)")
            sys.exit(0)
        except (OSError, ValueError) as dolphin_prompt_error:
            logger.error(f"Error: {dolphin_prompt_error}")

    # Unreachable - while True always returns or calls sys.exit()
    # Explicit return to satisfy type checker
    return ""


# ======================================================================
#  ROM PATH RESOLUTION
# ======================================================================

_ROM_EXTENSIONS = (".rvz", ".iso", ".wbfs", ".gcm")
_ROM_KEYWORDS = ("monster", "hunter", "mhtri", "rmhp")


def resolve_rom_path(cli_rom_path: str | None, dolphin_path: str) -> str | None:
    """
    Resolve the ROM path using this priority order:
      1. Explicit CLI argument (--rom-path)
      2. Previously saved config file (config/rom_path_config.json)
      3. Auto-detection relative to the Dolphin folder
      4. Interactive prompt

    Returns a valid ROM file path, or None if user cancels.
    """
    # 1. Explicit CLI path
    if cli_rom_path is not None:
        cli_rom_path = os.path.abspath(cli_rom_path)
        if os.path.isfile(cli_rom_path):
            _save_rom_path(cli_rom_path)
            logger.info(f"Using CLI ROM path: {cli_rom_path}")
            return cli_rom_path
        logger.warning(f"ROM path provided but not found: {cli_rom_path}")

    # 2. Saved config
    saved = _load_saved_rom_path()
    if saved is not None:
        logger.info(f"Using saved ROM path: {saved}")
        return saved

    # 3. Auto-detect relative to Dolphin directory
    dolphin_dir = os.path.dirname(dolphin_path) if os.path.isfile(dolphin_path) else dolphin_path
    detected = _auto_detect_rom(dolphin_dir)
    if detected is not None:
        _save_rom_path(detected)
        return detected

    # 4. Interactive prompt (same pattern as Dolphin path)
    prompted = _prompt_rom_path()
    if prompted is not None:
        _save_rom_path(prompted)
    return prompted


def _auto_detect_rom(dolphin_dir: str) -> str | None:
    """Search for Monster Hunter Tri ROM relative to the Dolphin directory.

    Searches at multiple ancestor levels to handle various folder structures:
      Dolphin-x64/               → parent
      IA_jeux/Dolphin-x64/       → grandparent
      Emulateur/IA_jeux/Dolphin/ → great-grandparent
    """
    ancestors = [dolphin_dir]
    current = dolphin_dir
    # Collect up to 3 parent levels
    for _ in range(3):
        parent = os.path.dirname(current)
        if parent == current:  # filesystem root
            break
        ancestors.append(parent)
        current = parent

    search_roots = []
    for ancestor in ancestors:
        for subdir in ("Jeux", "Games", "ROMs"):
            candidate = os.path.join(ancestor, subdir)
            if candidate not in search_roots:
                search_roots.append(candidate)

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                if fname.lower().endswith(_ROM_EXTENSIONS):
                    if any(kw in fname.lower() for kw in _ROM_KEYWORDS):
                        found = os.path.join(dirpath, fname)
                        logger.info(f"ROM auto-detected: {found}")
                        return found
    return None


def _prompt_rom_path() -> str | None:
    """Interactive prompt for ROM path (fallback when auto-detect fails)."""
    print()
    print("=" * 60)
    print("  ROM NOT FOUND")
    print("=" * 60)
    print("  Monster Hunter Tri ROM was not auto-detected.")
    print("  Please provide the full path to the ROM file.")
    print()
    print("  Accepted formats: .rvz, .iso, .wbfs, .gcm")
    print("  Example: C:\\Games\\MHtri\\MonsterHunterTri.rvz")
    print()
    print("  NOTE: Provide the path to the FILE, not just the folder.")
    print("        e.g. ...\\Jeux\\MHtri\\MonsterHunterTri.rvz")
    print("        NOT  ...\\Jeux\\MHtri\\")
    print("=" * 60)

    while True:
        try:
            user_path = input("ROM path: ").strip().strip('"').strip("'")
            if not user_path:
                continue

            user_path = os.path.abspath(user_path)

            # If user gave a directory, search for ROM files inside it
            if os.path.isdir(user_path):
                found_in_dir = _auto_detect_rom_in_dir(user_path)
                if found_in_dir is not None:
                    print(f"  Found ROM in directory: {found_in_dir}")
                    logger.info(f"ROM found in provided directory: {found_in_dir}")
                    return found_in_dir
                print(f"  No ROM file found in: {user_path}")
                print(f"  Provide the full path to the .rvz/.iso/.wbfs file, not just the folder.")
                continue

            if os.path.isfile(user_path):
                if user_path.lower().endswith(_ROM_EXTENSIONS):
                    logger.info(f"ROM path accepted: {user_path}")
                    return user_path
                print(f"  Not a supported ROM format ({', '.join(_ROM_EXTENSIONS)}): {user_path}")
            else:
                print(f"  File not found: {user_path}")
        except KeyboardInterrupt:
            logger.warning("ROM path prompt cancelled by user (Ctrl+C)")
            return None
        except (OSError, ValueError) as rom_prompt_error:
            logger.error(f"Error: {rom_prompt_error}")


def _auto_detect_rom_in_dir(directory: str) -> str | None:
    """Search for a Monster Hunter Tri ROM file inside a specific directory."""
    for dirpath, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(_ROM_EXTENSIONS):
                if any(kw in fname.lower() for kw in _ROM_KEYWORDS):
                    return os.path.join(dirpath, fname)
    return None


def _load_saved_rom_path() -> str | None:
    """Load the previously saved ROM path from the config file."""
    config_file = os.path.join(_project_root(), _ROM_CONFIG_FILE_REL)
    if not os.path.exists(config_file):
        return None
    try:
        with open(config_file, 'r') as f:
            saved = json.load(f)
            path = saved.get('rom_path')
            if path and os.path.isfile(path):
                return path
            if path:
                logger.warning(f"Saved ROM path no longer exists: {path}")
    except (json.JSONDecodeError, OSError, KeyError) as rom_config_load_error:
        logger.warning(f"Could not load saved ROM config: {rom_config_load_error}")
    return None


def _save_rom_path(path: str) -> None:
    """Persist the ROM path for the next session."""
    config_dir = os.path.join(_project_root(), "config")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "rom_path_config.json")
    try:
        with open(config_file, 'w') as f:
            json.dump({'rom_path': path}, f, indent=2)
        logger.info(f"ROM path saved to {config_file}")
    except (OSError, TypeError) as rom_config_save_error:
        logger.warning(f"Could not save ROM config: {rom_config_save_error}")

# ======================================================================
#  LAUNCH
# ======================================================================

def launch_dolphin_instances_via_powershell(
    num_instances: int,
    dolphin_path: str,
    minimize_dolphin: bool = True,
    minimize_game: bool = False,
    rom_path: str | None = None,
) -> bool:
    """
    Launch Dolphin instances via the PowerShell helper script.

    Args:
        num_instances:    Number of instances to start.
        dolphin_path:     Path to Dolphin.exe or its parent folder.
        minimize_dolphin: Minimize the Dolphin menu windows on launch.
        minimize_game:    Minimize the game render windows on launch.

    Returns True if the PowerShell script ran successfully.
    """
    root = _project_root()

    # Ensure temp directory for PID files
    temp_dir = os.path.join(root, "vision", "temp")
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except Exception as exc:
        logger.error(f"Failed to create temp directory: {exc}")
        return False

    _ensure_gitignore(temp_dir)
    _ensure_gitignore(os.path.join(root, "vision", "debug"),
                      patterns=["*.png", "*.jpg"])

    # Normalize and validate the Dolphin path
    dolphin_path = os.path.abspath(dolphin_path)
    if os.path.isdir(dolphin_path):
        exe = os.path.join(dolphin_path, "Dolphin.exe")
        if not os.path.isfile(exe):
            logger.error(f"Dolphin.exe not found in folder: {dolphin_path}")
            return False
        dolphin_path = exe

    if not os.path.isfile(dolphin_path):
        logger.error(f"Dolphin.exe not found: {dolphin_path}")
        return False

    dolphin_dir = os.path.dirname(dolphin_path)

    # Locate the PowerShell launch script
    ps_script = _find_ps_script(dolphin_dir, root)
    if ps_script is None:
        return False

    # Build the command — parameter names must match the .ps1 param() block exactly
    ps_script = os.path.normpath(ps_script)
    dolphin_path = os.path.normpath(dolphin_path)

    # Ensure PS1 has UTF-8 BOM — prevents parsing failures on French Windows systems
    _ensure_ps1_utf8_bom(ps_script)

    cmd = [
        "powershell.exe",
        "-ExecutionPolicy", "Bypass",
        "-NoProfile",
        "-File", ps_script,
        "-NumInstances", str(num_instances),
        "-NoGUI",
        "-DolphinExePath", dolphin_path,
        "-PidDirectory", os.path.normpath(temp_dir),
    ]
    # Pass ROM path if resolved by Python (avoids PS1 re-searching)
    if rom_path:
        cmd.extend(["-RomFilePath", rom_path])
    if minimize_dolphin:
        cmd.append("-MinimizeDolphin")
    if minimize_game:
        cmd.append("-MinimizeGame")

    # Dynamic timeout: base 10s + 10s per instance
    dynamic_timeout = 10 + (num_instances * 10)

    logger.info(f"Launching {num_instances} Dolphin instance(s)...")
    logger.debug(f"Command: {' '.join(cmd)}")

    ps_process = None  # Initialized before try
    try:
        ps_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding='utf-8',  # Read PowerShell output as UTF-8 (not system ANSI)
            errors='replace',  # Replace undecodable chars instead of crashing
            cwd=os.path.normpath(root),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        stdout, stderr = ps_process.communicate(timeout=dynamic_timeout)

        if ps_process.returncode != 0:
            logger.error(f"PowerShell exited with code {ps_process.returncode}")
            if stderr:
                for line in stderr.strip().split('\n')[:10]:
                    logger.error(f"  PS> {line}")
            return False

        # Log PowerShell stdout if present
        if stdout and stdout.strip():
            for line in stdout.strip().split('\n'):
                logger.debug(f"  PS> {line}")

        logger.info("PowerShell script completed successfully")
        return True


    except subprocess.TimeoutExpired:
        logger.error(f"PowerShell script timed out ({dynamic_timeout}s)")
        if ps_process:
            ps_process.kill()
        return False
    except FileNotFoundError:
        logger.error("PowerShell not found - is it installed?")
        return False
    except (OSError, ValueError, UnicodeDecodeError) as ps_popen_err:
        logger.error(f"Failed to run PowerShell script: {ps_popen_err}")
        return False

def _find_ps_script(dolphin_dir: str, project_root: str) -> str | None:
    """Search known locations for launch_dolphin_instances.ps1."""
    candidates = [
        os.path.join(dolphin_dir, "launch_dolphin_instances.ps1"),
        os.path.join(project_root, "launch_dolphin_instances.ps1"),
        os.path.abspath("launch_dolphin_instances.ps1"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            logger.debug(f"PowerShell script found: {path}")
            return path

    logger.error("PowerShell launch script not found. Searched:")
    for i, p in enumerate(candidates, 1):
        logger.error(f"  {i}. {p}")
    return None

def _ensure_ps1_utf8_bom(ps1_path: str) -> None:
    """
    Ensure the PowerShell script is saved as UTF-8 with BOM.

    PowerShell on French Windows defaults to CP1252/CP850 for .ps1 files
    without a BOM. This causes parsing failures when the script contains
    accented characters (ô, é, è, â...), producing 'missing string terminator'
    errors that crash the entire training launch.

    Writing UTF-8 BOM (0xEF 0xBB 0xBF) at the start of the file tells
    PowerShell explicitly to use UTF-8 encoding.
    """
    try:
        # Check if UTF-8 BOM already present — skip if so
        with open(ps1_path, 'rb') as _f:
            _raw = _f.read(3)
        if _raw == b'\xef\xbb\xbf':
            logger.debug(f"PS1 already has UTF-8 BOM: {ps1_path}")
            return

        # Try reading with multiple encodings (best-effort detection)
        _content: str | None = None
        for _enc in ('utf-8', 'cp1252', 'latin-1'):
            try:
                with open(ps1_path, 'r', encoding=_enc) as _f:
                    _content = _f.read()
                logger.debug(f"PS1 file decoded as {_enc}, adding UTF-8 BOM")
                break
            except UnicodeDecodeError:
                continue

        if _content is None:
            logger.warning(
                f"Could not detect encoding of PS1 file '{ps1_path}' — running as-is "
                "(PowerShell may fail to parse French characters)"
            )
            return

        # Rewrite file with UTF-8 BOM so PowerShell parses it correctly
        with open(ps1_path, 'w', encoding='utf-8-sig') as _f:
            _f.write(_content)
        logger.info(f"PS1 re-encoded as UTF-8 with BOM: {ps1_path}")

    except (OSError, PermissionError) as ps1_bom_os_err:
        logger.warning(f"Could not add BOM to PS1 file (OS error): {ps1_bom_os_err}")
    except UnicodeError as ps1_bom_unicode_err:
        logger.warning(f"Could not add BOM to PS1 file (encoding error): {ps1_bom_unicode_err}")

def _ensure_gitignore(directory: str, patterns: list[str] | None = None) -> None:
    """Create a .gitignore in *directory* that ignores all files."""
    os.makedirs(directory, exist_ok=True)
    gi = os.path.join(directory, ".gitignore")
    if os.path.exists(gi):
        return
    try:
        with open(gi, 'w') as f:
            for p in (patterns or ["*"]):
                f.write(f"{p}\n")
            f.write("!.gitignore\n")
    except Exception:
        pass


# ======================================================================
#  PID RECOVERY
# ======================================================================

def read_pids_from_temp(num_instances: int) -> list:
    """Read Dolphin PIDs from the temp files written by the PowerShell script."""
    temp_dir = os.path.join(_project_root(), "vision", "temp")
    pids = []
    for i in range(num_instances):
        pid_file = os.path.join(temp_dir, f"dolphin_pid_{i}.tmp")
        try:
            if os.path.exists(pid_file):
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                    pids.append(pid)
                    logger.info(f"Instance {i}: PID {pid}")
            else:
                logger.warning(f"Instance {i}: PID file not found")
                pids.append(None)
        except Exception as exc:
            logger.error(f"Instance {i}: error reading PID — {exc}")
            pids.append(None)
    return pids


def clean_pid_files(num_instances: int) -> None:
    """Remove temp PID files after a successful startup."""
    temp_dir = os.path.join(_project_root(), "vision", "temp")
    for i in range(num_instances):
        pid_file = os.path.join(temp_dir, f"dolphin_pid_{i}.tmp")
        try:
            if os.path.exists(pid_file):
                os.remove(pid_file)
        except Exception:
            pass
    # Remove the temp directory itself if empty
    try:
        if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except OSError:
        pass


# ======================================================================
#  WINDOW DETECTION
# ======================================================================

def _is_dolphin_process(hwnd) -> bool:
    """Check that the window belongs to an actual Dolphin.exe process."""
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        proc = psutil.Process(pid)
        return 'dolphin' in proc.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess,
            OSError, ValueError):
        return False


def is_mhtri_window_open() -> bool:
    """Return True if any visible Monster Hunter Tri / MHTri window exists
    AND belongs to an actual Dolphin.exe process."""
    found = []

    def _enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).lower()
            if title.startswith("mhtri") or "monster hunter" in title:
                # Verify the window actually belongs to Dolphin.exe
                if _is_dolphin_process(hwnd):
                    found.append(hwnd)
                else:
                    logger.debug(
                        f"Ignoring non-Dolphin window with matching title: "
                        f"'{win32gui.GetWindowText(hwnd)}'")
        return True

    win32gui.EnumWindows(_enum_cb, None)
    return len(found) > 0


def wait_for_dolphin_windows(
    num_instances: int,
    timeout: int = 60,
    check_interval: int = 10,
) -> bool:
    """
    Poll until *num_instances* Dolphin game windows are visible.

    Returns True if all windows were detected within *timeout* seconds.
    """
    logger.debug(f"Waiting for {num_instances} Dolphin window(s) "
                 f"(timeout={timeout}s, interval={check_interval}s)...")

    start = time.time()
    attempt = 0
    windows: list[dict] = []

    while time.time() - start < timeout:
        attempt += 1
        windows = []

        def _enum_cb(hwnd, wins):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                tl = title.lower()
                if tl.startswith("mhtri") or "monster hunter" in tl:
                    wins.append({'hwnd': hwnd, 'title': title})
            return True

        win32gui.EnumWindows(_enum_cb, windows)
        windows.sort(key=lambda w: w['title'])

        logger.debug(f"  Attempt {attempt}: {len(windows)}/{num_instances} windows")

        if len(windows) >= num_instances:
            logger.debug("All Dolphin windows detected")
            return True

        remaining = timeout - (time.time() - start)
        if remaining > check_interval:
            time.sleep(check_interval)

    logger.error(f"Timeout: only {len(windows)}/{num_instances} windows detected")
    return False


# ======================================================================
#  CLEANUP
# ======================================================================

def close_existing_dolphin_instances() -> bool:
    """Find and close any already-running Dolphin.exe processes.

    Returns True if none remain afterwards.
    """
    existing = []
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            if (proc.info['name']
                    and 'dolphin' in proc.info['name'].lower()
                    and proc.info['exe']
                    and 'dolphin.exe' in proc.info['exe'].lower()):
                existing.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not existing:
        logger.info("No existing Dolphin instances found")
        return True

    logger.warning(f"Found {len(existing)} existing Dolphin instance(s) — closing...")
    cleanup_dolphin_processes(existing, emergency=False)

    time.sleep(1.0)
    still_running = [pid for pid in existing if psutil.pid_exists(pid)]
    if still_running:
        logger.error(f"Failed to close PIDs: {still_running}")
        return False

    logger.info("All existing Dolphin instances closed successfully")
    return True


def cleanup_dolphin_processes(dolphin_pids: list, emergency: bool = False) -> None:
    """Terminate a list of Dolphin PIDs (graceful first, then force kill)."""
    if not dolphin_pids:
        return

    tag = "EMERGENCY" if emergency else "CLEANUP"
    logger.warning(f"[{tag}] Closing {len(dolphin_pids)} Dolphin process(es)...")

    closed = failed = 0
    for pid in dolphin_pids:
        if pid is None or pid < 0:
            continue
        try:
            if not psutil.pid_exists(pid):
                continue
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=3)
                closed += 1
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
                closed += 1
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            logger.error(f"Access denied for PID {pid}")
            failed += 1
        except Exception as exc:
            logger.error(f"Failed to close PID {pid}: {exc}")
            failed += 1

    logger.warning(f"[{tag}] Done — {closed} closed, {failed} failed")


# ======================================================================
#  SIGNAL HANDLER
# ======================================================================

def emergency_signal_handler(signum, _frame):
    """
    Global signal handler for SIGINT / SIGTERM.

    Suppresses frame_capture warnings, cleans up Dolphin processes,
    then re-raises KeyboardInterrupt so the training try/except can save models.
    """
    global global_cleanup_done

    if global_cleanup_done:
        return

    # Suppress capture warnings before killing Dolphin:
    # capture threads keep running briefly after windows die
    import logging as _logging
    _logging.getLogger('mh_frame_capture').setLevel(_logging.CRITICAL)
    _logging.getLogger('mh_dolphin_capture_dll').setLevel(_logging.CRITICAL)

    logger.warning(f"SIGNAL RECEIVED: {signal.Signals(signum).name}")

    if global_dolphin_pids:
        logger.warning(f"Emergency cleanup: {len(global_dolphin_pids)} instance(s)...")
        try:
            cleanup_dolphin_processes(global_dolphin_pids, emergency=True)
        except Exception as exc:
            logger.error(f"Emergency cleanup error: {exc}")

    global_cleanup_done = True
    raise KeyboardInterrupt


def register_signal_handlers() -> None:
    """Register SIGINT and SIGTERM to run the emergency signal handler."""
    signal.signal(signal.SIGINT, emergency_signal_handler)
    signal.signal(signal.SIGTERM, emergency_signal_handler)
    logger.info("Signal handlers registered (SIGINT, SIGTERM)")