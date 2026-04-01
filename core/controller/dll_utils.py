"""
dll_utils.py — Dolphin DLL injection and PID resolution utilities.

Extracted from the monolithic controller.py for better separation of concerns.
Handles:
    - Auto-locating or building dolphin_input_hook.dll
    - Finding Dolphin render windows by title → PID
    - Injecting the DLL via CreateRemoteThread(LoadLibraryA)
"""

import ctypes
import ctypes.wintypes
import os
import shutil
import subprocess
import time

import win32con
import win32gui
import win32process

from info.module_logger import get_module_logger
from core.controller.constants import (
    FILE_MAP_ALL_ACCESS, SHARED_MEM_SIZE,
)

logger = get_module_logger('dll_utils')

# ============================================================================
# kernel32 wrappers (explicit argtypes / restype for 64-bit correctness)
# ============================================================================
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

kernel32.OpenProcess.restype      = ctypes.wintypes.HANDLE
kernel32.OpenProcess.argtypes     = [
    ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD]

kernel32.VirtualAllocEx.restype   = ctypes.wintypes.LPVOID
kernel32.VirtualAllocEx.argtypes  = [
    ctypes.wintypes.HANDLE, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.wintypes.DWORD, ctypes.wintypes.DWORD]

kernel32.WriteProcessMemory.restype  = ctypes.wintypes.BOOL
kernel32.WriteProcessMemory.argtypes = [
    ctypes.wintypes.HANDLE, ctypes.c_void_p,
    ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]

kernel32.CreateRemoteThread.restype  = ctypes.wintypes.HANDLE
kernel32.CreateRemoteThread.argtypes = [
    ctypes.wintypes.HANDLE, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.wintypes.DWORD, ctypes.POINTER(ctypes.wintypes.DWORD)]

kernel32.WaitForSingleObject.restype  = ctypes.wintypes.DWORD
kernel32.WaitForSingleObject.argtypes = [
    ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD]

kernel32.GetModuleHandleA.restype  = ctypes.wintypes.HANDLE
kernel32.GetModuleHandleA.argtypes = [ctypes.c_char_p]

kernel32.GetProcAddress.restype  = ctypes.c_void_p
kernel32.GetProcAddress.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_char_p]

kernel32.CloseHandle.restype  = ctypes.wintypes.BOOL
kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]

kernel32.OpenFileMappingA.restype  = ctypes.wintypes.HANDLE
kernel32.OpenFileMappingA.argtypes = [
    ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.c_char_p]

kernel32.MapViewOfFile.restype  = ctypes.c_void_p
kernel32.MapViewOfFile.argtypes = [
    ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD, ctypes.wintypes.DWORD, ctypes.c_size_t]

kernel32.UnmapViewOfFile.restype  = ctypes.wintypes.BOOL
kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]


# ============================================================================
# Shared memory name
# ============================================================================

def shared_mem_name(pid: int) -> str:
    """Return the shared memory region name for a given Dolphin PID."""
    return f"DolphinInputHook_SharedMem_{pid}"


# ============================================================================
# Project root resolution
# ============================================================================

def _project_root() -> str:
    """Resolve project root: core/controller/dll_utils.py → go up two levels."""
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ============================================================================
# DLL auto-locate / auto-build
# ============================================================================

def ensure_dll(dll_path: str = None) -> str:
    """
    Return the path to dolphin_input_hook.dll, building it if necessary.

    Search order:
        1. Explicit dll_path (if provided)
        2. vision/dolphin_input_hook.dll
        3. <root>/dolphin_input_hook.dll
        4. hook/target/release/dolphin_input_hook.dll

    Auto-build:
        If none exist, runs `cargo build --release` in hook/.
        Copies the result to vision/ for future runs.

    Raises FileNotFoundError if cargo is unavailable or hook/ is missing.
    """
    if dll_path and os.path.isfile(dll_path):
        return dll_path

    root = _project_root()

    candidates = [
        os.path.join(root, 'vision', 'dolphin_input_hook.dll'),
        os.path.join(root, 'dolphin_input_hook.dll'),
        os.path.join(root, 'hook', 'target', 'release', 'dolphin_input_hook.dll'),
    ]
    lib_rs = os.path.join(root, 'hook', 'src', 'lib.rs')

    for path in candidates:
        if os.path.isfile(path):
            # Stale check: rebuild if source is newer
            if os.path.isfile(lib_rs) and os.path.getmtime(lib_rs) > os.path.getmtime(path):
                logger.info(f"lib.rs is newer than DLL at {path} — rebuilding...")
                break
            logger.debug(f"DLL found at: {path}")
            return path

    # Auto-build with cargo
    hook_dir   = os.path.join(root, 'hook')
    cargo_toml = os.path.join(hook_dir, 'Cargo.toml')

    if not os.path.isfile(cargo_toml):
        raise FileNotFoundError(
            "dolphin_input_hook.dll not found and hook/Cargo.toml is missing.\n"
            f"Searched: {candidates}\n"
            "Either place a pre-built DLL at vision/dolphin_input_hook.dll\n"
            "or create the hook/ directory with Cargo.toml + src/lib.rs.")

    try:
        subprocess.run(['cargo', '--version'], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise FileNotFoundError(
            "dolphin_input_hook.dll not found and 'cargo' is not in PATH.\n"
            "Install Rust from https://rustup.rs and restart your terminal,\n"
            "or copy a pre-built DLL to vision/dolphin_input_hook.dll.")

    logger.info("dolphin_input_hook.dll not found — auto-building with cargo ...")
    result = subprocess.run(
        ['cargo', 'build', '--release'],
        cwd=hook_dir, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"cargo build --release failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr[-3000:]}")

    built = os.path.join(hook_dir, 'target', 'release', 'dolphin_input_hook.dll')
    if not os.path.isfile(built):
        raise FileNotFoundError(f"cargo succeeded but DLL not found at: {built}")

    # Copy to vision/ for next time
    vision_dir = os.path.join(root, 'vision')
    os.makedirs(vision_dir, exist_ok=True)
    dest = os.path.join(vision_dir, 'dolphin_input_hook.dll')
    shutil.copy2(built, dest)
    logger.info(f"DLL auto-built and saved to: {dest}")
    return dest


# ============================================================================
# PID resolution
# ============================================================================

def find_dolphin_pid(window_title: str) -> int:
    """
    Resolve a Dolphin window title to its PID.
    Priority: exact match → partial match → first Dolphin/MHTri window.
    """
    # 1. Exact match
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd and win32gui.IsWindowVisible(hwnd):
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        logger.debug(f"Exact window match '{window_title}' → PID {pid}")
        return pid

    # 2. Partial match
    results: list = []
    def _partial(h, _):
        t = win32gui.GetWindowText(h)
        if window_title.lower() in t.lower() and win32gui.IsWindowVisible(h):
            results.append((h, t))
    win32gui.EnumWindows(_partial, None)
    if results:
        hwnd, found = results[0]
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        logger.debug(f"Partial window match '{found}' → PID {pid}")
        return pid

    # 3. First Dolphin window
    all_wins: list = []
    def _any(h, _):
        t = win32gui.GetWindowText(h)
        if t and win32gui.IsWindowVisible(h):
            if any(k in t.lower() for k in ('dolphin', 'mhtri', 'monster hunter')):
                all_wins.append((h, t))
    win32gui.EnumWindows(_any, None)
    if all_wins:
        hwnd, found = all_wins[0]
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        logger.warning(
            f"Window '{window_title}' not found; auto-selecting '{found}' PID {pid}")
        return pid

    raise ValueError(
        f"No Dolphin window found for '{window_title}'. Is Dolphin running?")


# ============================================================================
# DLL injection
# ============================================================================

def inject_dll(pid: int, dll_path: str):
    """Inject dll_path into process pid via CreateRemoteThread(LoadLibraryA)."""
    dll_path = os.path.abspath(dll_path)
    logger.info(f"Injecting '{os.path.basename(dll_path)}' into PID {pid}")

    h_process = kernel32.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)
    if not h_process:
        raise OSError(f"OpenProcess failed (error {ctypes.get_last_error()})")
    try:
        dll_bytes  = (dll_path + '\0').encode('utf-8')
        remote_mem = kernel32.VirtualAllocEx(
            h_process, None, len(dll_bytes),
            win32con.MEM_COMMIT | win32con.MEM_RESERVE,
            win32con.PAGE_READWRITE)
        if not remote_mem:
            raise OSError("VirtualAllocEx failed")

        written = ctypes.c_size_t(0)
        if not kernel32.WriteProcessMemory(
                h_process, remote_mem,
                dll_bytes, len(dll_bytes), ctypes.byref(written)):
            raise OSError("WriteProcessMemory failed")

        k32      = kernel32.GetModuleHandleA(b'kernel32.dll')
        load_lib = kernel32.GetProcAddress(k32, b'LoadLibraryA')
        if not load_lib:
            raise OSError("GetProcAddress(LoadLibraryA) failed")

        h_thread = kernel32.CreateRemoteThread(
            h_process, None, 0, load_lib, remote_mem, 0, None)
        if not h_thread:
            raise OSError("CreateRemoteThread failed")

        kernel32.WaitForSingleObject(h_thread, 8000)
        kernel32.CloseHandle(h_thread)
        logger.info(f"DLL injected into PID {pid}")
    finally:
        kernel32.CloseHandle(h_process)


# ============================================================================
# Shared memory connection
# ============================================================================

def open_shared_memory(pid: int, timeout: float = 10.0):
    """
    Open the per-PID shared memory region created by the injected DLL.

    Args:
        pid:     Dolphin process ID.
        timeout: Max seconds to wait for the DLL to create the region.

    Returns:
        (handle, map_ptr, buf) where buf is a ctypes byte array mapped
        to the shared memory.

    Raises:
        TimeoutError if the shared memory is not available in time.
    """
    name     = shared_mem_name(pid).encode('ascii')
    deadline = time.time() + timeout

    handle = None
    while time.time() < deadline:
        h = kernel32.OpenFileMappingA(FILE_MAP_ALL_ACCESS, False, name)
        if h:
            handle = h
            break
        time.sleep(0.05)

    if not handle:
        raise TimeoutError(
            f"Shared memory '{shared_mem_name(pid)}' not available after {timeout}s — "
            "check dolphin_hook_debug.txt next to Dolphin.exe for DLL errors")

    map_ptr = kernel32.MapViewOfFile(
        handle, FILE_MAP_ALL_ACCESS, 0, 0, SHARED_MEM_SIZE)
    if not map_ptr:
        kernel32.CloseHandle(handle)
        raise OSError(
            f"MapViewOfFile failed (error {ctypes.get_last_error()})")

    buf = (ctypes.c_uint8 * SHARED_MEM_SIZE).from_address(map_ptr)
    return handle, map_ptr, buf


def close_shared_memory(handle, map_ptr):
    """Safely unmap and close a shared memory region."""
    if map_ptr:
        kernel32.UnmapViewOfFile(map_ptr)
    if handle:
        kernel32.CloseHandle(handle)


def is_dll_already_injected(pid: int) -> bool:
    """Check if the DLL's shared memory region already exists for this PID."""
    probe = kernel32.OpenFileMappingA(
        FILE_MAP_ALL_ACCESS, False,
        shared_mem_name(pid).encode('ascii'))
    if probe:
        kernel32.CloseHandle(probe)
        return True
    return False
