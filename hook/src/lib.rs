// dolphin_input_hook.dll — DInput keyboard hook only
// =====================================================
// Hooks IDirectInputDevice8::GetDeviceState (keyboard)
// and GetForegroundWindow / GetAsyncKeyState / GetKeyState
// so Dolphin accepts injected inputs without needing focus.
//
// WGI (Windows.Gaming.Input) gamepad code has been removed entirely.
// Only the DInput keyboard channel remains.
//
// Shared memory layout (280 bytes, name = "DolphinInputHook_SharedMem_<PID>"):
//   [0..255]    keyboard DIK scan codes  (1 = key pressed, 0 = released)
//   [256..263]  mouse buttons            (unused by Python side, kept in struct)
//   [264..279]  reserved / unused
//
// Log file: dolphin_hook_debug.txt next to Dolphin.exe

#![allow(non_snake_case, dead_code, non_camel_case_types)]

use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use windows::Win32::Foundation::*;
use windows::Win32::System::LibraryLoader::*;
use windows::Win32::System::Memory::*;
use windows::Win32::System::Threading::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::Win32::UI::Input::KeyboardAndMouse::MapVirtualKeyA;
use windows::Win32::System::ProcessStatus::*;

// ------------------------------------------------------------------
// Logging — writes next to Dolphin.exe (relative via current_exe)
// ------------------------------------------------------------------
fn log_path() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("dolphin_hook_debug.txt")))
        .unwrap_or_else(|| std::path::PathBuf::from("dolphin_hook_debug.txt"))
}

fn log(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open(log_path())
    {
        let _ = writeln!(f, "{}", msg);
    }
}

// ------------------------------------------------------------------
// Shared memory
// ------------------------------------------------------------------
// add HOTKEY_OFFSET at reserved bytes [264..279]
const SHARED_MEM_SIZE: usize = 280;
const MOUSE_OFFSET:    usize = 256;
// Byte 264: VK code to send as WM_KEYDOWN (0 = nothing pending)
// Written by Python controller, cleared by DLL after sending.
// Used for Dolphin system hotkeys (save states F1-F8) which go through
// the Win32 message queue, NOT DInput GetDeviceState.
const HOTKEY_OFFSET:   usize = 264;

static HOOK_ACTIVE:         AtomicBool = AtomicBool::new(false);
static CALL_COUNT:          AtomicU32  = AtomicU32::new(0);
static SEEN_16:             AtomicBool = AtomicBool::new(false);
static SEEN_20:             AtomicBool = AtomicBool::new(false);
static SEEN_256:            AtomicBool = AtomicBool::new(false);
static SEEN_OTHER:          AtomicBool = AtomicBool::new(false);

static mut SHARED_MEM_PTR:       *mut u8 = ptr::null_mut();
static mut FILE_MAPPING_HANDLE:  HANDLE  = HANDLE(0);
static mut DOLPHIN_HWND:         HWND    = HWND(0);

// ------------------------------------------------------------------
// Hook function types
// ------------------------------------------------------------------
type GetDeviceStateFn      = unsafe extern "system" fn(*mut (), u32, *mut u8) -> i32;
type GetAsyncKeyStateFn    = unsafe extern "system" fn(i32) -> i16;
type GetKeyStateFn         = unsafe extern "system" fn(i32) -> i16;
type GetForegroundWindowFn = unsafe extern "system" fn() -> HWND;

static mut ORIG_GET_DEVICE_STATE:      Option<GetDeviceStateFn>      = None;
static mut ORIG_GET_ASYNC_KEY_STATE:   Option<GetAsyncKeyStateFn>    = None;
static mut ORIG_GET_KEY_STATE:         Option<GetKeyStateFn>         = None;
static mut ORIG_GET_FOREGROUND_WINDOW: Option<GetForegroundWindowFn> = None;

// ------------------------------------------------------------------
// IDirectInputDevice8::GetDeviceState hook (vtable slot 9)
// Intercepts keyboard (cb_data=256) and mouse (cb_data=16/20) polls.
// ------------------------------------------------------------------
unsafe extern "system" fn hooked_GetDeviceState(
    this:    *mut (),
    cb_data: u32,
    lpv:     *mut u8,
) -> i32 {
    CALL_COUNT.fetch_add(1, Ordering::Relaxed);

    match cb_data {
        16  => { if !SEEN_16.swap(true,  Ordering::Relaxed) { log("[HOOK] cb_data=16 (mouse DIMOUSESTATE)"); } }
        20  => { if !SEEN_20.swap(true,  Ordering::Relaxed) { log("[HOOK] cb_data=20 (mouse DIMOUSESTATE2)"); } }
        256 => { if !SEEN_256.swap(true, Ordering::Relaxed) { log("[HOOK] cb_data=256 (keyboard DIK)"); } }
        n   => { if !SEEN_OTHER.swap(true, Ordering::Relaxed) { log(&format!("[HOOK] cb_data={} (other)", n)); } }
    }

    if lpv.is_null() { return -1; }

    // If the hook is not yet active, pass real input through normally.
    // Once active: BLOCK all real physical input — only shared memory keys reach Dolphin.
    // This is the critical fix for user keyboard/mouse bleeding into training instances.
    // The old pynput/ViGEmBus system injected at OS level and never touched DInput;
    // this DLL-based system must explicitly block the physical device state.
    if !HOOK_ACTIVE.load(Ordering::Relaxed) || SHARED_MEM_PTR.is_null() {
        return match ORIG_GET_DEVICE_STATE {
            Some(orig) => orig(this, cb_data, lpv),
            None       => -1,
        };
    }

    match cb_data {
        // Keyboard (cb_data=256): zero the entire buffer, then set only AI-injected keys.
        // Real physical keyboard is completely blocked from this Dolphin instance.
        256 => {
            static KBD_INJECTED: AtomicBool = AtomicBool::new(false);
            let mut any = false;
            for i in 0..256usize {
                if *SHARED_MEM_PTR.add(i) != 0 {
                    *lpv.add(i) = 0x80;
                    any = true;
                } else {
                    *lpv.add(i) = 0; // Explicitly block real key state
                }
            }
            if any && !KBD_INJECTED.swap(true, Ordering::Relaxed) {
                for i in 0..256usize {
                    if *SHARED_MEM_PTR.add(i) != 0 {
                        log(&format!("[HOOK-KBD] first injection DIK=0x{:02X}", i));
                        break;
                    }
                }
            }
            0 // S_OK — always succeed, never pass real keyboard through
        }

        // Mouse (cb_data=16/20): call orig for axes/movement data (bytes 0-11),
        // then override buttons (bytes 12+) from shared memory only.
        // Axes are needed for camera movement; buttons are AI-controlled.
        16 | 20 => {
            static MOUSE_INJECTED: AtomicBool = AtomicBool::new(false);
            let hr = match ORIG_GET_DEVICE_STATE {
                Some(orig) => orig(this, cb_data, lpv),
                None       => return -1,
            };
            let btn_offset = 12usize;
            let n_buttons  = (cb_data as usize - btn_offset).min(8);
            let mut any = false;
            for i in 0..n_buttons {
                // Block real mouse buttons; only injected state passes through
                *lpv.add(btn_offset + i) = if *SHARED_MEM_PTR.add(MOUSE_OFFSET + i) != 0 {
                    any = true;
                    0x80
                } else {
                    0
                };
            }
            if any && !MOUSE_INJECTED.swap(true, Ordering::Relaxed) {
                log("[HOOK-MOUSE] first injection");
            }
            hr
        }

        // Other buffer sizes: block real input, inject from shared memory if in range
        n if n > 20 && n <= 512 => {
            let mut any = false;
            for i in 0..(n as usize).min(256) {
                if *SHARED_MEM_PTR.add(i) != 0 {
                    *lpv.add(i) = 0x80;
                    any = true;
                } else {
                    *lpv.add(i) = 0; // Block real input
                }
            }
            if any { 0 } else { 0 } // Always S_OK when hook active
        }

        // Unknown types: pass through unchanged
        _ => {
            match ORIG_GET_DEVICE_STATE {
                Some(orig) => orig(this, cb_data, lpv),
                None       => -1,
            }
        }
    }
}

// ------------------------------------------------------------------
// Mouse VK → shared memory button index
// ------------------------------------------------------------------
unsafe fn mouse_vk_to_btn(vk: i32) -> Option<usize> {
    match vk {
        0x01 => Some(0), 0x02 => Some(1), 0x04 => Some(2),
        0x05 => Some(3), 0x06 => Some(4), _ => None,
    }
}

// F1-F8 return injected hotkey state; all other VKs pass through
unsafe extern "system" fn hooked_GetAsyncKeyState(vk: i32) -> i16 {
    if HOOK_ACTIVE.load(Ordering::Relaxed) && !SHARED_MEM_PTR.is_null() {
        // Mouse buttons: return injected state only
        if let Some(btn) = mouse_vk_to_btn(vk) {
            return if *SHARED_MEM_PTR.add(MOUSE_OFFSET + btn) != 0 { -32768i16 } else { 0 };
        }

        // F1-F8: pass through to real GetAsyncKeyState.
        // Dolphin's Qt HotkeyManager polls GetAsyncKeyState for save state hotkeys.
        // We do NOT intercept F-keys here — pynput handles save states directly
        // via the real Win32 API. Blocking them (old behavior) prevented F5 from
        // ever reaching Dolphin.
        if vk >= 0x70 && vk <= 0x77 {
            return match ORIG_GET_ASYNC_KEY_STATE {
                Some(orig) => orig(vk),
                None => 0,
            };
        }
    }
    match ORIG_GET_ASYNC_KEY_STATE { Some(orig) => orig(vk), None => 0 }
}

unsafe extern "system" fn hooked_GetKeyState(vk: i32) -> i16 {
    if HOOK_ACTIVE.load(Ordering::Relaxed) && !SHARED_MEM_PTR.is_null() {
        if let Some(btn) = mouse_vk_to_btn(vk) {
            // Same as GetAsyncKeyState: injected state only, real state blocked
            return if *SHARED_MEM_PTR.add(MOUSE_OFFSET + btn) != 0 { -32768i16 } else { 0 };
        }
        return 0;
    }
    match ORIG_GET_KEY_STATE { Some(orig) => orig(vk), None => 0 }
}

// GetForegroundWindow hook: return Dolphin's HWND whenever any key is injected
// so Dolphin believes it has focus and processes the input.
unsafe extern "system" fn hooked_GetForegroundWindow() -> HWND {
    // Do NOT spoof foreground window anymore.
    // Spoofing caused WM_KEYDOWN bleed: Dolphin thought it had focus and processed
    // real keyboard messages from the Windows message queue in addition to DInput.
    // DInput interception alone (GetDeviceState hook) is sufficient for injecting AI input.
    match ORIG_GET_FOREGROUND_WINDOW { Some(orig) => orig(), None => HWND(0) }
}

// ------------------------------------------------------------------
// GUIDs for DirectInput8
// ------------------------------------------------------------------
const GUID_SYS_KEYBOARD: windows::core::GUID = windows::core::GUID::from_values(
    0x6F1D2B61, 0xD5A0, 0x11CF,
    [0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00],
);
const IID_DIRECT_INPUT8W: windows::core::GUID = windows::core::GUID::from_values(
    0xBF798031, 0x483A, 0x4DA2,
    [0xAA, 0x99, 0x5D, 0x64, 0xED, 0x36, 0x97, 0x00],
);

type DI8CreateFn    = unsafe extern "system" fn(HINSTANCE, u32, *const windows::core::GUID, *mut *mut *const usize, *mut ()) -> i32;
type CreateDeviceFn = unsafe extern "system" fn(*mut *const usize, *const windows::core::GUID, *mut *mut *const usize, *mut ()) -> i32;
type ReleaseFn      = unsafe extern "system" fn(*mut *const usize) -> u32;

// ------------------------------------------------------------------
// Vtable helpers
// ------------------------------------------------------------------
unsafe fn patch_vtable_slot(vtable: *const usize, slot: usize, new_fn: usize) -> usize {
    let slot_ptr = vtable.add(slot) as *mut usize;
    let old      = *slot_ptr;
    let mut old_prot = PAGE_PROTECTION_FLAGS(0);
    let _ = VirtualProtect(slot_ptr as *const _, std::mem::size_of::<usize>(), PAGE_READWRITE, &mut old_prot);
    *slot_ptr = new_fn;
    let _ = VirtualProtect(slot_ptr as *const _, std::mem::size_of::<usize>(), old_prot, &mut old_prot);
    old
}

unsafe fn patch_iat_fn(base: *mut u8, target_addr: usize, new_addr: usize) -> Option<usize> {
    let dos = base as *const [u8; 2];
    if (*dos) != *b"MZ" { return None; }
    let e_lfanew = *(base.add(0x3C) as *const i32);
    if e_lfanew <= 0 || e_lfanew > 0x10000 { return None; }
    let nt = base.add(e_lfanew as usize);
    if *(nt as *const u32) != 0x4550 { return None; }
    let import_rva  = *(nt.add(0x90) as *const u32);
    let import_size = *(nt.add(0x94) as *const u32);
    if import_rva == 0 || import_size == 0 { return None; }

    let mut desc = base.add(import_rva as usize) as *const [u32; 5];
    loop {
        let [_orig_thunk, _, _, _, first_thunk] = *desc;
        if first_thunk == 0 { break; }
        let mut iat = base.add(first_thunk as usize) as *mut usize;
        loop {
            let val = *iat;
            if val == 0 { break; }
            if val == target_addr {
                let mut old = PAGE_PROTECTION_FLAGS(0);
                let _ = VirtualProtect(iat as *const _, 8, PAGE_READWRITE, &mut old);
                *iat = new_addr;
                let _ = VirtualProtect(iat as *const _, 8, old, &mut old);
                return Some(val);
            }
            iat = iat.add(1);
        }
        desc = desc.add(1);
    }
    None
}

// ------------------------------------------------------------------
// DInput hook: patches IDirectInputDevice8 vtable slot 9
// ------------------------------------------------------------------
unsafe fn hook_dinput() -> bool {
    let dinput8 = match GetModuleHandleA(windows::core::PCSTR(b"dinput8.dll\0".as_ptr())) {
        Ok(h)  => { log("[DIAG] dinput8.dll found"); h }
        Err(_) => { log("[DIAG] dinput8.dll not loaded — DInput hook skipped"); return true; }
    };

    let create_proc = match GetProcAddress(dinput8,
        windows::core::PCSTR(b"DirectInput8Create\0".as_ptr()))
    { Some(f) => f, None => { log("[DIAG] DirectInput8Create not found"); return false; } };

    let di8_create: DI8CreateFn = std::mem::transmute(create_proc);
    let exe = GetModuleHandleA(windows::core::PCSTR(ptr::null())).unwrap_or(HMODULE(0));

    let mut di_ptr: *mut *const usize = ptr::null_mut();
    let hr = di8_create(HINSTANCE(exe.0), 0x0800, &IID_DIRECT_INPUT8W, &mut di_ptr, ptr::null_mut());
    log(&format!("[DIAG] DirectInput8Create hr=0x{:08X}", hr as u32));
    if hr != 0 || di_ptr.is_null() { return false; }

    let vtable_di = *di_ptr;
    let create_device: CreateDeviceFn = std::mem::transmute(*vtable_di.add(3));
    let mut dev_ptr: *mut *const usize = ptr::null_mut();
    let hr2 = create_device(di_ptr, &GUID_SYS_KEYBOARD, &mut dev_ptr, ptr::null_mut());
    log(&format!("[DIAG] CreateDevice(Keyboard) hr=0x{:08X}", hr2 as u32));

    let release_di: ReleaseFn = std::mem::transmute(*vtable_di.add(2));
    release_di(di_ptr);

    if hr2 != 0 || dev_ptr.is_null() { log("[DIAG] CreateDevice failed"); return false; }

    let vtable_dev = *dev_ptr;
    let old = patch_vtable_slot(vtable_dev, 9, hooked_GetDeviceState as *const () as usize);
    ORIG_GET_DEVICE_STATE = Some(std::mem::transmute(old));
    log("[DIAG] DInput vtable[9] patched (keyboard + mouse GetDeviceState)");

    let release_dev: ReleaseFn = std::mem::transmute(*vtable_dev.add(2));
    release_dev(dev_ptr);

    patch_get_foreground_window();
    true
}

// ------------------------------------------------------------------
// Patch GetForegroundWindow, GetAsyncKeyState, GetKeyState (IAT)
// These make Dolphin believe it has focus so it processes our keys.
// ------------------------------------------------------------------
// re-enable GetAsyncKeyState patching for F-key hotkey injection.
// Dolphin Qt HotkeyManager polls GetAsyncKeyState for save state hotkeys,
// NOT WM_KEYDOWN. We only intercept F1-F8; all other VKs pass through.
unsafe fn patch_get_foreground_window() {
    log("[DIAG] Patching GetAsyncKeyState for F-key hotkey injection");
    patch_async_key_hooks();
}

unsafe fn patch_async_key_hooks() {
    let user32 = match GetModuleHandleA(windows::core::PCSTR(b"user32.dll\0".as_ptr())) {
        Ok(h) => h, Err(_) => return,
    };
    let gaks_addr = match GetProcAddress(user32,
        windows::core::PCSTR(b"GetAsyncKeyState\0".as_ptr()))
    { Some(f) => f as usize, None => return };
    let gks_addr  = match GetProcAddress(user32,
        windows::core::PCSTR(b"GetKeyState\0".as_ptr()))
    { Some(f) => f as usize, None => return };

    let h_process = GetCurrentProcess();
    let mut modules = vec![HMODULE::default(); 1024];
    let mut needed: u32 = 0;
    if EnumProcessModules(h_process, modules.as_mut_ptr(),
        (modules.len() * std::mem::size_of::<HMODULE>()) as u32, &mut needed).is_err() { return; }
    let count = needed as usize / std::mem::size_of::<HMODULE>();

    let (mut p_gaks, mut p_gks) = (0usize, 0usize);
    // break early once both functions are patched in the main exe
    for i in 0..count {
        let base = modules[i].0 as *mut u8;
        if base.is_null() { continue; }
        if p_gaks == 0 {
            if let Some(o) = patch_iat_fn(base, gaks_addr,
                hooked_GetAsyncKeyState as *const () as usize) {
                if ORIG_GET_ASYNC_KEY_STATE.is_none() {
                    ORIG_GET_ASYNC_KEY_STATE = Some(std::mem::transmute(o));
                }
                p_gaks += 1;
            }
        }
        if p_gks == 0 {
            if let Some(o) = patch_iat_fn(base, gks_addr,
                hooked_GetKeyState as *const () as usize) {
                if ORIG_GET_KEY_STATE.is_none() {
                    ORIG_GET_KEY_STATE = Some(std::mem::transmute(o));
                }
                p_gks += 1;
            }
        }
        // Both found — no need to scan remaining modules
        if p_gaks > 0 && p_gks > 0 { break; }
    }
    log(&format!("[DIAG] GetAsyncKeyState patched in {} module(s)", p_gaks));
    log(&format!("[DIAG] GetKeyState      patched in {} module(s)", p_gks));
}

// ------------------------------------------------------------------
// Find the main visible Dolphin window in this process
// ------------------------------------------------------------------
// search for the main Dolphin window (NOT the render window renamed "MHTri-N")
// the main window ("Dolphin XXXX") handles app-level hotkeys (save states)
// the render window ("MHTri-N") handles game display
unsafe fn find_dolphin_hwnd() -> HWND {
    struct FindData { pid: u32, hwnd: HWND }
    static mut FIND: FindData = FindData { pid: 0, hwnd: HWND(0) };
    FIND.pid  = GetCurrentProcessId();
    FIND.hwnd = HWND(0);

    unsafe extern "system" fn enum_cb(hwnd: HWND, _: LPARAM) -> BOOL {
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if pid == FIND.pid {
            let mut title = [0u16; 256];
            let len = GetWindowTextW(hwnd, &mut title);
            if len > 0 {
                let s = String::from_utf16_lossy(&title[..len as usize]);
                // Skip the render window renamed to "MHTri-N" by PowerShell.
                // Only the main Dolphin window ("Dolphin XXXX") processes
                // app-level hotkeys like save states via its wxWidgets event loop.
                if !s.starts_with("MHTri") {
                    FIND.hwnd = hwnd;
                    return BOOL(0); // found main window, stop enumeration
                }
            }
        }
        BOOL(1)
    }
    let _ = EnumWindows(Some(enum_cb), LPARAM(0));
    FIND.hwnd
}

// ------------------------------------------------------------------
// DLL entry point
// ------------------------------------------------------------------
#[no_mangle]
pub unsafe extern "system" fn DllMain(
    _hmodule:  HINSTANCE,
    reason:    u32,
    _reserved: *const (),
) -> bool {
    match reason {
        1 => {
            // DLL_PROCESS_ATTACH
            let _ = std::fs::write(
                log_path(),
                "=== dolphin_hook_debug.txt (DInput keyboard mode) ===\n");
            log("[MAIN] DllMain ATTACH");
            let _ = CreateThread(
                None, 0, Some(init_thread), None, THREAD_CREATION_FLAGS(0), None);
        }
        0 => {
            // DLL_PROCESS_DETACH
            cleanup();
        }
        _ => {}
    }
    true
}

unsafe extern "system" fn init_thread(_: *mut std::ffi::c_void) -> u32 {
    log("[INIT] waiting 500ms for Dolphin startup...");
    std::thread::sleep(std::time::Duration::from_millis(500));

    let pid      = GetCurrentProcessId();
    let shm_name = format!("DolphinInputHook_SharedMem_{}\0", pid);
    log(&format!("[INIT] creating shared memory: {}", shm_name.trim_end_matches('\0')));

    match CreateFileMappingA(
        INVALID_HANDLE_VALUE, None, PAGE_READWRITE,
        0, SHARED_MEM_SIZE as u32,
        windows::core::PCSTR(shm_name.as_ptr()),
    ) {
        Ok(h) => {
            let view = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, SHARED_MEM_SIZE);
            if view.Value.is_null() {
                log("[INIT] MapViewOfFile returned null");
                return 1;
            }
            SHARED_MEM_PTR      = view.Value as *mut u8;
            FILE_MAPPING_HANDLE = h;
            log("[INIT] shared memory created");
        }
        Err(e) => {
            log(&format!("[INIT] CreateFileMapping failed: {:?}", e));
            return 1;
        }
    }

    log("[INIT] hooking DInput (keyboard + mouse)...");
    hook_dinput();

    // APRÈS — hotkey state managed via GetAsyncKeyState hook only, no thread needed
    HOOK_ACTIVE.store(true, Ordering::SeqCst);
    log("[INIT] *** HOOK ACTIVE (DInput keyboard) ***");
    // No HotkeyInjector thread — Python writes VK to byte 264, holds for 300ms,
    // then clears. hooked_GetAsyncKeyState returns pressed state while byte is set.

    // Watchdog: log call count every 3 s for 15 s
    for i in 1..=5u32 {
        std::thread::sleep(std::time::Duration::from_millis(3000));
        log(&format!(
            "[WATCHDOG t={}s] dinput_calls={}",
            i * 3,
            CALL_COUNT.load(Ordering::Relaxed)));
    }
    0
}

unsafe fn cleanup() {
    HOOK_ACTIVE.store(false, Ordering::SeqCst);
    if !SHARED_MEM_PTR.is_null() {
        let _ = UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: SHARED_MEM_PTR as *mut _ });
        SHARED_MEM_PTR = ptr::null_mut();
    }
    if FILE_MAPPING_HANDLE.0 != 0 {
        let _ = CloseHandle(FILE_MAPPING_HANDLE);
        FILE_MAPPING_HANDLE = HANDLE(0);
    }
}