"""
Test script for DolphinCapture.dll
Tests PrintWindow capture with automatic minimize detection

Prerequisites:
1. Build DolphinCapture.dll manually first: python build_dll.py
2. Launch Dolphin with Monster Hunter Tri loaded

Capture Method:
- PrintWindow/BitBlt with automatic minimize detection
- Automatically restores minimized windows PERMANENTLY
- Windows stay restored to prevent flickering

Note: Minimized windows are restored once and kept restored
      to prevent flickering
"""

import ctypes
import numpy as np
import cv2
import time
import win32gui
from pathlib import Path
import sys

# ============================================================================
# PYTEST FIXTURES (for pytest compatibility)
# ============================================================================

import pytest

# Global DLL instance (shared across all tests)
_dll_instance = None


@pytest.fixture(scope="session", autouse=True)
def setup_dll():
    """
    PyTest fixture: Setup DLL before all tests, cleanup after
    This runs automatically when using pytest
    """
    global _dll_instance

    print("\n" + "=" * 70)
    print("PYTEST SESSION SETUP")
    print("=" * 70)

    # Load DLL
    print("\n📦 Loading DolphinCapture.dll...")

    try:
        _dll_instance = DolphinCaptureDLL()
        print("✅ DLL loaded and ready for all tests")
        print("=" * 70)
        print()
    except FileNotFoundError:
        pytest.fail("DLL not found - run 'python build_dll.py' first")
    except Exception as e:
        pytest.fail(f"DLL load error: {e}")

    # Run all tests (yield gives control to pytest)
    yield _dll_instance

    # Cleanup after all tests
    print("\n" + "=" * 70)
    print("PYTEST SESSION CLEANUP")
    print("=" * 70)
    if _dll_instance:
        _dll_instance.destroy_all()
        print("✅ All DLL instances destroyed")


@pytest.fixture
def dll():
    """
    PyTest fixture: Provide DLL instance to each test
    Usage: def test_something(dll): ...
    """
    global _dll_instance
    if _dll_instance is None:
        pytest.fail("DLL not initialized - check setup_dll fixture")
    return _dll_instance

# ============================================================================
# DLL WRAPPER CLASS
# ============================================================================

# noinspection PyTypeChecker
class DolphinCaptureDLL:
    """
    Python wrapper for DolphinCapture.dll
    Provides easy access to C++ capture engine
    """

    def __init__(self, dll_path: str = "DolphinCapture.dll"):
        """
        Load DLL and setup function signatures

        Args:
            dll_path: Path to DolphinCapture.dll
        """
        # Load DLL
        dll_full_path = Path(dll_path).absolute()
        if not dll_full_path.exists():
            raise FileNotFoundError(f"DLL not found: {dll_full_path}")

        print(f"Loading DLL: {dll_full_path}")

        try:
            self.dll = ctypes.WinDLL(str(dll_full_path))
        except OSError as e:
            if "WinError 193" in str(e) or "not a valid Win32 application" in str(e):
                python_bits = 64 if sys.maxsize > 2 ** 32 else 32
                print(f"\n❌ Architecture mismatch error!")
                print(f"   Python: {python_bits}-bit")
                print(f"   DLL: Likely {32 if python_bits == 64 else 64}-bit")
                print(f"\n💡 Solution:")
                if python_bits == 64:
                    print(f"   1. Open Visual Studio")
                    print(f"   2. Set Configuration to 'Release'")
                    print(f"   3. Set Platform to 'x64' (not x86)")
                    print(f"   4. Rebuild → Copy DLL to project folder")
                else:
                    print(f"   Install 64-bit Python from python.org")
                raise RuntimeError(f"DLL architecture mismatch: {e}") from e
            else:
                raise

        # Setup function signatures
        # int CreateInstance(HWND hwnd)
        self.dll.DolphinCapture_CreateInstance.argtypes = [ctypes.c_void_p]
        self.dll.DolphinCapture_CreateInstance.restype = ctypes.c_int

        # int CaptureFrame(int instance_id, unsigned char* buffer, int buffer_size)
        self.dll.DolphinCapture_CaptureFrame.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int
        ]
        self.dll.DolphinCapture_CaptureFrame.restype = ctypes.c_int

        # void GetDimensions(int instance_id, int* width, int* height)
        self.dll.DolphinCapture_GetDimensions.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.dll.DolphinCapture_GetDimensions.restype = None

        # void DestroyInstance(int instance_id)
        self.dll.DolphinCapture_DestroyInstance.argtypes = [ctypes.c_int]
        self.dll.DolphinCapture_DestroyInstance.restype = None

        # void DestroyAll()
        self.dll.DolphinCapture_DestroyAll.argtypes = []
        self.dll.DolphinCapture_DestroyAll.restype = None

        print("✅ DLL loaded successfully")

    def create_instance(self, hwnd: int) -> int:
        """
        Create capture instance for specific window

        Args:
            hwnd: Window handle (HWND as integer)

        Returns:
            Instance ID (0+), or -1 on failure
        """
        instance_id = self.dll.DolphinCapture_CreateInstance(hwnd)
        if instance_id >= 0:
            print(f"✅ Created instance {instance_id} for HWND {hwnd}")
        else:
            print(f"❌ Failed to create instance for HWND {hwnd}")
        return instance_id

    def capture_frame(self, instance_id: int) -> np.ndarray:
        """
        Capture frame from instance

        NOTE: DLL automatically uses DirectX hook fallback if PrintWindow returns black frame

        Args:
            instance_id: Instance ID from create_instance()

        Returns:
            numpy array (H, W, 4) BGRA format, or None on failure
        """
        # Get dimensions
        width = ctypes.c_int()
        height = ctypes.c_int()
        self.dll.DolphinCapture_GetDimensions(instance_id, ctypes.byref(width), ctypes.byref(height))

        w = width.value
        h = height.value

        if w <= 0 or h <= 0:
            return None

        # Allocate buffer
        buffer_size = w * h * 4  # BGRA = 4 bytes per pixel
        buffer = (ctypes.c_ubyte * buffer_size)()

        # Capture (DLL handles fallback internally)
        bytes_captured = self.dll.DolphinCapture_CaptureFrame(
            instance_id,
            buffer,
            buffer_size
        )

        if bytes_captured <= 0:
            return None

        # Convert to numpy array
        frame = np.ctypeslib.as_array(buffer)
        frame = frame.reshape((h, w, 4))  # BGRA format

        return frame

    def destroy_instance(self, instance_id: int):
        """Destroy specific instance"""
        self.dll.DolphinCapture_DestroyInstance(instance_id)
        print(f"🗑️ Destroyed instance {instance_id}")

    def destroy_all(self):
        """Destroy all instances"""
        self.dll.DolphinCapture_DestroyAll()
        print("🗑️ Destroyed all instances")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_dolphin_windows() -> list:
    """
    Find all Dolphin windows with Monster Hunter Tri

    Returns:
        List of (hwnd, title) tuples
    """
    windows = []

    def callback(hwnd, wins):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            title_lower = title.lower()

            # Look for MHTri or Monster Hunter
            if "mhtri" in title_lower or "monster hunter" in title_lower:
                wins.append((hwnd, title))
        return True

    win32gui.EnumWindows(callback, windows)
    return windows


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_single_capture(dll):
    """
    Test 1: Single window capture

    Args:
        dll: DolphinCaptureDLL fixture (injected by pytest)
    """
    print("\n" + "=" * 70)
    print("TEST 1: SINGLE WINDOW CAPTURE")
    print("=" * 70)

    # Search for Dolphin windows (DLL already loaded via fixture)
    print("\n🔍 Searching for Dolphin windows...")
    windows = find_dolphin_windows()

    # Find Dolphin window
    windows = find_dolphin_windows()
    if not windows:
        print("❌ No Dolphin window found")
        print("💡 Launch Dolphin with Monster Hunter Tri first")
        return False

    hwnd, title = windows[0]
    print(f"📺 Found window: '{title}' (HWND: {hwnd})")

    # Create instance
    instance_id = dll.create_instance(hwnd)
    if instance_id < 0:
        print("❌ Failed to create capture instance")
        return False

    # Capture test frames
    print("\n📸 Capturing 10 test frames...")
    success_count = 0

    for i in range(10):
        frame = dll.capture_frame(instance_id)

        if frame is not None:
            success_count += 1
            print(f"  ✅ Frame {i + 1}: {frame.shape}")

            # Save first 3 frames
            if i < 3:
                # Convert BGRA to BGR for saving
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                filename = f"test_capture_dll_frame{i}.png"
                cv2.imwrite(filename, bgr)
                print(f"  💾 Saved: {filename}")
        else:
            print(f"  ❌ Frame {i + 1}: Failed")

        time.sleep(0.1)

    # Results
    print(f"\n📊 Results: {success_count}/10 successful captures")
    return success_count >= 8  # 80% success rate

def test_multi_instance(dll):
    """
    Test 2: Multi-instance capture

    Args:
        dll: DolphinCaptureDLL fixture (injected by pytest)
    """
    print("\n" + "=" * 70)
    print("TEST 2: MULTI-INSTANCE CAPTURE")
    print("=" * 70)

    # Search for windows (DLL already loaded)
    print("\n🔍 Searching for Dolphin windows...")
    windows = find_dolphin_windows()

    # Find all Dolphin windows
    windows = find_dolphin_windows()
    if len(windows) < 2:
        print(f"⚠️ Found only {len(windows)} window(s)")
        print("💡 Launch 2+ Dolphin instances for this test")
        return False

    print(f"📺 Found {len(windows)} windows:")
    for i, (hwnd, title) in enumerate(windows):
        print(f"  [{i}] {title} (HWND: {hwnd})")

    # Create instances for all windows
    instances = []
    for hwnd, title in windows:
        instance_id = dll.create_instance(hwnd)
        if instance_id >= 0:
            instances.append((instance_id, title))

    if len(instances) < 2:
        print("❌ Failed to create multiple instances")
        dll.destroy_all()
        return False

    print(f"\n✅ Created {len(instances)} capture instances")

    # Capture from all instances simultaneously
    print("\n📸 Capturing from all instances...")

    for frame_idx in range(5):
        print(f"\nFrame {frame_idx + 1}/5:")

        for instance_id, title in instances:
            frame = dll.capture_frame(instance_id)

            if frame is not None:
                print(f"  ✅ Instance {instance_id} ({title}): {frame.shape}")

                # Save first frame from each instance
                if frame_idx == 0:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    filename = f"test_multi_instance_{instance_id}.png"
                    cv2.imwrite(filename, bgr)
                    print(f"     💾 Saved: {filename}")
            else:
                print(f"  ❌ Instance {instance_id} ({title}): Failed")

        time.sleep(0.1)

    # Cleanup
    dll.destroy_all()

    print("\n✅ Multi-instance test completed")
    return True


def test_covered_window(dll):
    """
    Test 3: Capture window even when covered

    Args:
        dll: DolphinCaptureDLL fixture (injected by pytest)
    """
    print("\n" + "=" * 70)
    print("TEST 3: COVERED WINDOW CAPTURE")
    print("=" * 70)
    print("💡 Manually cover the Dolphin window with another window")
    print("   during the next 10 seconds...")

    # Search for windows (DLL already loaded)
    print("\n🔍 Searching for Dolphin windows...")
    windows = find_dolphin_windows()

    # Find window
    windows = find_dolphin_windows()
    if not windows:
        print("❌ No Dolphin window found")
        return False

    hwnd, title = windows[0]
    print(f"📺 Target: '{title}'")

    # Create instance
    instance_id = dll.create_instance(hwnd)
    if instance_id < 0:
        print("❌ Failed to create instance")
        return False

    # Countdown
    print("\nStarting capture in:")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\n📸 Capturing for 10 seconds (cover the window NOW)...")

    start_time = time.time()
    frame_count = 0
    success_count = 0

    while time.time() - start_time < 10:
        frame = dll.capture_frame(instance_id)
        frame_count += 1

        if frame is not None:
            success_count += 1

            # Show progress every second
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  {elapsed:.1f}s: {success_count}/{frame_count} successful ({success_count * 100 // frame_count}%)")

        time.sleep(1 / 30)  # 30 FPS

    # Cleanup
    dll.destroy_instance(instance_id)

    success_rate = (success_count * 100) // frame_count if frame_count > 0 else 0
    print(f"\n📊 Final results: {success_count}/{frame_count} successful ({success_rate}%)")

    if success_rate >= 80:
        print("✅ Test PASSED - Robust capture working!")
        return True
    else:
        print("⚠️ Test INCONCLUSIVE - Success rate too low")
        return False


# ============================================================================
# MAIN TEST SUITE
# ============================================================================

def main():
    print("=" * 70)
    print("DOLPHINCAPTURE.DLL TEST SUITE")
    print("=" * 70)
    print("\nThis script tests the PrintWindow capture DLL")
    print("Make sure Dolphin is running with Monster Hunter Tri loaded\n")

    # Load DLL
    print("=" * 70)
    print("DLL INITIALIZATION")
    print("=" * 70)
    print("\n📦 Loading DolphinCapture.dll...")

    try:
        dll = DolphinCaptureDLL()
        print("✅ DLL loaded and ready for all tests")
        print("=" * 70)
        print()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Build the DLL first:")
        print("   python build_dll.py")
        return
    except Exception as e:
        print(f"❌ DLL load error: {e}")
        print("\n💡 Possible causes:")
        print("   1. DLL is corrupted - rebuild with: python build_dll.py")
        print("   2. Architecture mismatch (32-bit vs 64-bit)")
        print("   3. Missing dependencies (Visual C++ Runtime)")
        return

    # Run all tests with the same DLL instance
    results = {}

    # Test 1: Single capture
    results['single'] = test_single_capture()

    # Test 2: Multi-instance (optional)
    input("\nPress ENTER to continue to multi-instance test...")
    results['multi'] = test_multi_instance()

    # Test 3: Covered window
    input("\nPress ENTER to continue to covered window test...")
    results['covered'] = test_covered_window()

    # Cleanup DLL at the end
    print("\n" + "=" * 70)
    print("CLEANUP")
    print("=" * 70)
    dll.destroy_all()
    print("✅ All DLL instances destroyed")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper():20s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n💡 Next steps:")
        print("   1. Integrate DLL into frame_capture.py")
        print("   2. Update MonsterHunterEnv to use new capture")
        print("   3. Test multi-instance training")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("💡 Check:")
        print("   1. Dolphin is running")
        print("   2. Window is not hidden")
        print("   3. Try rebuilding DLL")

    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()