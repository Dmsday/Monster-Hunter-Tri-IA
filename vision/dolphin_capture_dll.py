"""
Python wrapper for DolphinCapture.dll
Provides robust window capture with minimize detection
"""

import ctypes
import numpy as np
import sys
from pathlib import Path
from typing import Optional

from utils.module_logger import get_module_logger

logger = get_module_logger('dolphin_capture_dll')


class DolphinCaptureDLL:
    """
    Wrapper for DolphinCapture.dll - PrintWindow capture with auto-restore
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

        logger.info(f"Loading DolphinCapture.dll: {dll_full_path}")

        try:
            self.dll = ctypes.WinDLL(str(dll_full_path))
        except OSError as init_win_error:
            if "WinError 193" in str(init_win_error):
                python_bits = 64 if sys.maxsize > 2 ** 32 else 32
                raise RuntimeError(
                    f"DLL architecture mismatch: Python is {python_bits}-bit. "
                    f"Rebuild DLL for {python_bits}-bit architecture."
                ) from init_win_error
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

        logger.info("DLL loaded successfully")

    def create_instance(self, hwnd: int) -> int:
        """
        Create capture instance for specific window

        Args:
            hwnd: Window handle (HWND as integer)

        Returns:
            Instance ID (>= 0) or -1 on failure
        """
        instance_id = self.dll.DolphinCapture_CreateInstance(hwnd)

        if instance_id >= 0:
            logger.info(f"Created DLL instance {instance_id} for HWND {hwnd}")
        else:
            logger.error(f"Failed to create DLL instance for HWND {hwnd}")

        return instance_id

    def capture_frame(self, instance_id: int) -> Optional[np.ndarray]:
        """
        Capture frame from instance

        Args:
            instance_id: Instance ID from create_instance()

        Returns:
            numpy array (H, W, 4) BGRA format, or None on failure
        """
        # Get dimensions
        width = ctypes.c_int()
        height = ctypes.c_int()
        self.dll.DolphinCapture_GetDimensions(
            instance_id,
            ctypes.byref(width),
            ctypes.byref(height)
        )

        w = width.value
        h = height.value

        if w <= 0 or h <= 0:
            logger.warning(f"Invalid dimensions from DLL: {w}x{h}")
            return None

        # Allocate buffer
        buffer_size = w * h * 4  # BGRA
        buffer = (ctypes.c_ubyte * buffer_size)()

        # Capture
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
        logger.info(f"Destroyed DLL instance {instance_id}")

    def destroy_all(self):
        """Destroy all instances"""
        self.dll.DolphinCapture_DestroyAll()
        logger.info("Destroyed all DLL instances")