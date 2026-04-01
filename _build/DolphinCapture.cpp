/*
 * DolphinCapture.dll - Simple PrintWindow Capture with Minimize Detection
 *
 * Features:
 * - PrintWindow/BitBlt capture method
 * - Auto-restore minimized windows
 * - Multi-instance support via HWND targeting
 *
 * Build: Visual Studio 2019+ with Windows SDK 10.0.19041.0+
 * Target: x64 Release
 */

#include <Windows.h>
#include <vector>
#include <memory>
#include <mutex>
#include <dwmapi.h>

// Window show commands
#define SW_RESTORE 9
#define SW_SHOWNOACTIVATE 4
#define SW_MINIMIZE 6

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "gdi32.lib")

// PrintWindow flag for full content capture
#ifndef PW_RENDERFULLCONTENT
#define PW_RENDERFULLCONTENT 0x00000002
#endif

// ============================================================================
// CAPTURE ENGINE
// ============================================================================

class DolphinCaptureEngine {
private:
    HWND target_hwnd_;
    int width_;
    int height_;
    bool initialized_;
    std::mutex capture_mutex_;

public:
    DolphinCaptureEngine() :
        target_hwnd_(nullptr),
        width_(0),
        height_(0),
        initialized_(false) {}

    ~DolphinCaptureEngine() {
        Cleanup();
    }

    // Initialize DirectX capture for specific HWND
    bool Initialize(HWND hwnd) {
        std::lock_guard<std::mutex> lock(capture_mutex_);

        target_hwnd_ = hwnd;

        // Check if window is minimized and restore it PERMANENTLY
        if (IsIconic(hwnd)) {
            // Restore window WITHOUT activating (keeps it in background)
            ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            Sleep(150);  // Wait for window to fully restore

            char debug_msg[256];
            sprintf_s(debug_msg, "DolphinCapture: Window was minimized, restored PERMANENTLY to prevent flickering\n");
            OutputDebugStringA(debug_msg);
        }

        // Get window dimensions
        RECT rect;
        if (!GetClientRect(hwnd, &rect)) {
            // Fallback: Try GetWindowRect if GetClientRect fails
            if (!GetWindowRect(hwnd, &rect)) {
                width_ = 1280;
                height_ = 720;
                char debug_msg[256];
                sprintf_s(debug_msg, "DolphinCapture: GetClientRect failed, using default 1280x720\n");
                OutputDebugStringA(debug_msg);
            } else {
                width_ = rect.right - rect.left;
                height_ = rect.bottom - rect.top;
            }
        } else {
            width_ = rect.right - rect.left;
            height_ = rect.bottom - rect.top;
        }

        // Validate dimensions
        if (width_ <= 0 || height_ <= 0) {
            width_ = 1280;
            height_ = 720;
            char debug_msg[256];
            sprintf_s(debug_msg, "DolphinCapture: Invalid dimensions, using default 1280x720\n");
            OutputDebugStringA(debug_msg);
        }

        char debug_msg[256];
        sprintf_s(debug_msg, "DolphinCapture: Initialized for HWND %p - Dimensions: %dx%d (window kept restored)\n",
                  hwnd, width_, height_);
        OutputDebugStringA(debug_msg);

        initialized_ = true;
        return true;
    }

    // Capture current frame using DWM Thumbnail API (works even when minimized)
    int CaptureFrame(unsigned char* out_buffer, int buffer_size) {
        if (!initialized_ || !out_buffer) {
            return 0;
        }

        std::lock_guard<std::mutex> lock(capture_mutex_);

        // Expected buffer size (BGRA = 4 bytes per pixel)
        int expected_size = width_ * height_ * 4;
        if (buffer_size < expected_size) {
            return 0;
        }

        // Check if window got minimized again and restore it PERMANENTLY
        if (IsIconic(target_hwnd_)) {
            // Restore window WITHOUT activating (keeps in background)
            ShowWindow(target_hwnd_, SW_SHOWNOACTIVATE);
            Sleep(50);  // Wait for window to restore

            // Force window to redraw
            SendMessageTimeout(target_hwnd_, WM_PAINT, 0, 0, SMTO_NORMAL, 10, NULL);
            Sleep(10);  // Wait for paint to complete

            OutputDebugStringA("DolphinCapture: Window was minimized again, restored permanently\n");
        }

        // Get window DC
        HDC window_dc = GetDC(target_hwnd_);
        if (!window_dc) {
            return 0;
        }

        // Create compatible DC and bitmap
        HDC mem_dc = CreateCompatibleDC(window_dc);
        if (!mem_dc) {
            ReleaseDC(target_hwnd_, window_dc);
            return 0;
        }

        HBITMAP bitmap = CreateCompatibleBitmap(window_dc, width_, height_);
        if (!bitmap) {
            DeleteDC(mem_dc);
            ReleaseDC(target_hwnd_, window_dc);
            return 0;
        }

        HBITMAP old_bitmap = (HBITMAP)SelectObject(mem_dc, bitmap);

        // Use PrintWindow to capture content
        BOOL result = PrintWindow(target_hwnd_, mem_dc, PW_RENDERFULLCONTENT);

        // Fallback to BitBlt if PrintWindow fails
        if (!result) {
            result = BitBlt(mem_dc, 0, 0, width_, height_, window_dc, 0, 0, SRCCOPY);
        }

        if (!result) {
            SelectObject(mem_dc, old_bitmap);
            DeleteObject(bitmap);
            DeleteDC(mem_dc);
            ReleaseDC(target_hwnd_, window_dc);
            return 0;
        }

        // Get bitmap data
        BITMAPINFO bmi = {};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = width_;
        bmi.bmiHeader.biHeight = -height_;  // Negative for top-down bitmap
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        bmi.bmiHeader.biPlanes = 1;

        int bytes_copied = GetDIBits(mem_dc, bitmap, 0, height_, out_buffer, &bmi, DIB_RGB_COLORS);

        // Cleanup
        SelectObject(mem_dc, old_bitmap);
        DeleteObject(bitmap);
        DeleteDC(mem_dc);
        ReleaseDC(target_hwnd_, window_dc);

        if (bytes_copied == 0) {
            return 0;
        }

        return width_ * height_ * 4;
    }

    void Cleanup() {
        std::lock_guard<std::mutex> lock(capture_mutex_);
        initialized_ = false;
    }

    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    bool IsInitialized() const { return initialized_; }
};

// ============================================================================
// GLOBAL INSTANCE MANAGER
// ============================================================================

class CaptureInstanceManager {
private:
    std::vector<std::unique_ptr<DolphinCaptureEngine>> instances_;
    std::mutex manager_mutex_;

public:
    CaptureInstanceManager() {}

public:
    // Create new capture instance for HWND
    // Returns: instance ID (0-based), or -1 on failure
    int CreateInstance(HWND hwnd) {
        std::lock_guard<std::mutex> lock(manager_mutex_);

        auto engine = std::make_unique<DolphinCaptureEngine>();
        if (!engine->Initialize(hwnd)) {
            return -1;
        }

        instances_.push_back(std::move(engine));
        return static_cast<int>(instances_.size() - 1);
    }

    // Capture frame from specific instance
    int CaptureFrame(int instance_id, unsigned char* buffer, int buffer_size) {
        std::lock_guard<std::mutex> lock(manager_mutex_);

        if (instance_id < 0 || instance_id >= instances_.size()) {
            return 0;
        }

        return instances_[instance_id]->CaptureFrame(buffer, buffer_size);
    }

    // Get dimensions of specific instance
    void GetDimensions(int instance_id, int* width, int* height) {
        std::lock_guard<std::mutex> lock(manager_mutex_);

        if (instance_id < 0 || instance_id >= instances_.size()) {
            *width = 0;
            *height = 0;
            return;
        }

        *width = instances_[instance_id]->GetWidth();
        *height = instances_[instance_id]->GetHeight();
    }

    // Destroy specific instance
    void DestroyInstance(int instance_id) {
        std::lock_guard<std::mutex> lock(manager_mutex_);

        if (instance_id >= 0 && instance_id < instances_.size()) {
            instances_[instance_id]->Cleanup();
            instances_.erase(instances_.begin() + instance_id);
        }
    }

    void DestroyAll() {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        instances_.clear();
    }
};

// Global manager singleton
static CaptureInstanceManager g_manager;

// ============================================================================
// DLL EXPORTS (C API for Python ctypes)
// ============================================================================

extern "C" {

__declspec(dllexport) int __stdcall DolphinCapture_CreateInstance(HWND hwnd) {
    return g_manager.CreateInstance(hwnd);
}

__declspec(dllexport) int __stdcall DolphinCapture_CaptureFrame(
    int instance_id,
    unsigned char* buffer,
    int buffer_size
) {
    return g_manager.CaptureFrame(instance_id, buffer, buffer_size);
}

__declspec(dllexport) void __stdcall DolphinCapture_GetDimensions(
    int instance_id,
    int* width,
    int* height
) {
    g_manager.GetDimensions(instance_id, width, height);
}

__declspec(dllexport) void __stdcall DolphinCapture_DestroyInstance(int instance_id) {
    g_manager.DestroyInstance(instance_id);
}

__declspec(dllexport) void __stdcall DolphinCapture_DestroyAll() {
    g_manager.DestroyAll();
}

} // extern "C"

// ============================================================================
// DLL ENTRY POINT
// ============================================================================

BOOL APIENTRY DllMain(
    HMODULE hModule,
    DWORD ul_reason_for_call,
    LPVOID lpReserved
) {
    switch (ul_reason_for_call) {
        case DLL_PROCESS_ATTACH:
            DisableThreadLibraryCalls(hModule);
            OutputDebugStringA("DolphinCapture: DLL loaded - PrintWindow capture mode\n");
            break;
        case DLL_PROCESS_DETACH:
            g_manager.DestroyAll();
            OutputDebugStringA("DolphinCapture: DLL unloaded\n");
            break;
    }
    return TRUE;
}