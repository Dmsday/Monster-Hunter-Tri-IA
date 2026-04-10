"""
Interactive tool to adjust HUD cropping
Uses OpenCV for real-time display and keyboard adjustment
"""

import cv2
import numpy as np
import json
import os

from vision.frame_capture import FrameCapture
from info.module_logger import get_module_logger

logger = get_module_logger('hud_crop_tuner')


class HudCropTunerLoadError(Exception):
    """Raised when the HUD crop tuner cannot load a source frame."""

class HUDCropTuner:
    """
    Interactive tool to calibrate HUD cropping

    Keyboard controls:
    - W/S : Adjust top crop
    - A/D : Adjust left crop
    - I/K : Adjust bottom crop
    - J/L : Adjust right crop
    - R : Reset to default values
    - SPACE : Capture a new frame
    - ENTER : Save and exit
    - ESC : Exit without saving
    """

    def __init__(self):
        self.capturer = None
        self.current_frame: np.ndarray | None = None

        # Crop values (proportions 0.0 - 1.0)
        self.top_crop = 0.12
        self.bottom_crop = 0.15
        self.left_crop = 0.05
        self.right_crop = 0.05

        # Adjustment increment
        self.step = 0.01  # 1%

        # Window
        self.window_name = "HUD Crop Tuner - Monster Hunter"

        # Default config
        self.default_config = {
            'top_crop': 0.12,
            'bottom_crop': 0.15,
            'left_crop': 0.05,
            'right_crop': 0.05
        }

    def capture_frame_from_dolphin(self):
        """Capture a frame from Dolphin via DolphinCapture.dll (window can be hidden)"""
        try:
            if self.capturer is None:
                print("\n📸 Connecting to Dolphin via DLL capture...")
                # Force DLL-based capture: Dolphin window no longer needs to be visible/foreground
                # use_dll=True enables DolphinCapture.dll which uses PrintWindow with auto-restore
                self.capturer = FrameCapture(
                    window_name="Dolphin",
                    use_dll=True,
                )

                # Verify DLL capture path is actually active (fallback to GDI is silent otherwise)
                if not self.capturer.use_dll:
                    logger.warning("DLL capture unavailable - falling back to GDI (window must be visible)")
                    print("⚠️  DolphinCapture.dll not loaded - window must remain visible")
                else:
                    print("✅ DLL capture active - Dolphin window can be hidden")

            frame = self.capturer.capture_frame()

            if frame is None or frame.size == 0:
                print("❌ Captured frame is empty !")
                return None

            print(f"✅ Frame captured successfully: {frame.shape}")
            return frame

        except ValueError as capture_window_error:
            print(f"❌ Error: {capture_window_error}")
            print("\n💡 Make sure that:")
            print("   - Dolphin is running")
            print("   - A game is running")
            return None
        except Exception as capture_unexpected_error:
            logger.error(f"Unexpected capture error: {capture_unexpected_error}")
            print(f"❌ Unexpected error: {capture_unexpected_error}")
            return None

    def draw_crop_overlay(self, frame):
        """
        Dessine l'overlay du crop sur la frame

        Args:
            frame: Image RGB

        Returns:
            Frame avec overlay dessiné
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Compute crop coordinates
        top = int(h * self.top_crop)
        bottom = int(h * (1 - self.bottom_crop))
        left = int(w * self.left_crop)
        right = int(w * (1 - self.right_crop))

        # Darken areas to remove
        overlay = display_frame.copy()

        # Top area (dark red)
        cv2.rectangle(overlay, (0, 0), (w, top), (100, 0, 0), -1)

        # Bottom area (dark red)
        cv2.rectangle(overlay, (0, bottom), (w, h), (100, 0, 0), -1)

        # Left area (dark red)
        cv2.rectangle(overlay, (0, top), (left, bottom), (100, 0, 0), -1)

        # Right area (dark red)
        cv2.rectangle(overlay, (right, top), (w, bottom), (100, 0, 0), -1)

        # Apply transparency
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

        # Draw kept area (green rectangle)
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 3)

        # Draw crop lines (yellow dashed)
        # Top line
        for x in range(0, w, 20):
            cv2.line(display_frame, (x, top), (min(x + 10, w), top), (0, 255, 255), 2)

        # Bottom line
        for x in range(0, w, 20):
            cv2.line(display_frame, (x, bottom), (min(x + 10, w), bottom), (0, 255, 255), 2)

        # Left line
        for y in range(top, bottom, 20):
            cv2.line(display_frame, (left, y), (left, min(y + 10, bottom)), (0, 255, 255), 2)

        # Right line
        for y in range(top, bottom, 20):
            cv2.line(display_frame, (right, y), (right, min(y + 10, bottom)), (0, 255, 255), 2)

        # Add text info
        self._add_info_text(display_frame)

        return display_frame

    def _add_info_text(self, frame):
        """Add textual information to the frame"""
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 220), (400, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        y = h - 195
        line_height = 20

        texts = [
            "=== HUD CROP TUNER ===",
            f"Top:    {self.top_crop:.2f} (W/S)",
            f"Bottom: {self.bottom_crop:.2f} (I/K)",
            f"Left:   {self.left_crop:.2f} (A/D)",
            f"Right:  {self.right_crop:.2f} (J/L)",
            "",
            "R: Reset | SPACE: Capture",
            "ENTER: Save | ESC: Exit"
        ]

        for text in texts:
            cv2.putText(frame, text, (20, y), font, font_scale, color, thickness)
            y += line_height

        # Display cropped area dimensions
        top_px = int(h * self.top_crop)
        bottom_px = int(h * (1 - self.bottom_crop))
        left_px = int(w * self.left_crop)
        right_px = int(w * (1 - self.right_crop))

        crop_w = right_px - left_px
        crop_h = bottom_px - top_px

        cv2.putText(
            frame,
            f"Zone: {crop_w}x{crop_h}px",
            (w - 180, 30),
            font,
            0.6,
            (0, 255, 0),
            2
        )

    def handle_key(self, key):
        """
        Handles keyboard inputs

        Args:
            key: OpenCV keycode

        Returns:
            action: 'continue', 'save', 'quit'
        """
        # W/S - Top crop
        if key == ord('w') or key == ord('W'):
            self.top_crop = max(0.0, self.top_crop - self.step)
            print(f"Top crop: {self.top_crop:.2f}")

        elif key == ord('s') or key == ord('S'):
            self.top_crop = min(0.5, self.top_crop + self.step)
            print(f"Top crop: {self.top_crop:.2f}")

        # I/K - Bottom crop
        elif key == ord('i') or key == ord('I'):
            self.bottom_crop = max(0.0, self.bottom_crop - self.step)
            print(f"Bottom crop: {self.bottom_crop:.2f}")

        elif key == ord('k') or key == ord('K'):
            self.bottom_crop = min(0.5, self.bottom_crop + self.step)
            print(f"Bottom crop: {self.bottom_crop:.2f}")

        # A/D - Left crop
        elif key == ord('a') or key == ord('A'):
            self.left_crop = max(0.0, self.left_crop - self.step)
            print(f"Left crop: {self.left_crop:.2f}")

        elif key == ord('d') or key == ord('D'):
            self.left_crop = min(0.5, self.left_crop + self.step)
            print(f"Left crop: {self.left_crop:.2f}")

        # J/L - Right crop
        elif key == ord('j') or key == ord('J'):
            self.right_crop = max(0.0, self.right_crop - self.step)
            print(f"Right crop: {self.right_crop:.2f}")

        elif key == ord('l') or key == ord('L'):
            self.right_crop = min(0.5, self.right_crop + self.step)
            print(f"Right crop: {self.right_crop:.2f}")

        # R - Reset
        elif key == ord('r') or key == ord('R'):
            self.reset_to_default()
            print("🔄 Reset to default values")

        # SPACE - New capture
        elif key == ord(' '):
            print("\n📸 Capturing a new frame...")
            frame = self.capture_frame_from_dolphin()
            if frame is not None:
                self.current_frame = frame
                print("✅ New frame captured")
            return 'capture'

        # ENTER - Save
        elif key == 13 or key == 10:  # Enter
            return 'save'

        # ESC - Quit
        elif key == 27:  # Escape
            return 'quit'

        return 'continue'

    def reset_to_default(self):
        """Reset to default values"""
        self.top_crop = self.default_config['top_crop']
        self.bottom_crop = self.default_config['bottom_crop']
        self.left_crop = self.default_config['left_crop']
        self.right_crop = self.default_config['right_crop']

    def save_config(self, filepath='config/crop_config.json'):
        """Save crop configuration to config/ (tracked, not user-specific)"""
        config = {
            'top_crop': self.top_crop,
            'bottom_crop': self.bottom_crop,
            'left_crop': self.left_crop,
            'right_crop': self.right_crop
        }

        # Create config directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n💾 Configuration saved: {filepath}")
        print(f"   top_crop: {self.top_crop:.2f}")
        print(f"   bottom_crop: {self.bottom_crop:.2f}")
        print(f"   left_crop: {self.left_crop:.2f}")
        print(f"   right_crop: {self.right_crop:.2f}")

    def load_config(self, filepath='config/crop_config.json'):
        """Load crop configuration from config/ (tracked file)"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config = json.load(f)

            self.top_crop = config['top_crop']
            self.bottom_crop = config['bottom_crop']
            self.left_crop = config['left_crop']
            self.right_crop = config['right_crop']

            print(f"📂 Configuration loaded: {filepath}")
            return True
        return False

    def run(self):
        """Launch the interactive tool"""
        print("\n" + "=" * 70)
        print("🎯 HUD CROP TUNER - MONSTER HUNTER TRI")
        print("=" * 70)
        print("\n📋 INSTRUCTIONS:")
        print("   1. Make sure Dolphin is running with the game")
        print("   2. Be IN-GAME (not in menus)")
        print("   3. Use keys to adjust the crop:")
        print("      - W/S : Top")
        print("      - I/K : Bottom")
        print("      - A/D : Left")
        print("      - J/L : Right")
        print("      - R : Reset")
        print("      - ESPACE : New capture")
        print("      - ENTRÉE : Save and exit")
        print("      - ESC : Exit without saving")
        print("\n💡 Goal: Remove HUD while keeping the monster !\n")

        # Try to load existing config
        self.load_config()

        input("Appuie sur ENTRÉE pour commencer...")

        # Capture the first frame
        print("\n📸 Capturing the first frame...")
        self.current_frame = self.capture_frame_from_dolphin()

        if self.current_frame is None:
            print("\n❌ Unable to capture a frame!")
            print("Checks:")
            print("   - Is Dolphin running?")
            print("   - Is the game visible?")
            return

        # Create OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        print("\n✅ Tool ready! Adjust the crop with the keys...")
        print("   (Check the OpenCV window)\n")

        # Main loop
        while True:
            # Draw overlay
            # Narrow Optional[ndarray] to ndarray (current_frame was already validated above, this satisfies the type checker)
            if self.current_frame is None:
                logger.error("HudCropTunerLoadError: self.current_frame became None during run loop")
                raise HudCropTunerLoadError("self.current_frame is None inside run loop")
            current_frame = self.current_frame
            display_frame = np.asarray(self.draw_crop_overlay(current_frame))

            # Convert RGB -> BGR for OpenCV
            # Wrap in np.asarray so cv2.imshow/imwrite receive a strict ndarray (fixes IDE union warnings)
            display_frame_bgr = np.asarray(cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

            # Display
            cv2.imshow(self.window_name, display_frame_bgr)

            # Wait for input (1 ms refresh)
            key = cv2.waitKey(1) & 0xFF

            if key != 255:  # If a key is pressed
                action = self.handle_key(key)

                if action == 'save':
                    self.save_config()

                    # Save example image to vision/debug/
                    # Config folder should only contain configuration files
                    example_dir = os.path.join('vision', 'debug')
                    os.makedirs(example_dir, exist_ok=True)
                    example_path = os.path.join(example_dir, 'crop_example.png')
                    cv2.imwrite(example_path, display_frame_bgr)
                    print(f"📸 Example saved: {example_path}")

                    break

                elif action == 'quit':
                    print("\n👋 Cancelled - configuration not saved")
                    break

        # Close window
        cv2.destroyAllWindows()

        print("\n✅ Done!")


# ============================================================
# SIMPLIFIED VERSION WITHOUT DOLPHIN CAPTURE (for testing purposes)
# ============================================================

def run_with_test_image(image_path: str = None):
    """
    Simplified version with a test image
    Useful if you cannot capture from Dolphin
    """
    run_tuner = HUDCropTuner()

    if image_path and os.path.exists(image_path):
        # Load provided image
        frame = cv2.imread(image_path)
        # Guard against imread failures: cv2.imread returns None on missing/corrupt files
        if frame is None:
            logger.error("HudCropTunerLoadError: failed to read test image, frame is None")
            raise HudCropTunerLoadError(f"Could not load image at {image_path}")
        # Force a concrete ndarray so type checkers stop complaining about Mat | UMat unions
        frame = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        run_tuner.current_frame = frame
        print(f"✅ Image loaded: {image_path}")
    else:
        # Create test image
        print("⚠️  No image provided - creating a test image")
        # Explicit ndarray cast to satisfy static type checker
        # (np.random.randint return type is inferred as a broad union by PyCharm)
        frame: np.ndarray = np.asarray(
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
            dtype=np.uint8,
        )

        # Simulate HUD elements
        # Top health bar
        cv2.rectangle(frame, (50, 20), (300, 60), (255, 0, 0), -1)
        cv2.putText(frame, "HP: 100/150", (60, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Left minimap
        cv2.circle(frame, (80, 400), 60, (0, 255, 0), -1)
        cv2.putText(frame, "MAP", (55, 410),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Bottom items
        for i in range(5):
            x = 50 + i * 70
            cv2.rectangle(frame, (x, 650), (x + 50, 700), (100, 100, 100), -1)

        # Monster in the center (keep  in the crop!)
        cv2.rectangle(frame, (500, 250), (800, 500), (150, 50, 200), -1)
        cv2.putText(frame, "MONSTRE", (550, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        run_tuner.current_frame = frame

    run_tuner.load_config()

    # Create window
    cv2.namedWindow(run_tuner.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(run_tuner.window_name, 1280, 720)

    print("\n✅ Test mode active - Adjust with keyboard keys")

    # Loop
    while True:
        # Narrow Optional[ndarray] to ndarray so the type checker accepts the draw_crop_overlay call
        if run_tuner.current_frame is None:
            logger.error("HudCropTunerLoadError: current_frame is None before draw_crop_overlay")
            raise HudCropTunerLoadError("run_tuner.current_frame is None, cannot draw overlay")
        current_frame = run_tuner.current_frame
        display_frame = np.asarray(run_tuner.draw_crop_overlay(current_frame))
        display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(run_tuner.window_name, display_frame_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key != 255:
            action = run_tuner.handle_key(key)

            if action == 'save':
                run_tuner.save_config()

                # Save to vision/debug/
                example_dir = os.path.join('vision', 'debug')
                os.makedirs(example_dir, exist_ok=True)
                example_path = os.path.join(example_dir, 'crop_example.png')
                cv2.imwrite(example_path, display_frame_bgr)
                print(f"📸 Example: {example_path}")
                break

            elif action == 'quit':
                print("\n👋 Cancelled")
                break

    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    print("\n🎮 HUD CROP TUNER")
    print("\nChoose a mode:")
    print("  1. Capture from Dolphin (recommended)")
    print("  2. Test mode with simulated image")

    if len(sys.argv) > 1:
        # Image fournie en argument
        run_with_test_image(sys.argv[1])
    else:
        choice = input("\nChoice (1 or 2): ").strip()

        if choice == '1':
            tuner = HUDCropTuner()
            tuner.run()
        else:
            run_with_test_image()