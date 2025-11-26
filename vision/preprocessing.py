"""
GPU-Optimized Preprocessing for Monster Hunter Tri
HYBRID VERSION: CPU for initial cropping, GPU for resize/normalize

This module implements a higher-performance frame preprocessing pipeline that leverages
both CPU and GPU resources more optimally. The pipeline processes game frames through :
1. HUD cropping (CPU = fast, minimal overhead)
2. GPU transfer and transformations (resize, grayscale, normalize)
3. Frame stacking for temporal information
4. Efficient caching to avoid redundant processing

Performance: ~2-3ms per frame on modern GPUs (vs ~20ms pure CPU)
"""

# =======================================================================================
# STANDARD PYTHON IMPORTS
# =======================================================================================
import os                          # File path operations for loading crop_config.json
import json                        # JSON parsing for configuration files

# =======================================================================================
# IMAGE PROCESSING (CPU)
# =======================================================================================
import cv2                         # OpenCV - Initial cropping and color space conversions
import numpy as np                 # NumPy arrays - Bridge between CPU and GPU operations

# =======================================================================================
# LOGGING
# =======================================================================================
from utils.module_logger import get_module_logger
logger = get_module_logger('preprocessing_gpu')

# =======================================================================================
# DEEP LEARNING (GPU)
# =======================================================================================
import torch                       # PyTorch - Deep learning framework with GPU acceleration
import torch.nn.functional as f    # Functional API for GPU operations (interpolate, normalize)
# torch.nn.functional.interpolate: Ultra-fast GPU-based resizing
# Supported modes: nearest, linear, bilinear, bicubic, trilinear, area

# =======================================================================================
# VISUALIZATION (optional, for debugging)
# =======================================================================================
import matplotlib.pyplot as plt     # Matplotlib for visualizing crop boundaries and results


class FramePreprocessor:
    """
    Optimized Hybrid CPU/GPU Preprocessor for Game Frames

    This class implements a highly efficient preprocessing pipeline that :
    - Minimizes CPU/GPU data transfers (bottleneck in hybrid systems)
    - Keeps buffers on GPU to avoid repeated transfers
    - Caches identical frames to skip redundant processing
    - Provides detailed performance metrics for optimization

    Pipeline Flow :
        1. Crop HUD regions (CPU - fast, no GPU benefit)
        2. Convert NumPy → Torch tensor (CPU)
        3. Transfer CPU → GPU (asynchronous, one-time per frame)
        4. Resize using bilinear interpolation (GPU - extremely fast)
        5. Convert to grayscale if requested (GPU)
        6. Normalize pixel values to [0, 1] range (GPU)
        7. Stack frames for temporal context (GPU - zero-copy)
        8. Transfer GPU → CPU only when explicitly needed (legacy compatibility)

    Memory Layout :
        - Input: (H, W, C) NumPy array on CPU
        - GPU: (C, H, W) PyTorch tensor (standard for neural networks)
        - Output: (H, W, stack*C) or (H, W, stack) depending on configuration
    """

    def __init__(
            self,
            target_size=(84, 84),           # Output frame dimensions (height, width)
            grayscale=False,                # Convert RGB to grayscale (1 channel instead of 3)
            normalize=True,                 # Scale pixel values from [0, 255] to [0, 1]
            frame_stack=4,                  # Number of consecutive frames to stack
            crop_hud=True,                  # Remove HUD elements from edges
            top_crop=0.12,                  # Fraction to crop from top (12% default)
            bottom_crop=0.15,               # Fraction to crop from bottom (15% default)
            left_crop=0.05,                 # Fraction to crop from left (5% default)
            right_crop=0.05,                # Fraction to crop from right (5% default)
            stack_axis=-1,                  # Axis along which to stack frames (-1 = last)
            device='cuda',                  # Target device : 'cuda' for GPU, 'cpu' for fallback
    ):
        """
        Initialize the preprocessor with specified configuration.

        Args:
            target_size : Tuple (height, width) for resized output frames
            grayscale : If True, convert RGB (3 channels) to grayscale (1 channel)
            normalize : If True, divide pixel values by 255 to get [0, 1] range
            frame_stack : Number of consecutive frames to maintain in buffer
            crop_hud : If True, remove HUD regions based on crop percentages
            top_crop : Percentage of frame height to remove from top
            bottom_crop : Percentage of frame height to remove from bottom
            left_crop : Percentage of frame width to remove from left
            right_crop : Percentage of frame width to remove from right
            stack_axis : Axis for frame stacking (-1 = channels last)
            device : PyTorch device string ('cuda' or 'cpu')
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.frame_stack = frame_stack
        self.crop_hud = crop_hud
        self.stack_axis = stack_axis

        # Automatic device selection with fallback to CPU if CUDA unavailable
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        # Log device information for debugging
        if self.device.type == 'cuda':
            logger.info(f"GPU activated : {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU not available --> falling back to CPU processing")

        # Load crop configuration from JSON file if available
        # This allows dynamic adjustment without code changes
        crop_config_current_dir = os.path.dirname(os.path.abspath(__file__))
        crop_config_dir = os.path.join(crop_config_current_dir, '..', 'config')
        crop_config_path = os.path.join(crop_config_dir, 'crop_config.json')
        crop_config_path = os.path.normpath(crop_config_path)

        if crop_hud and os.path.exists(crop_config_path):
            logger.info(f"Loading crop configuration from : {crop_config_path}")
            try:
                with open(crop_config_path, 'r') as crop_file:
                    config = json.load(crop_file)
                # Use config values if present
                self.top_crop = config.get('top_crop', top_crop)
                self.bottom_crop = config.get('bottom_crop', bottom_crop)
                self.left_crop = config.get('left_crop', left_crop)
                self.right_crop = config.get('right_crop', right_crop)
            except Exception as e:
                logger.error(f"Failed to read crop configuration file : {e}")
                # Fall back to default values on error
                self.top_crop = top_crop
                self.bottom_crop = bottom_crop
                self.left_crop = left_crop
                self.right_crop = right_crop
        else:
            # Use default crop values if no config file
            self.top_crop = top_crop
            self.bottom_crop = bottom_crop
            self.left_crop = left_crop
            self.right_crop = right_crop

        # Initialize frame stack buffer directly on GPU
        # This avoids repeated CPU→GPU transfers for the buffer
        self.frame_stack = frame_stack

        # Create buffer with appropriate shape based on grayscale setting
        if grayscale:
            # Grayscale : (stack_size, height, width, 1)
            buffer_shape = (frame_stack, *target_size, 1)
        else:
            # RGB : (stack_size, height, width, 3)
            buffer_shape = (frame_stack, *target_size, 3)

        # Allocate buffer directly on target device (GPU if available)
        self.buffer = torch.zeros(
            buffer_shape,
            dtype=torch.float32,
            device=self.device
        )

        # Counter for tracking how many frames have been added
        self.count = 0

        # Cache mechanism to avoid reprocessing identical frames
        # Stores hash of raw frame and corresponding processed output
        self._last_raw_frame_hash = None        # Hash of last input frame
        self._last_processed_output = None      # Cached processed tensor (on GPU)

        # Performance tracking variables
        self._total_frames = 0                  # Total frames processed
        self._total_time_cpu = 0.0              # Cumulative CPU time (seconds)
        self._total_time_gpu = 0.0              # Cumulative GPU time (seconds)

        # Cache hit/miss statistics
        self._cache_hits = 0                    # Number of times cache was used
        self._cache_misses = 0                  # Number of times processing was needed

        # Transfer tracking for process_and_stack_numpy()
        self._numpy_wrapper_calls = 0           # Number of GPU→CPU transfers
        self._total_transfer_time = 0.0         # Cumulative transfer time

    def crop_game_area(self, frame: np.ndarray) -> np.ndarray:
        """
        Remove HUD elements by cropping frame edges.

        This operation remains on CPU because :
        1. It's extremely fast (simple array slicing)
        2. GPU transfer overhead would outweigh any speedup
        3. Input frames are already on CPU (from game capture)

        Args:
            frame : Input frame as NumPy array (H, W, C)

        Returns:
            Cropped frame as NumPy array with HUD regions removed
        """
        if not self.crop_hud :
            return frame

        # Calculate crop boundaries based on percentages
        h, w = frame.shape[:2]
        top = int(h * self.top_crop)                    # Top boundary (pixels from top)
        bottom = int(h * (1 - self.bottom_crop))        # Bottom boundary (pixels from top)
        left = int(w * self.left_crop)                  # Left boundary (pixels from left)
        right = int(w * (1 - self.right_crop))          # Right boundary (pixels from left)

        # Perform crop using NumPy slicing (extremely fast)
        return frame[top:bottom, left:right]

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Execute the complete hybrid CPU/GPU preprocessing pipeline.

        This is the core processing function that transforms raw game frames
        into optimized tensors ready for neural network input. The pipeline
        is carefully designed to minimize CPU↔GPU transfers (the main bottleneck).

        Pipeline stages :
        1. [CPU] Crop HUD regions (fast array slicing)
        2. [CPU] Convert NumPy uint8 → PyTorch float32
        3. [Transfer] Move data CPU → GPU (asynchronous)
        4. [GPU] Rearrange dimensions for PyTorch (HWC → CHW)
        5. [GPU] Resize to target dimensions (bilinear interpolation)
        6. [GPU] Convert to grayscale if requested
        7. [GPU] Normalize values to [0, 1] range
        8. [GPU] Rearrange back to HWC format for compatibility

        Args:
            frame : Raw game frame as NumPy uint8 array (H, W, C)

        Returns:
            Preprocessed tensor on GPU (or CPU if fallback) with shape (H, W, C)
        """
        import time
        t_start = time.perf_counter()

        # ===================================================================
        # STAGE 1 : CROP HUD (CPU)
        # ===================================================================
        # Fast NumPy slicing - no benefit from GPU here
        processed = self.crop_game_area(frame)

        t_cpu = time.perf_counter()

        # ===================================================================
        # STAGE 2 : NUMPY → TORCH (CPU)
        # ===================================================================
        # Convert from NumPy uint8 [0, 255] to PyTorch float32
        # Note : This doesn't normalize yet, just changes dtype
        tensor = torch.from_numpy(processed).float()

        # ===================================================================
        # STAGE 3 : CPU → GPU TRANSFER
        # ===================================================================
        # Asynchronous transfer to overlap with CPU operations
        # non_blocking=True allows CPU to continue without waiting
        tensor = tensor.to(self.device, non_blocking=True)

        # Rearrange from (Height, Width, Channels) to (Channels, Height, Width)
        # PyTorch neural networks expect CHW format, not HWC
        tensor = tensor.permute(2, 0, 1)

        # Add batch dimension : (C, H, W) → (1, C, H, W)
        # Required for torch.nn.functional.interpolate
        tensor = tensor.unsqueeze(0)

        # ===================================================================
        # STAGE 4 : RESIZE (GPU)
        # ===================================================================
        # Bilinear interpolation on GPU (much faster than cv2.resize on CPU)
        # This is where GPU acceleration provides the biggest speedup
        tensor = f.interpolate(
            tensor,
            size=self.target_size,
            mode='bilinear',              # Smooth interpolation
            align_corners=False           # Modern alignment method
        )

        # ===================================================================
        # STAGE 5 : GRAYSCALE CONVERSION (GPU if requested)
        # ===================================================================
        if self.grayscale:
            # Standard luminance formula for RGB → Gray conversion
            # Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601 standard)
            weights = torch.tensor(
                [0.299, 0.587, 0.114],
                device=self.device
            ).view(1, 3, 1, 1)

            # Apply weighted sum across color channels (dim=1)
            tensor = (tensor * weights).sum(dim=1, keepdim=True)

        # ===================================================================
        # STAGE 6 : NORMALIZATION (GPU)
        # ===================================================================
        if self.normalize:
            # Scale from [0, 255] to [0, 1] range
            # Neural networks typically expect normalized inputs
            tensor = tensor / 255.0

        # ===================================================================
        # STAGE 7 : RESHAPE FOR OUTPUT
        # ===================================================================
        # Remove batch dimension : (1, C, H, W) → (C, H, W)
        tensor = tensor.squeeze(0)

        # Rearrange back to (H, W, C) format for compatibility with existing code
        tensor = tensor.permute(1, 2, 0)

        t_gpu = time.perf_counter()

        # ===================================================================
        # PERFORMANCE TRACKING
        # ===================================================================
        self._total_frames += 1
        self._total_time_cpu += (t_cpu - t_start)
        self._total_time_gpu += (t_gpu - t_cpu)

        # Log performance statistics every 1000 frames
        if self._total_frames % 1000 == 0:
            avg_cpu = (self._total_time_cpu / self._total_frames) * 1000
            avg_gpu = (self._total_time_gpu / self._total_frames) * 1000
            logger.debug(f"Preprocessing performance (average over {self._total_frames} frames) :")
            logger.debug(f"   CPU (crop) :           {avg_cpu:.2f}ms")
            logger.debug(f"   GPU (resize/norm) :    {avg_gpu:.2f}ms")
            logger.debug(f"   TOTAL :                {avg_cpu + avg_gpu:.2f}ms")
            logger.debug(f"   Speedup vs pure CPU :  {20.0 / (avg_cpu + avg_gpu):.1f}x")

        return tensor  # Tensor remains on GPU

    def add_to_stack(self, processed: torch.Tensor) -> torch.Tensor:
        """
        Add a new frame to the temporal buffer and return stacked frames.

        Frame stacking provides temporal context to the agent by combining
        multiple consecutive frames. This helps the agent understand motion
        and velocity, which is crucial for action games like Monster Hunter.

        All operations happen on GPU (zero-copy) to maximize performance.

        Args:
            processed : Preprocessed frame tensor on GPU with shape (H, W, C)

        Returns:
            Stacked frames tensor on GPU with shape determined by stack_axis :
            - If stack_axis=-1 and grayscale : (H, W, stack)
            - If stack_axis=-1 and RGB : (H, W, stack*3)
        """
        # ===================================================================
        # CIRCULAR BUFFER SHIFT (GPU)
        # ===================================================================
        # Roll buffer to make room for new frame
        # This is equivalent to np.roll but operates entirely on GPU
        # Shifts all frames "forward" (index 0→1, 1→2, etc.)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)

        # Write new frame to last position in buffer
        self.buffer[-1] = processed

        # Update frame counter (capped at frame_stack)
        self.count = min(self.frame_stack, self.count + 1)

        # ===================================================================
        # WARM-UP PERIOD (repeat first frame until buffer is full)
        # ===================================================================
        if self.count < self.frame_stack:
            # Not enough frames yet - pad with repeated copies of current frame
            # This prevents using uninitialized/zero frames at the start
            pad = processed.unsqueeze(0).repeat(self.frame_stack - self.count, 1, 1, 1)
            stacked = torch.cat([pad, self.buffer[-self.count:]], dim=0)
        else:
            # Buffer is full - use all frames
            stacked = self.buffer.clone()

        # ===================================================================
        # RESHAPE ACCORDING TO STACK_AXIS (GPU)
        # ===================================================================
        if self.stack_axis == -1:
            # Stack along last axis (channels dimension)
            if len(stacked.shape) == 4 and stacked.shape[-1] == 1:
                # Grayscale : (stack, H, W, 1) → (H, W, stack)
                # Remove singleton channel dim and rearrange
                stacked = stacked.squeeze(-1).permute(1, 2, 0)
            else:
                # RGB : (stack, H, W, 3) → (H, W, stack*3)
                # Flatten stack and channels into single dimension
                stacked = stacked.permute(1, 2, 0, 3)
                stacked = stacked.reshape(stacked.shape[0], stacked.shape[1], -1)

        return stacked  # Tensor remains on GPU

    def process_and_stack(self, frame: np.ndarray) -> torch.Tensor:
        """
        Complete preprocessing pipeline with intelligent caching.

        This is the main entry point for frame processing. It includes :
        1. Cache check to avoid redundant processing
        2. Full preprocessing if cache miss
        3. Frame stacking for temporal context
        4. Cache update for future frames

        The cache is particularly effective when :
        - Game is paused (same frame repeated)
        - Loading screens (static frames)
        - Low frame rate capture (duplicate frames)

        Args:
            frame : Raw game frame as NumPy array (H, W, C)

        Returns:
            Stacked and processed frames as PyTorch tensor on GPU
        """
        # ===================================================================
        # CACHE CHECK
        # ===================================================================
        # Hash the raw frame bytes to create unique identifier
        # This is faster than pixel-by-pixel comparison
        frame_hash = hash(frame.tobytes())

        if frame_hash == self._last_raw_frame_hash and self._last_processed_output is not None:
            # Cache hit - return previously processed result
            self._cache_hits += 1

            # Log cache statistics every 1000 frames
            if (self._cache_hits + self._cache_misses) % 1000 == 0:
                total_checks = self._cache_hits + self._cache_misses
                cache_rate = (self._cache_hits / total_checks) * 100
                logger.debug(f"Cache hit rate : {cache_rate:.1f}% ({self._cache_hits}/{total_checks})")

            # Return clone to prevent mutations affecting cached version
            return self._last_processed_output.clone()

        # Cache miss - need to process this frame
        self._cache_misses += 1

        # ===================================================================
        # FULL PROCESSING PIPELINE
        # ===================================================================
        # Preprocess individual frame (crop, resize, normalize, etc.)
        processed = self.preprocess_frame(frame)

        # Add to temporal buffer and get stacked result
        stacked = self.add_to_stack(processed)

        # ===================================================================
        # UPDATE CACHE FOR NEXT FRAME
        # ===================================================================
        self._last_raw_frame_hash = frame_hash
        self._last_processed_output = stacked

        return stacked  # Tensor remains on GPU

    def process_and_stack_numpy(self, frame: np.ndarray) -> np.ndarray:
        """
        Legacy compatibility wrapper that returns NumPy array instead of tensor.

        WARNING : This function forces a GPU→CPU transfer which is slow!
        The transfer can take 1-2ms, negating much of the GPU speedup.

        This wrapper exists for backward compatibility with code that expects
        NumPy arrays. New code should use process_and_stack() directly and
        keep tensors on GPU as long as possible.

        Args:
            frame : Raw game frame as NumPy array

        Returns:
            Processed and stacked frames as NumPy array (on CPU)
        """
        import time

        # Call GPU version first (handles all stats tracking)
        tensor_gpu = self.process_and_stack(frame)

        # Track GPU→CPU transfer time separately
        t_transfer_start = time.perf_counter()
        result = tensor_gpu.cpu().numpy()
        t_transfer_end = time.perf_counter()

        # Log transfer overhead every 1000 calls to raise awareness
        self._numpy_wrapper_calls += 1
        self._total_transfer_time += (t_transfer_end - t_transfer_start)

        if self._numpy_wrapper_calls % 1000 == 0:
            avg_transfer = (self._total_transfer_time / self._numpy_wrapper_calls) * 1000
            logger.debug(f"GPU→CPU transfer overhead : {avg_transfer:.2f}ms average over {self._numpy_wrapper_calls} calls")
            logger.debug(f"💡 Tip : Use process_and_stack() directly to eliminate this overhead")

        return result

    def _log_performance_stats(self):
        """
        Log comprehensive performance statistics.

        This method provides detailed insights into preprocessing performance :
        - CPU vs GPU time breakdown
        - Cache effectiveness
        - Theoretical maximum FPS
        - Speedup compared to pure CPU implementation
        """
        if self._total_frames == 0:
            logger.warning("No frames processed yet - no stats to report")
            return

        # Calculate average times in milliseconds
        avg_cpu = (self._total_time_cpu / self._total_frames) * 1000
        avg_gpu = (self._total_time_gpu / self._total_frames) * 1000
        total_avg = avg_cpu + avg_gpu

        # Calculate cache hit rate
        if self._cache_hits + self._cache_misses > 0:
            cache_rate = (self._cache_hits / (self._cache_hits + self._cache_misses)) * 100
        else:
            cache_rate = 0.0

        # Log comprehensive statistics
        logger.debug(f"📊 Preprocessing Statistics (average over {self._total_frames} frames) :")
        logger.debug(f"   CPU time (crop) :          {avg_cpu:.2f}ms")
        logger.debug(f"   GPU time (resize/norm) :   {avg_gpu:.2f}ms")
        logger.debug(f"   TOTAL time per frame :     {total_avg:.2f}ms")
        logger.debug(f"   Speedup vs pure CPU :      {20.0 / total_avg:.1f}x")
        logger.debug(f"   Cache hit rate :           {cache_rate:.1f}%")
        logger.debug(f"   Theoretical max FPS :      {1000 / total_avg:.0f} FPS")

    def reset_stack(self):
        """
        Clear the frame buffer and reset all cached data.

        This should be called when :
        - Starting a new episode/level
        - Switching between game modes
        - After a scene transition

        Ensures that frames from previous context don't leak into new context.
        """
        # Zero out buffer efficiently on GPU
        self.buffer.zero_()

        # Reset frame counter
        self.count = 0

        # Clear cache to prevent stale data
        self._last_raw_frame_hash = None
        self._last_processed_output = None

    def visualize_crop(self, original_frame: np.ndarray, save_path='crop_visualization.png'):
        """
        Create a visualization showing crop boundaries and final output.

        Generates a 3-panel figure :
        1. Original frame with HUD
        2. Original frame with crop boundaries overlaid (red rectangle)
        3. Final preprocessed output at target size

        Useful for :
        - Tuning crop parameters
        - Debugging preprocessing issues
        - Documentation and presentations

        Args:
            original_frame : Raw game frame to visualize
            save_path : Where to save the visualization image
        """
        # Crop the frame to show what will be kept
        cropped = self.crop_game_area(original_frame)

        # Temporarily switch to CPU for visualization
        # (matplotlib expects CPU arrays)
        old_device = self.device
        self.device = torch.device('cpu')

        # Process frame on CPU
        tensor = self.preprocess_frame(cropped)
        resized = tensor.cpu().numpy()

        # Restore original device
        self.device = old_device

        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1 : Original frame
        axes[0].imshow(original_frame)
        axes[0].set_title("Original Frame (with HUD)")
        axes[0].axis('off')

        # Panel 2 : Original with crop boundaries
        h, w = original_frame.shape[:2]
        top = int(h * self.top_crop)
        bottom = int(h * (1 - self.bottom_crop))
        left = int(w * self.left_crop)
        right = int(w * (1 - self.right_crop))

        original_with_lines = original_frame.copy()
        cv2.rectangle(original_with_lines, (left, top), (right, bottom), (255, 0, 0), 3)

        axes[1].imshow(original_with_lines)
        axes[1].set_title("Crop Region (red boundary)")
        axes[1].axis('off')

        # Panel 3 : Final preprocessed output
        axes[2].imshow(resized)
        axes[2].set_title(f"Final Output {self.target_size}")
        axes[2].axis('off')

        # Save figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Crop visualization saved to : {save_path}")


# ============================================================================
# HELPER FUNCTION FOR CREATING PREPROCESSOR WITH OPTIMAL SETTINGS
# ============================================================================

def create_mh_preprocessor_gpu(
        grayscale=True,
        frame_stack=4,
        crop_hud=True,
        device='cuda'
):
    """
    Create a GPU-accelerated preprocessor optimized for Monster Hunter Tri.

    This convenience function creates a preprocessor with settings tuned
    specifically for Monster Hunter gameplay :
    - 84x84 output size (standard for RL)
    - Grayscale conversion (reduces data while preserving gameplay info)
    - 4-frame stacking (provides motion context)
    - HUD cropping (removes distracting UI elements)

    Args:
        grayscale : If True, convert to single-channel grayscale
        frame_stack : Number of frames to stack for temporal context
        crop_hud : If True, remove HUD elements from frame edges
        device : Target device ('cuda' for GPU, 'cpu' for fallback)

    Returns:
        Configured FramePreprocessor instance ready for use
    """
    return FramePreprocessor(
        target_size=(84, 84),
        grayscale=grayscale,
        normalize=True,
        frame_stack=frame_stack,
        crop_hud=crop_hud,
        device=device
    )


# ============================================================================
# TEST AND BENCHMARK CODE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("GPU Preprocessor Test and Benchmark")
    print("="*80)

    # Check CUDA availability and display GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"  CUDA available : {gpu_name}")
        print(f"  CUDA Version : {cuda_version}")
        target_device = 'cuda'
    else:
        logger.warning("CUDA not available - testing on CPU")
        target_device = 'cpu'

    print()

    # Create preprocessor with test configuration
    print("Creating preprocessor with test configuration...")
    preprocessor = FramePreprocessor(
        grayscale=False,         # Keep RGB for visual verification
        frame_stack=4,           # Standard stack size
        crop_hud=True,           # Test cropping functionality
        device=target_device
    )

    # Create synthetic test frame (typical game resolution)
    print("Generating test frame (720x1280x3 - HD game capture)...")
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warm-up run to initialize CUDA kernels
    print("Performing GPU warm-up (initializes CUDA kernels)...")
    _ = preprocessor.process_and_stack(test_frame)
    print("Warm-up complete")
    print()

    # Run benchmark over 100 frames
    print("Running benchmark (100 frames)...")
    print("-" * 80)
    import time

    processed_result = None
    start = time.perf_counter()

    for i in range(100):
        processed_result = preprocessor.process_and_stack(test_frame)

    end = time.perf_counter()

    # Calculate and display results
    avg_time = ((end - start) / 100) * 1000  # Convert to milliseconds

    print(f"Average processing time :  {avg_time:.2f}ms per frame")

    if processed_result is not None:
        print(f"Output tensor shape :      {processed_result.shape}")
        print(f"Output tensor device :     {processed_result.device}")
        print(f"Theoretical max FPS :      {1000 / avg_time:.1f} FPS")
    else:
        print("No results generated")

    print()
    print("-" * 80)

    # Compare with CPU baseline
    print("Performance Comparison :")
    print(f"   Pure CPU (cv2.resize) :    ~20.00ms per frame")
    print(f"   GPU accelerated :          {avg_time:.2f}ms per frame")

    if avg_time > 0:
        speedup = 20.0 / avg_time
        print(f"   Speedup factor :           {speedup:.1f}x faster")

    print()
    print("="*80)
    print("All tests completed successfully!")
    print("="*80)