"""
episode_mixin.py — Episode lifecycle management.

Provides:
    reset()                     — full episode reset with save state reload
    _reload_save_state()        — send F5 to Dolphin (PostMessage)
    _should_reload_save_state() — check if reload is needed
    _check_terminated()         — 3-death detection
    _check_truncated()          — step timeout

Mixin attributes expected on `self`:
    instance_id, _agent_id, memory, use_memory, controller,
    auto_reload_save_state, save_state_slot, save_state_reload_count,
    _first_reset_done, _episode_ending, _vision_available, preprocessor,
    observation_space, frame_capture, reward_calc, prev_raw_memory,
    episode_start_time, episode_steps, total_steps, episode_count,
    total_reward, episode_reward, episode_length,
    _capture_running, _capture_thread, _obs_queue, rt_vision, rt_minimap
"""

import os
import time
import shutil
import threading

from info.agent_context import AgentContext, EnvContext
from info.module_logger import get_module_logger

logger = get_module_logger('episode')

try:
    from pynput.keyboard import Key
    _PYNPUT = True
except ImportError:
    _PYNPUT = False


# Fallback slots: if primary slot has expired quest, try these in order
FALLBACK_SLOTS = [6, 7, 8]

# Save state file extensions per slot number
_SLOT_EXTENSIONS = {5: ".s05", 6: ".s06", 7: ".s07", 8: ".s08"}

# Game ID used in save state filenames
_GAME_ID = "RMHP08"

# Max episode length before truncation
MAX_EPISODE_STEPS = 10_000


class EpisodeMixin:
    """Episode lifecycle management for MonsterHunterEnv."""

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Sequence:
            1. Wait for any post-quest cooldown
            2. Reload save state if needed (3 deaths / time expired / MAP=45)
            3. Reset controller, reward calculator, frame buffer
            4. Wait for game readiness
            5. (Re)start capture thread
            6. Return initial observation + info
        """
        super().reset(seed=seed)

        AgentContext.set_current_agent(self._agent_id)
        EnvContext.set_current_env(self.instance_id)
        logger.info(f"Reset episode #{self.episode_count + 1}...")

        # Clean stale state immediately
        self.prev_raw_memory = None

        # 1. Post-quest cooldown
        self._wait_quest_cooldown()

        # 2. Save state reload
        self._maybe_reload_save_state()

        # 3. Reset sub-systems
        self._reset_controller()
        self._reset_reward_calculator()

        # 4. Wait for game readiness (verify CURRENT_MAP)
        self._wait_game_ready()

        # 5. Reset frame buffer + episode counters
        if self._vision_available and self.preprocessor is not None:
            self.preprocessor.reset_stack()

        self.episode_start_time = time.time()
        self.episode_steps = 0
        self.total_reward = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.prev_raw_memory = None
        self._episode_ending = False

        # 6. (Re)start capture thread if needed
        self._ensure_capture_thread()

        time.sleep(0.1)  # Allow first frame to arrive

        # 7. Build initial observation
        observation = self._get_observation()
        if observation is None:
            observation = self._get_dummy_observation()

        # Verify observation keys match observation_space
        from gymnasium import spaces
        if isinstance(observation, dict) and isinstance(self.observation_space, spaces.Dict):
            missing = set(self.observation_space.spaces.keys()) - set(observation.keys())
            if missing:
                logger.debug(f"Missing keys in initial observation: {missing}")

        reset_info = self._get_info()
        self.current_state = observation

        from environment.sanitizer import sanitize_info
        reset_info = sanitize_info(reset_info)

        logger.info("Reset complete")
        return observation, reset_info

    # ------------------------------------------------------------------
    # Save state reload
    # ------------------------------------------------------------------

    def _reload_save_state(self):
        """
        Reload save state — PRIMARY SLOT ONLY (fast path).

        Only tries the primary slot (2 attempts) to avoid blocking
        other agents. If primary fails, returns False immediately
        and the caller should isolate this env + init fallback recovery.

        Returns:
            True   — reload confirmed (MAP=100, deaths<3, quest_time>10)
            False  — primary slot failed; caller should isolate and start
                     lazy fallback recovery via _attempt_next_fallback()
        """
        if not self.auto_reload_save_state:
            return False

        slot = self.save_state_slot
        slot_key = f'f{slot}'
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            logger.info(
                f"Reloading save state slot {slot} "
                f"({slot_key.upper()}) for instance #{self.instance_id} "
                f"— attempt {attempt}/{max_attempts}"
            )

            if not self._send_save_state_key(slot_key):
                return False

            self.save_state_reload_count += 1
            logger.info(f"Waiting for load (1.5s)...")
            time.sleep(1.5)

            result = self._verify_reload(attempt, max_attempts)
            if result is True:
                return True
            if result == 'expired':
                logger.warning(
                    f"Primary slot F{slot} has expired quest — "
                    f"will try fallbacks lazily after isolation"
                )
                return False
            if result is False:
                return False
            # result is None → retry
            if attempt < max_attempts:
                time.sleep(1.0)

        # Primary slot exhausted
        logger.warning(
            f"Instance #{self.instance_id}: primary slot F{slot} failed "
            f"after {max_attempts} attempts — isolating for lazy fallback recovery"
        )
        return False

    # ------------------------------------------------------------------
    # Lazy fallback recovery (called from trainer loop on isolated envs)
    # ------------------------------------------------------------------

    def _init_fallback_recovery(self):
        """Initialize state for lazy (one-slot-per-call) fallback recovery."""
        # Build list of fallback slots to try (excluding primary)
        self._fallback_queue = [
            s for s in FALLBACK_SLOTS if s != self.save_state_slot
        ]
        # Track which slots failed (for copy-back when one works)
        self._failed_slots = [self.save_state_slot]
        self._fallback_cross_user_pending = True  # try cross-user after local slots
        self._last_recovery_attempt = 0.0
        logger.info(
            f"Instance #{self.instance_id}: fallback recovery initialized "
            f"— will try slots {self._fallback_queue} then cross-user"
        )

    def _attempt_next_fallback(self) -> str:
        """
        Try the next fallback slot in the queue. Non-blocking for other envs
        because the trainer only calls this for isolated envs, one at a time.

        Returns:
            'recovered' — a valid save state was found + loaded + propagated
            'pending'   — this slot failed but more slots remain to try
            'exhausted' — all fallbacks (local + cross-user) have been tried and failed
        """
        import time as _time
        self._last_recovery_attempt = _time.time()

        # Phase 1: local fallback slots
        if hasattr(self, '_fallback_queue') and self._fallback_queue:
            slot = self._fallback_queue.pop(0)
            slot_key = f'f{slot}'

            logger.info(
                f"Instance #{self.instance_id}: trying fallback slot F{slot}..."
            )

            if not self._send_save_state_key(slot_key):
                self._failed_slots.append(slot)
                return 'pending' if (self._fallback_queue or self._fallback_cross_user_pending) else 'exhausted'

            self.save_state_reload_count += 1
            _time.sleep(2.0)

            result = self._verify_reload(attempt=1, max_attempts=1)
            if result is True:
                logger.info(
                    f"Instance #{self.instance_id}: fallback F{slot} SUCCESS! "
                    f"Propagating to failed slots: {self._failed_slots}"
                )
                self._propagate_working_save_state(slot)
                return 'recovered'

            logger.warning(
                f"Instance #{self.instance_id}: fallback F{slot} failed — "
                f"{'more slots remain' if self._fallback_queue else 'trying cross-user next'}"
            )
            self._failed_slots.append(slot)
            return 'pending' if (self._fallback_queue or self._fallback_cross_user_pending) else 'exhausted'

        # Phase 2: cross-user recovery
        if getattr(self, '_fallback_cross_user_pending', False):
            self._fallback_cross_user_pending = False
            logger.info(
                f"Instance #{self.instance_id}: trying cross-user save state recovery..."
            )
            if self._try_cross_user_recovery():
                logger.info(
                    f"Instance #{self.instance_id}: cross-user recovery SUCCESS! "
                    f"Propagating to all local slots"
                )
                # Cross-user: copy to ALL local slots since they all failed
                self._propagate_cross_user_save_state()
                return 'recovered'
            logger.error(
                f"Instance #{self.instance_id}: cross-user recovery also failed"
            )
            return 'exhausted'

        return 'exhausted'

    def _propagate_working_save_state(self, working_slot: int):
        """
        Copy a working save state file to all earlier failed slots.

        Example: if F7 works and F5/F6 failed, copy F7's file to F5 and F6.
        """
        dolphin_dir = self._get_dolphin_base_dir()
        if not dolphin_dir:
            logger.warning("Cannot propagate save state: dolphin_base_dir not set")
            return

        my_folder = self._get_user_folder_name()
        save_dir = os.path.join(dolphin_dir, my_folder, "StateSaves")
        src_ext = _SLOT_EXTENSIONS.get(working_slot, f".s{working_slot:02d}")
        src_file = os.path.join(save_dir, f"{_GAME_ID}{src_ext}")

        if not os.path.isfile(src_file):
            logger.warning(f"Source save state not found: {src_file}")
            return

        for failed_slot in self._failed_slots:
            if failed_slot == working_slot:
                continue
            dst_ext = _SLOT_EXTENSIONS.get(failed_slot, f".s{failed_slot:02d}")
            dst_file = os.path.join(save_dir, f"{_GAME_ID}{dst_ext}")
            try:
                shutil.copy2(src_file, dst_file)
                logger.info(
                    f"Propagated working save state F{working_slot} → F{failed_slot} "
                    f"for instance #{self.instance_id}"
                )
            except Exception as exc:
                logger.error(f"Failed to propagate F{working_slot} → F{failed_slot}: {exc}")

    def _propagate_cross_user_save_state(self):
        """
        After cross-user recovery, copy the recovered save state to ALL
        local slots (F5-F8) since they all failed.
        """
        dolphin_dir = self._get_dolphin_base_dir()
        if not dolphin_dir:
            return

        my_folder = self._get_user_folder_name()
        save_dir = os.path.join(dolphin_dir, my_folder, "StateSaves")

        # Find which slot file was actually written by cross-user recovery
        # (it overwrites the same slot it copied to)
        source_file = None
        source_slot = None
        for slot in [self.save_state_slot] + FALLBACK_SLOTS:
            ext = _SLOT_EXTENSIONS.get(slot, f".s{slot:02d}")
            candidate = os.path.join(save_dir, f"{_GAME_ID}{ext}")
            if os.path.isfile(candidate) and os.path.getsize(candidate) > 1024:
                source_file = candidate
                source_slot = slot
                break

        if not source_file:
            logger.warning("No valid source file found for cross-user propagation")
            return

        all_slots = [self.save_state_slot] + FALLBACK_SLOTS
        for target_slot in all_slots:
            if target_slot == source_slot:
                continue
            dst_ext = _SLOT_EXTENSIONS.get(target_slot, f".s{target_slot:02d}")
            dst_file = os.path.join(save_dir, f"{_GAME_ID}{dst_ext}")
            try:
                shutil.copy2(source_file, dst_file)
                logger.info(
                    f"Cross-user propagation: F{source_slot} → F{target_slot} "
                    f"for instance #{self.instance_id}"
                )
            except Exception as exc:
                logger.error(f"Cross-user propagation F{source_slot} → F{target_slot}: {exc}")

    # ------------------------------------------------------------------
    # Cross-user save state recovery
    # ------------------------------------------------------------------

    def _get_user_folder_name(self) -> str:
        """Return the Dolphin User folder name for this instance.

        Instance 0 → 'User', instance 1 → 'User1', etc.
        """
        if self.instance_id == 0:
            return "User"
        return f"User{self.instance_id}"

    def _get_dolphin_base_dir(self) -> str | None:
        """Resolve the Dolphin base directory from stored path or config."""
        # Prefer explicitly set attribute (passed from runner)
        if hasattr(self, 'dolphin_base_dir') and self.dolphin_base_dir:
            return self.dolphin_base_dir

        # Fallback: try to load from saved config
        try:
            from train.dolphin import resolve_dolphin_path
            path = resolve_dolphin_path(None)
            if path and os.path.isdir(path):
                return path
        except Exception:
            pass
        return None

    def _list_all_user_folders(self, dolphin_dir: str) -> list[str]:
        """List all UserX folders in the Dolphin directory, sorted."""
        folders = []
        for entry in os.listdir(dolphin_dir):
            full = os.path.join(dolphin_dir, entry)
            if not os.path.isdir(full):
                continue
            # Match "User", "User0", "User1", ... "User99"
            if entry == "User" or (
                entry.startswith("User") and entry[4:].isdigit()
            ):
                folders.append(entry)
        folders.sort(key=lambda n: -1 if n == "User" else int(n[4:]))
        return folders

    def _try_cross_user_recovery(self) -> bool:
        """
        Copy a valid save state file from another user profile, overwrite
        the corrupted one, then reload and verify.

        Tries each slot (F5→F8) from each other user until success.

        Returns True if recovery succeeded, False if all users are corrupted.
        """
        dolphin_dir = self._get_dolphin_base_dir()
        if not dolphin_dir:
            logger.error(
                "Cannot attempt cross-user recovery: "
                "dolphin_base_dir not set"
            )
            return False

        my_folder = self._get_user_folder_name()
        all_folders = self._list_all_user_folders(dolphin_dir)

        # Remove our own folder from the donor list
        donor_folders = [f for f in all_folders if f != my_folder]
        if not donor_folders:
            logger.error("No other user profiles found for cross-user recovery")
            return False

        # Slots to try, in priority order
        slots_to_recover = [self.save_state_slot] + [
            s for s in FALLBACK_SLOTS if s != self.save_state_slot
        ]

        for donor in donor_folders:
            for slot in slots_to_recover:
                ext = _SLOT_EXTENSIONS.get(slot, f".s{slot:02d}")
                donor_file = os.path.join(
                    dolphin_dir, donor, "StateSaves", f"{_GAME_ID}{ext}"
                )

                if not os.path.isfile(donor_file):
                    continue
                if os.path.getsize(donor_file) < 1024:
                    # Suspiciously small — skip
                    continue

                # Copy donor save state to our folder
                my_save_dir = os.path.join(
                    dolphin_dir, my_folder, "StateSaves"
                )
                os.makedirs(my_save_dir, exist_ok=True)
                target_file = os.path.join(my_save_dir, f"{_GAME_ID}{ext}")

                logger.warning(
                    f"Cross-user recovery: copying {donor}/{_GAME_ID}{ext} "
                    f"→ {my_folder}/{_GAME_ID}{ext}"
                )
                try:
                    shutil.copy2(donor_file, target_file)
                except Exception as exc:
                    logger.error(
                        f"Failed to copy save state from {donor}: {exc}"
                    )
                    continue

                # Reload this slot and verify
                slot_key = f'f{slot}'
                if not self._send_save_state_key(slot_key):
                    continue

                self.save_state_reload_count += 1
                time.sleep(2.0)  # Extra wait for cross-user load

                result = self._verify_reload(attempt=1, max_attempts=1)
                if result is True:
                    logger.info(
                        f"Cross-user recovery SUCCESS: "
                        f"using save state from {donor} slot F{slot} "
                        f"for instance #{self.instance_id}"
                    )
                    return True

                logger.warning(
                    f"Save state from {donor} slot F{slot} "
                    f"also invalid — trying next"
                )

        return False

    @staticmethod
    def _should_reload_save_state(current_state: dict) -> bool:
        """Check if a reload is needed based on game state."""
        current_map = current_state.get('current_map')
        death_count = current_state.get('death_count', 0) or 0
        quest_time = current_state.get('quest_time', 5400)

        if current_map == 45:
            logger.info("Reward screen detected (MAP=45) — reload needed")
            return True
        if quest_time is not None and quest_time <= 1:
            logger.info(f"Time expired (quest_time={quest_time}) — reload needed")
            return True
        if death_count >= 3:
            logger.info(f"3 deaths reached (count={death_count}) — reload needed")
            return True
        return False

    # ------------------------------------------------------------------
    # Termination checks
    # ------------------------------------------------------------------

    def _check_terminated(self, _observation) -> bool:
        """Check if the episode ended (3 deaths)."""
        if self.use_memory and self.memory:
            try:
                state = self.memory.read_game_state()
                death_count = state.get('death_count', 0) or 0
                if death_count >= 3:
                    logger.info("3 deaths reached — episode terminated")
                    return True
            except (AttributeError, KeyError, TypeError):
                pass
        return False

    def _check_truncated(self) -> bool:
        """Check if the episode should be truncated (step timeout)."""
        if self.episode_steps >= MAX_EPISODE_STEPS:
            logger.info(f"Timeout after {MAX_EPISODE_STEPS} steps")
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_quest_cooldown(self):
        """Wait for post-quest detection cooldown if set."""
        if hasattr(self, '_quest_end_cooldown_until'):
            wait = self._quest_end_cooldown_until - time.time()
            if wait > 0:
                time.sleep(min(wait, 3.0))
            delattr(self, '_quest_end_cooldown_until')
            logger.debug("Post-quest cooldown complete")

    def _maybe_reload_save_state(self):
        """Reload save state if game state requires it.

        On primary slot failure, immediately isolates this env and initializes
        lazy fallback recovery. The trainer loop will call _attempt_next_fallback()
        periodically to try remaining slots without blocking other agents.
        """
        if not (self.auto_reload_save_state and self.memory is not None):
            return

        try:
            if not self._first_reset_done:
                logger.debug(
                    f"First reset for instance #{self.instance_id} — "
                    f"forcing reload (DME may read wrong Dolphin in multi-instance)"
                )
                needs_reload = True
            else:
                state = self.memory.read_game_state()
                needs_reload = self._should_reload_save_state(state)

            if needs_reload:
                if not self._reload_save_state():
                    # Primary slot failed — isolate immediately, init lazy fallback
                    self._isolated = True
                    self._init_fallback_recovery()
                    logger.error(
                        f"ISOLATION: Instance #{self.instance_id} isolated after "
                        f"primary slot failure. Lazy fallback recovery started. "
                        f"Other agents continue training normally."
                    )
                    return  # Do not raise = let other agents continue

            self._first_reset_done = True

        except Exception as exc:
            logger.error(f"Reload check error: {exc}")
            self._isolated = True
            self._init_fallback_recovery()
            logger.error(
                f"ISOLATION: Instance #{self.instance_id} isolated "
                f"due to unexpected error: {exc}. Fallback recovery started."
            )

    def _reset_controller(self):
        """Release all controller inputs (including held multi-head keys)."""
        if self.use_controller and self.controller:
            self.controller.release_all_managed()  # Clear hold/release tracking state
            self.controller.reset_all()  # Zero out shared memory buffer

    def _reset_reward_calculator(self):
        """Reset reward calculator and exploration tracker."""
        if not self.reward_calc:
            return

        self.reward_calc.reset()

        if hasattr(self.reward_calc, 'exploration_tracker'):
            self.reward_calc.exploration_tracker.pause_creation(duration=1.5)
            logger.debug("Cube creation paused (episode reset)")

        # Double safety: clear all delta-tracking state
        self.reward_calc.prev_hp = None
        self.reward_calc.prev_stamina = None
        self.reward_calc.prev_damage_flag = None
        self.reward_calc.prev_position = None
        self.reward_calc.prev_zone = None
        self.reward_calc.prev_orientation = None
        self.reward_calc.prev_sharpness = None
        self.reward_calc.prev_oxygen = None
        self.reward_calc.prev_death_count = 0
        self.reward_calc.last_damage_time = 0.0

    def _wait_game_ready(self):
        """Poll CURRENT_MAP to ensure the game is in a valid state."""
        if self.memory is None:
            return

        max_attempts = 3
        current_map = None

        try:
            for attempt in range(max_attempts):
                current_map = self.memory.read_value('CURRENT_MAP')

                if current_map is None:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts}: "
                        f"CURRENT_MAP read failed"
                    )
                    time.sleep(1.0)
                    continue

                if current_map != 45:
                    logger.debug(f"CURRENT_MAP={current_map} (valid)")
                    return

                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts}: "
                    f"still on reward screen (MAP=45)"
                )

                if self.auto_reload_save_state and attempt < max_attempts - 1:
                    try:
                        if self._reload_save_state():
                            time.sleep(1.0)
                            current_map = self.memory.read_value('CURRENT_MAP')
                            if current_map != 45:
                                return
                    except Exception as exc:
                        logger.error(f"Reload in wait_game_ready failed: {exc}")

                time.sleep(1.0)

        except Exception as exc:
            logger.error(f"CURRENT_MAP check error: {exc}")

        if current_map == 45:
            logger.error(
                f"Still on reward screen after {max_attempts} attempts. "
                f"Press F5 manually or restart the quest."
            )

    def _ensure_capture_thread(self):
        """Start or restart the vision capture thread if needed."""
        if not self._vision_available:
            return

        thread_dead = (
            self._capture_thread is not None
            and not self._capture_thread.is_alive()
        )

        if not self._capture_running or thread_dead:
            if thread_dead:
                logger.warning("Capture thread died — restarting")
            self._capture_running = True
            self._capture_thread = threading.Thread(
                target=self._async_capture_loop,
                daemon=True,
                name=f"FrameCaptureThread-{self.instance_id}",
            )
            self._capture_thread.start()

    def _send_save_state_key(self, slot_key: str) -> bool:
        """Send a save state key via DLL controller or pynput fallback."""
        if self.controller and self.controller.is_connected:
            self.controller.send_raw_key(slot_key, duration=0.5)
            return True

        logger.error(
            f"Controller not connected for instance #{self.instance_id} — "
            f"cannot send {slot_key.upper()}"
        )
        return False

    def _verify_reload(self, attempt: int, max_attempts: int):
        """
        Verify save state reload succeeded.

        Returns:
            True      — confirmed OK (MAP=100, deaths<3, quest_time>10)
            False     — fatal failure (should NOT retry this slot)
            None      — transient failure (should retry)
            'expired' — reload OK but quest_time expired (try next slot)
        """
        if not (self.use_memory and self.memory):
            logger.warning("No memory reader — assuming reload succeeded")
            return True

        try:
            state = self.memory.read_game_state()
            current_map = state.get('current_map')
            deaths = state.get('death_count', 0) or 0
            quest_time = state.get('quest_time', 5400)

            if current_map == 100 and deaths < 3:
                if quest_time is not None and quest_time <= 10:
                    logger.warning(
                        f"Reload landed on expired quest (quest_time={quest_time}) "
                        f"— save state slot {self.save_state_slot} is stale"
                    )
                    return 'expired'

                logger.debug(
                    f"Reload confirmed: MAP=100, deaths={deaths}, "
                    f"quest_time={quest_time}"
                )
                return True

            # Log specific failure reason
            reasons = {
                0: "title/loading screen — save state not saved inside a quest",
                45: "still on reward screen",
            }
            reason = reasons.get(current_map, f"unexpected MAP={current_map}")

            if current_map == 100 and deaths >= 3:
                reason = f"MAP=100 but deaths={deaths} — save state has 3+ deaths"

            logger.warning(
                f"Reload rejected for instance #{self.instance_id}: {reason}"
            )

            # On max attempts: return False instead of raising RuntimeError
            # so the caller can proceed to fallbacks / cross-user recovery
            if attempt >= max_attempts:
                logger.error(
                    f"Instance #{self.instance_id}: reload failed after "
                    f"{max_attempts} attempts (MAP={current_map}, deaths={deaths})"
                )
                return False
            return None  # Retry

        except Exception as exc:
            logger.error(f"Reload verification error: {exc}")
            if attempt >= max_attempts:
                return False  # Don't raise — let caller handle gracefully
            return None
