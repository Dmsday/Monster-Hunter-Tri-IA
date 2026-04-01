"""
Memory Reader v1.5
"""

import ctypes
import ctypes.wintypes
import time
import os
import struct
import config.memory_addresses as addr
import threading
import queue

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from core.memory_state_builder import build_game_state
from info.module_logger import get_module_logger
logger = get_module_logger('dynamic_memory_reader')

try:
    import dolphin_memory_engine as dme
    DME_AVAILABLE = True
except ImportError:
    dme = None
    DME_AVAILABLE = False
    logger.warning("(DME) dolphin-memory-engine non installé")

import threading as _threading

# --- Multi-instance DME fix ---
# dme is a global C++ singleton: it can only be hooked to ONE Dolphin process at a time.
# Several async reader threads (one per instance) would overwrite each other's hook,
# causing all agents to read from the same (wrong) Dolphin process.
# Fix: serialize ALL dme calls with a global lock + re-hook to the correct PID before reading.
_dme_global_lock = _threading.Lock()
_dme_hooked_pid: "int | None" = None  # Which PID dme is currently hooked to
_dme_pid_warning_logged: bool = False  # Whether the "PID not supported" warning was already shown


def _ensure_dme_pid(target_pid: "int | None") -> bool:
    # This DME version does not support PID-targeted hooks.
    # Instance isolation is not available — all agents share the first Dolphin's memory.
    # This function is kept as a no-op to preserve the call sites.
    return True

class _DolphinDirectReader:
    """
    Reads Dolphin emulated Wii RAM directly via ReadProcessMemory.
    Used for per-PID isolation when DME does not support hook(pid).

    Address translation:
        MEM1 game 0x80000000-0x817FFFFF  ->  mem1_base + (addr - 0x80000000)
        MEM2 game 0x90000000-0x93FFFFFF  ->  mem2_base + (addr - 0x90000000)
    """

    _PROCESS_ALL = 0x001F0FFF
    _MEM_COMMIT  = 0x1000
    _MEM_MAPPED  = 0x40000
    _MEM1_SIZE   = 0x1800000   # 24 MB
    _MEM2_SIZE   = 0x4000000   # 64 MB

    class _MBI(ctypes.Structure):
        _fields_ = [
            ("BaseAddress",       ctypes.c_ulonglong),
            ("AllocationBase",    ctypes.c_ulonglong),
            ("AllocationProtect", ctypes.c_uint32),
            ("_align1",           ctypes.c_uint32),
            ("RegionSize",        ctypes.c_ulonglong),
            ("State",             ctypes.c_uint32),
            ("Protect",           ctypes.c_uint32),
            ("Type",              ctypes.c_uint32),
            ("_align2",           ctypes.c_uint32),
        ]

    def __init__(self, pid: int):
        self._pid    = pid
        self._k32    = ctypes.WinDLL('kernel32', use_last_error=True)
        self._handle = None
        self._mem1   = None
        self._mem2   = None
        self._connect()
        self._discover()

    def _connect(self):
        h = self._k32.OpenProcess(self._PROCESS_ALL, False, self._pid)
        if not h:
            raise OSError(f"OpenProcess failed for PID {self._pid} "
                          f"(error {ctypes.get_last_error()})")
        self._handle = h

    def _discover(self):
        """Scan Dolphin VA space for MEM_MAPPED regions matching MEM1/MEM2 sizes."""
        mbi  = self._MBI()
        addr = 0
        limit = 1 << 47   # 128 TB user VA on 64-bit Windows

        while addr < limit:
            sz = self._k32.VirtualQueryEx(
                self._handle, ctypes.c_void_p(addr),
                ctypes.byref(mbi), ctypes.sizeof(mbi))
            if sz == 0:
                break

            if mbi.State == self._MEM_COMMIT and mbi.Type == self._MEM_MAPPED:
                if mbi.RegionSize == self._MEM1_SIZE and self._mem1 is None:
                    self._mem1 = mbi.BaseAddress
                    logger.debug(f"[Direct] PID {self._pid}: MEM1 base = 0x{mbi.BaseAddress:016X}")
                elif mbi.RegionSize == self._MEM2_SIZE and self._mem2 is None:
                    self._mem2 = mbi.BaseAddress
                    logger.debug(f"[Direct] PID {self._pid}: MEM2 base = 0x{mbi.BaseAddress:016X}")

            next_addr = (mbi.BaseAddress or addr) + mbi.RegionSize
            if next_addr <= addr:
                break
            addr = next_addr

        if self._mem1 is None:
            raise ConnectionError(
                f"Could not find MEM1 region in PID {self._pid}. "
                "Is Dolphin fully loaded with a game?")

    def _translate(self, game_addr: int) -> int:
        if 0x80000000 <= game_addr < 0x81800000:
            if self._mem1 is None:
                raise RuntimeError("MEM1 not mapped")
            return self._mem1 + (game_addr - 0x80000000)
        if 0x90000000 <= game_addr < 0x94000000:
            if self._mem2 is None:
                raise RuntimeError("MEM2 not mapped")
            return self._mem2 + (game_addr - 0x90000000)
        raise ValueError(f"Address 0x{game_addr:08X} outside known Wii RAM ranges")

    def _read_raw(self, game_addr: int, size: int) -> bytes | None:
        try:
            proc_addr = self._translate(game_addr)
        except (RuntimeError, ValueError):
            return None
        buf   = (ctypes.c_byte * size)()
        nread = ctypes.c_size_t(0)
        ok    = self._k32.ReadProcessMemory(
            self._handle, ctypes.c_void_p(proc_addr),
            buf, size, ctypes.byref(nread))
        if not ok or nread.value != size:
            return None
        return bytes(buf)

    def read_float(self, game_addr: int):
        b = self._read_raw(game_addr, 4)
        return struct.unpack('>f', b)[0] if b else None

    def read_word(self, game_addr: int):
        b = self._read_raw(game_addr, 4)
        return struct.unpack('>I', b)[0] if b else None

    def read_short_signed(self, game_addr: int):
        b = self._read_raw(game_addr, 2)
        return struct.unpack('>h', b)[0] if b else None

    def read_byte(self, game_addr: int):
        b = self._read_raw(game_addr, 1)
        return b[0] if b else None

    def close(self):
        if self._handle:
            self._k32.CloseHandle(self._handle)
            self._handle = None

class MemoryReader:
    """
     Memory Reader v1.6 avec mode asynchrone optionnel
    """

    def __init__(
            self,
            force_quest_mode=True,
            async_mode=False,
            read_frequency=100,
            target_pid: int = None,
            instance_id: int = 0,
    ):
        """
        Args:
            force_quest_mode: Forcer mode quête
            async_mode: Si True, active lecture asynchrone (non-bloquante)
            read_frequency: Fréquence de lecture en Hz (pour mode async)
        """
        if not DME_AVAILABLE:
            raise ImportError("dolphin-memory-engine requis !")

        # Store target PID before connect_with_retry so it can use it
        self._target_pid = target_pid
        self._instance_id = instance_id

        self.connected = False

        # Adresses
        self.addresses = {}
        self.address_types = {}
        self.dual_addresses = {}

        self._discover_addresses()

        # Charger depuis utils/ (path absolu)
        self.item_names = self._load_item_names()
        logger.info(f"{len(self.item_names)} noms d'items chargés depuis item_id.txt")

        logger.info(f"{len(self.addresses)} adresses découvertes")

        # Connection
        self.connect_with_retry(max_attempts=3)

        # Per-PID direct reader (bypasses DME for multi-instance isolation)
        self._direct: "_DolphinDirectReader | None" = None
        if self._target_pid is not None:
            try:
                self._direct = _DolphinDirectReader(self._target_pid)
                logger.info(f"Direct memory reader active for PID {self._target_pid} "
                            "(per-instance isolation enabled)")
            except Exception as _dr_err:
                logger.warning(f"Direct reader failed for PID {self._target_pid}: {_dr_err} "
                               "— falling back to DME (instance isolation broken)")

        # Forcer mode quête
        if force_quest_mode:
            self.switch_to_quest_mode()
            self.in_quest = True
            logger.info("QUEST MODE FORCED")
        else:
            self.in_quest = True

        # Support mode asynchrone
        self.async_mode = async_mode
        self.read_frequency = read_frequency

        if self.async_mode:
            # Queue pour stocker states
            self._state_queue = queue.Queue(maxsize=8)
            self._last_valid_state = None

            # Thread de lecture
            self._async_running = False
            self._async_thread = None

            # Stats
            self._async_reads_total = 0
            self._async_reads_failed = 0

            logger.debug(f"Asynchronous mode enabled ({read_frequency} Hz)")

        # Démarrer thread async si demandé
        if self.async_mode:
            self._start_async_reading()

    @staticmethod
    def _load_item_names() -> dict:
        """
        Charge item_id.txt depuis utils/ avec gestion des objets non renseignés
        """
        item_names = {}

        # Path absolu vers utils/item_id.txt
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..')
        item_id_path = os.path.join(project_root, 'utils', 'item_id.txt')

        # Normaliser le path
        item_id_path = os.path.normpath(item_id_path)

        if not os.path.exists(item_id_path):
            logger.warning(f"item_id.txt non trouvé : {item_id_path}")
            logger.warning("Utilisation IDs bruts")
            return {}

        try:
            with open(item_id_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Format : ID = Nom OU ID = (pas encore mis)
                    if '=' in line:
                        parts = line.split('=', 1)
                        try:
                            item_id = int(parts[0].strip())
                            item_name = parts[1].strip()

                            # Gérer item non renseigné
                            if item_name == "(pas encore mis)" or item_name == "":
                                item_name = f"Item ID {item_id}"

                            item_names[item_id] = item_name

                        except ValueError:
                            logger.warning(f"Ligne {line_num} invalide : {line}")
                            continue

        except Exception as item_id_file_read_error:
            logger.warning(f"Erreur lecture item_id.txt : {item_id_file_read_error}")

        return item_names

    def _discover_addresses(self):
        """
        Découvre adresses
        """
        for name in dir(addr):
            if name.startswith('_'):
                continue

            value = getattr(addr, name)

            if isinstance(value, tuple) and len(value) == 2:
                self.dual_addresses[name] = value
                self.addresses[name] = value[1]  # Quête par défaut
                self.address_types[name] = self._infer_type(name)
            elif isinstance(value, int) and value > 0:
                self.addresses[name] = value
                self.address_types[name] = self._infer_type(name)

        # S'assurer que IN_GAME_MENU_IS_OPEN est découvert
        if 'IN_GAME_MENU_IS_OPEN' in dir(addr):
            self.addresses['IN_GAME_MENU_IS_OPEN'] = addr.IN_GAME_MENU_IS_OPEN
            self.address_types['IN_GAME_MENU_IS_OPEN'] = 'byte'

        # S'assurer que les orientations NS et EW sont découvertes
        if 'PLAYER_NS_ORIENTATION' in dir(addr):
            self.addresses['PLAYER_NS_ORIENTATION'] = addr.PLAYER_NS_ORIENTATION
            self.address_types['PLAYER_NS_ORIENTATION'] = 'float'
        if 'PLAYER_EW_ORIENTATION' in dir(addr):
            self.addresses['PLAYER_EW_ORIENTATION'] = addr.PLAYER_EW_ORIENTATION
            self.address_types['PLAYER_EW_ORIENTATION'] = 'float'

    @staticmethod
    def _infer_type(name: str) -> str:
        """Infère type"""
        name_lower = name.lower()

        if 'hp' in name_lower or 'stamina' in name_lower:
            return 'int4'
        if 'damage' in name_lower:
            return 'float'
        if any(x in name_lower for x in ['player_x', 'player_y', 'player_z', 'orientation']):
            return 'float'
        if 'zone' in name_lower or 'death' in name_lower:
            return 'byte'
        if 'slot' in name_lower:
            return 'int2'
        if 'money' in name_lower or 'point' in name_lower:
            return 'int4'
        if 'sharpness' in name_lower:
            return 'int2'
        if 'quest_time' in name_lower:
            return 'int4'
        if 'underwater' in name_lower:
            return 'int2'
        if 'current_map' in name_lower:
            return 'int4'
        if 'item_selected' in name_lower:
            return 'int2'

        return 'int4'

    def switch_to_quest_mode(self):
        """Bascule vers quête"""
        for name, (village_addr, quest_addr) in self.dual_addresses.items():
            self.addresses[name] = quest_addr
        self.in_quest = True

    def connect_with_retry(self, max_attempts=3):
        """
        Connection with bug handling hook()
        """
        logger.info("Login to Dolphin...")

        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt + 1}/{max_attempts}...")
            try:
                with _dme_global_lock:
                    global _dme_hooked_pid
                    try:
                        dme.un_hook()
                    except (RuntimeError, AttributeError):
                        pass
                    # hook() return value is always None in this DME version — skip the check
                    # PID argument is not supported — always hook to first Dolphin found
                    dme.hook()
                    _dme_hooked_pid = None

                if not hasattr(dme, 'is_hooked') or not dme.is_hooked():
                    if attempt < max_attempts - 1:
                        time.sleep(1)
                        continue
                    raise ConnectionError("is_hooked() returned False after all attempts")

                test_byte = dme.read_byte(0x80000000)
                logger.debug(f"Test reading successful: {test_byte}")
                logger.info("Connection CONFIRMED by reading!")
                self.connected = True
                return

            except ConnectionError:
                raise
            except Exception as dolphin_hook_error:
                logger.error(f"Error: {dolphin_hook_error}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    raise ConnectionError(f"Unable to connect: {dolphin_hook_error}")

    def read_value(self, address_name: str):
        if not self.connected or address_name not in self.addresses:
            return None

        addr_val  = self.addresses[address_name]
        data_type = self.address_types[address_name]

        # Fast path: per-PID direct reader (no lock needed, fully independent)
        if self._direct is not None:
            try:
                if data_type == 'float':
                    return self._direct.read_float(addr_val)
                elif data_type == 'int4':
                    return self._direct.read_word(addr_val)
                elif data_type == 'int2':
                    return self._direct.read_short_signed(addr_val)
                elif data_type == 'byte':
                    return self._direct.read_byte(addr_val)
            except Exception:
                return None
            return None

        # Fallback: shared DME (single-instance mode)
        try:
            with _dme_global_lock:
                if data_type == 'float':
                    return dme.read_float(addr_val)
                elif data_type == 'int4':
                    return dme.read_word(addr_val)
                elif data_type == 'int2':
                    raw_bytes = dme.read_bytes(addr_val, 2)
                    return struct.unpack('>h', raw_bytes)[0]
                elif data_type == 'byte':
                    return dme.read_byte(addr_val)
        except (KeyError, struct.error, RuntimeError):
            return None

    def is_quest_active(self) -> bool:
        """
        Vérifie si une quête est en cours

        Returns:
            True si en quête, False si écran de fin
        """
        current_map = self.read_value('CURRENT_MAP')

        # Si lecture échoue, considérer comme actif par sécurité
        if current_map is None:
            return True

        # CURRENT_MAP = 45 = pas en quete
        return current_map != 45

    def is_on_reward_screen(self) -> bool:
        """
        Détecte si on est hors quête

        Returns:
            True si écran de fin détecté
        """
        return not self.is_quest_active()

    def _start_async_reading(self):
        """Démarre le thread de lecture asynchrone"""
        if self._async_running:
            return

        self._async_running = True
        self._async_thread = threading.Thread(
            target=self._async_read_loop,
            daemon=True,
            name="AsyncMemoryThread"
        )
        self._async_thread.start()
        logger.debug("Asynchronous thread started")

    def _async_read_loop(self):
        """
        Continuous memory read loop (separate thread).
        """
        from info.agent_context import AgentContext
        AgentContext.set_current_agent(self._instance_id)

        read_interval = 1.0 / self.read_frequency

        while self._async_running:
            try:
                # Lire state (bloque CE thread, pas le principal)
                state = build_game_state(self)

                self._async_reads_total += 1

                if state is not None:
                    self._last_valid_state = state

                    # Mettre à jour queue
                    try:
                        self._state_queue.put(state, block=False)
                    except queue.Full:
                        # Queue pleine : vider et remettre
                        try:
                            self._state_queue.get_nowait()
                            self._state_queue.put(state, block=False)
                        except queue.Empty:
                            pass
                else:
                    self._async_reads_failed += 1

                # Rate limiting
                time.sleep(read_interval)

            except TypeError as type_error_in_async_read_loop:
                # Protection spécifique pour erreurs de type (None + float, etc.)
                logger.error(f"Erreur de type dans lecture async: {type_error_in_async_read_loop}")
                self._async_reads_failed += 1
                time.sleep(read_interval)
                continue

            except (ConnectionError, TimeoutError, ValueError):
                self._async_reads_failed += 1
                time.sleep(read_interval)
                continue

    def get_latest_state(self):
        """
        Récupère la dernière state (NON-BLOQUANT en mode async).

        - Si async_mode=False : délègue à build_game_state() directement (sync).
        - Si async_mode=True  : tente de lire depuis la queue.
          - Queue non vide → retourne la dernière state produite par le thread.
          - Queue vide + _last_valid_state disponible → retourne la copie de la dernière state.
          - Queue vide + aucune state encore → appelle build_game_state() directement.
            NOTE : on appelle build_game_state(self) et PAS read_game_state(),
                   car read_game_state() rappellerait get_latest_state() → récursion infinie.
        """
        if not self.async_mode:
            return build_game_state(self)

        # Mode async : récupérer depuis queue (non-bloquant)
        try:
            return self._state_queue.get(block=False)
        except queue.Empty:
            pass

        # Queue vide : retourner dernière state valide si disponible
        if self._last_valid_state is not None:
            return self._last_valid_state.copy()

        # Aucune state en cache : lecture synchrone directe (évite la récursion)
        # On appelle build_game_state(self), PAS self.read_game_state() qui bouclerait
        return build_game_state(self)

    def stop_async_reading(self):
        """Arrête le thread asynchrone"""
        if not self.async_mode or not self._async_running:
            return

        logger.info("[INFO] Arrêt thread asynchrone...")
        self._async_running = False

        if self._async_thread and self._async_thread.is_alive():
            self._async_thread.join(timeout=2.0)

        # Stats
        if self._async_reads_total > 0:
            success_rate = ((self._async_reads_total - self._async_reads_failed) /
                            self._async_reads_total) * 100
            logger.info(f"[INFO] Stats async:")
            logger.info(f"Lectures: {self._async_reads_total}")
            logger.info(f"Succès: {success_rate:.1f}%")

    def read_game_state(self) -> dict:
        """
        Point d'entrée unifié (supporte les deux modes)

        Returns:
            Dict avec game state
        """
        if self.async_mode:
            # Mode async : récupérer depuis queue
            return self.get_latest_state()
        else:
            # Mode sync classique
            return build_game_state(self)

    def get_training_features(self) -> dict:
        """Features pour l'IA"""
        state = self.read_game_state()
        # In get_training_features
        inventory_items = self.read_inventory()  # Read once
        inventory_vector = self.get_inventory_vector()  # Convert in memory
        return {
            'player_hp': state['player_hp'],
            'player_hp_recoverable': state['player_hp_recoverable'],
            'player_stamina': state['player_stamina'],
            'player_hp_raw': state['player_hp_raw'],
            'player_stamina_raw': state['player_stamina_raw'],
            'player_x': state['player_x'],
            'player_y': state['player_y'],
            'player_z': state['player_z'],
            'player_orientation': state['player_orientation'],
            'current_zone': state['current_zone'],
            'damage_last_hit': state['damage_last_hit'],
            'money': state['money'],
            'death_count': state['death_count'],
            'stamina_low': state['stamina_low'],
            'quest_time': state['quest_time'],
            'sharpness': state['sharpness'],
            'in_game_menu': state.get('in_game_menu', False),
            'inventory_vector': inventory_vector,
            'inventory_items': inventory_items,
            'time_underwater': state.get('time_underwater'),
            'oxygen_valid': state.get('oxygen_valid', False)
        }

    def read_inventory(self) -> list:
        """ Lecture inventaire complet"""
        inventory = []

        # Lire tout les slots
        for slot_num in range(1, 25):
            id_name = f'ID_SLOT_{slot_num}'
            qty_name = f'NOITEM_SLOT_{slot_num}'

            if id_name not in self.addresses or qty_name not in self.addresses:
                continue

            try:
                item_id = self.read_value(id_name)
                quantity = self.read_value(qty_name)

                # Vérification robuste
                if item_id is not None and item_id > 0:
                    inventory.append({
                        'slot': slot_num,
                        'item_id': item_id,
                        'quantity': quantity if quantity is not None else 0,
                        'name': self._get_item_name(item_id)
                    })

            except (ValueError, TypeError, KeyError):
                # Silencieux pour éviter spam
                continue

        return inventory

    def get_inventory_vector(self) -> list:
        """
        Vecteur pour 24 slots
        """
        vector = []

        for slot_num in range(1, 25):
            id_name = f'ID_SLOT_{slot_num}'
            qty_name = f'NOITEM_SLOT_{slot_num}'

            try:
                item_id = self.read_value(id_name) or 0
                quantity = self.read_value(qty_name) or 0

                # Normalisation prudente
                item_id_norm = min(item_id / 746.0, 1.0)  # Cap à 1.0
                quantity_norm = min(quantity / 99.0, 1.0)  # Cap à 1.0

                vector.extend([item_id_norm, quantity_norm])

            except (TypeError, ZeroDivisionError, KeyError):
                vector.extend([0.0, 0.0])

        return vector

    def _get_item_name(self, item_id: int) -> str:
        """
        Retourne le nom depuis item_id.txt (avec fallback Item ID X)
        """
        return self.item_names.get(item_id, f"Item ID {item_id}")

# Test
if __name__ == "__main__":
    print("TEST MEMORY READER\n")

    if not DME_AVAILABLE:
        print("dolphin-memory-engine not installed — pip install dolphin-memory-engine")
        exit(1)

    try:
        reader = MemoryReader(force_quest_mode=True, async_mode=False)

        print("\n" + "=" * 70)
        print("LIVE GAME STATE")
        print("=" * 70)

        state = reader.read_game_state()

        # --- Player stats ---
        print("\n[Player]")
        hp = state.get('player_hp')
        print(f"  HP           : {hp:.1f}" if hp is not None else "  HP           : N/A")
        hp_rec = state.get('player_hp_recoverable')
        print(f"  HP (recover) : {hp_rec:.1f}" if hp_rec is not None else "  HP (recover) : N/A")
        sta = state.get('player_stamina')
        print(f"  Stamina      : {sta:.1f}" if sta is not None else "  Stamina      : N/A")
        print(f"  Deaths       : {state.get('death_count', 'N/A')}")
        print(f"  Money        : {state.get('money', 'N/A')}")
        print(f"  Sharpness    : {state.get('sharpness', 'N/A')}")
        print(f"  In menu      : {state.get('in_game_menu', 'N/A')}")
        print(f"  Item selected: {state.get('item_selected', 'N/A')}")

        # --- Position & orientation ---
        print("\n[Position]")
        print(f"  X            : {state.get('player_x', 'N/A')}")
        print(f"  Y            : {state.get('player_y', 'N/A')}")
        print(f"  Z            : {state.get('player_z', 'N/A')}")
        print(f"  Orientation  : {state.get('player_orientation', 'N/A')}°")
        print(f"  Zone         : {state.get('current_zone', 'N/A')}")

        # --- Quest state ---
        print("\n[Quest]")
        print(f"  Map ID       : {state.get('current_map', 'N/A')}  (45 = reward screen)")
        print(f"  Time left    : {state.get('quest_time', 'N/A')}s")
        print(f"  Quest ended  : {state.get('quest_ended', False)}")

        # --- Oxygen ---
        print("\n[Oxygen]")
        oxy = state.get('time_underwater')
        oxy_valid = state.get('oxygen_valid', False)
        if oxy is not None:
            print(f"  Level        : {oxy}  (valid={oxy_valid})")
            if oxy_valid and oxy < 25:
                print("  WARNING      : oxygen critically low!")
        else:
            print("  Level        : N/A (not underwater or read failed)")

        # --- Monsters ---
        print("\n[Monsters]")
        any_monster = False
        for i in range(1, 6):
            hp_m = state.get(f'smonster{i}_hp')
            if hp_m is not None and hp_m > 0:
                print(f"  Small {i}       : {hp_m} HP")
                any_monster = True
        lm = state.get('lmonster1_hp')
        if lm is not None and lm > 0:
            print(f"  Large 1       : {lm} HP")
            any_monster = True
        if not any_monster:
            print("  (none detected in current zone)")

        # --- Inventory ---
        print("\n[Inventory]")
        inventory = reader.read_inventory()
        if inventory:
            for item in inventory:
                slot = item.get('slot', '?')
                name = item.get('name', f"Item ID {item.get('item_id', '?')}")
                qty  = item.get('quantity', '?')
                print(f"  Slot {slot:2}  :  {name:35s}  x{qty}")
        else:
            print("  (empty or not in quest)")

        print("\n" + "=" * 70)
        print("Test successful!")

    except ConnectionError as e:
        print(f"\nConnection error: {e}")
        print("Solutions:")
        print("  1. Launch Dolphin as Administrator")
        print("  2. Load Monster Hunter Tri and enter a quest")
        print("  3. Re-run this script as Administrator")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()