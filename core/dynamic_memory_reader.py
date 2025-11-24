"""
Memory Reader v1.5
"""

import time
import os
import struct
import config.memory_addresses as addr
import threading              # Thread pour lecture asynchrone
import queue                  # Queue thread-safe

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from utils.module_logger import get_module_logger
logger = get_module_logger('dynamic_memory_reader')

try:
    import dolphin_memory_engine as dme
    DME_AVAILABLE = True
except ImportError:
    dme = None
    DME_AVAILABLE = False
    logger.warning("(DME) dolphin-memory-engine non installé")


class MemoryReader:
    """
     Memory Reader v1.6 avec mode asynchrone optionnel
    """

    def __init__(
            self,
            force_quest_mode=True,
            async_mode=False,
            read_frequency=100,
    ):
        """
        Args:
            force_quest_mode: Forcer mode quête
            async_mode: Si True, active lecture asynchrone (non-bloquante)
            read_frequency: Fréquence de lecture en Hz (pour mode async)
        """
        if not DME_AVAILABLE:
            raise ImportError("dolphin-memory-engine requis !")

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

        # Connexion
        self.connect_with_retry(max_attempts=3)

        # Forcer mode quête
        if force_quest_mode:
            self.switch_to_quest_mode()
            self.in_quest = True
            logger.info("MODE QUETE FORCE")
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

            logger.info(f"Mode asynchrone activé ({read_frequency} Hz)")

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

            logger.info(f"Chargé {len(item_names)} items depuis {item_id_path}")

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
        """Connexion avec gestion du bug hook()"""
        logger.info("Connexion à Dolphin...")

        for attempt in range(max_attempts):
            logger.info(f"Tentative {attempt + 1}/{max_attempts}...")

            try:
                try:
                    dme.un_hook()
                    time.sleep(0.2)
                except (RuntimeError, AttributeError):
                    pass

                hook_result = dme.hook()
                logger.info(f"hook() retourne: {hook_result} (type: {type(hook_result)})")

                if hook_result is None:
                    logger.warning(f"hook() retourne None (bug connu)")
                    logger.info(f"Vérification avec is_hooked()...")

                if hasattr(dme, 'is_hooked'):
                    is_connected = dme.is_hooked()
                    logger.info(f"is_hooked() retourne: {is_connected}")

                    if not is_connected:
                        logger.warning(f"is_hooked() retourne False")
                        if attempt < max_attempts - 1:
                            time.sleep(1)
                            continue
                        else:
                            raise ConnectionError("is_hooked() retourne False après 3 tentatives")
                else:
                    logger.warning(f"is_hooked() non disponible")

                logger.info(f"Test lecture 0x80000000...")
                try:
                    test_byte = dme.read_byte(0x80000000)
                    logger.info(f"Lecture test réussie: {test_byte}")
                    logger.info(f"Connexion CONFIRMEE par lecture !")
                    self.connected = True
                    return

                except Exception as read_error:
                    logger.error(f"Lecture test Échoue: {read_error}")

                    if attempt >= max_attempts - 1:
                        raise ConnectionError(
                            f"Lecture impossible après {max_attempts} tentatives."
                            f"SOLUTIONS:"
                            f"1. Dolphin lancé EN ADMIN ?"
                            f"2. Monster Hunter Tri chargé ?"
                            f"3. EN JEU (pas dans un menu) ?"
                        )

                    logger.info(f"RETRY Nouvelle tentative dans 1s...")
                    time.sleep(1)

            except Exception as dolphin_hook_error:
                logger.error(f"[ERREUR] Erreur: {dolphin_hook_error}")

                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    logger.error(f"Échec après {max_attempts} tentatives")
                    raise ConnectionError(f"Impossible de se connecter: {dolphin_hook_error}")

    def read_value(self, address_name: str):
        """Lit une valeur avec gestion erreur robuste"""
        if not self.connected or address_name not in self.addresses:
            return None

        try:
            addr_val = self.addresses[address_name]
            data_type = self.address_types[address_name]

            if data_type == 'float':
                return dme.read_float(addr_val)
            elif data_type == 'int4':
                return dme.read_word(addr_val)
            elif data_type == 'int2':
                raw_bytes = dme.read_bytes(addr_val, 2)
                value = struct.unpack('>h', raw_bytes)[0]  # signed
                return value
            elif data_type == 'byte':
                return dme.read_byte(addr_val)

        except (KeyError, struct.error, RuntimeError):
            # Logging silencieux pour éviter spam
            return None

    @staticmethod
    def normalize_stamina(raw_value: int) -> float:
        """Normalise stamina"""
        if raw_value is None:
            return 0.0

        min_stamina = 787032
        max_stamina = 39322200

        if max_stamina == min_stamina:
            return 0.0

        normalized = ((raw_value - min_stamina) / (max_stamina - min_stamina)) * 100.0
        return max(0.0, normalized)

    @staticmethod
    def normalize_hp(raw_value: int) -> float:
        """Normalise HP"""
        if raw_value is None:
            return 0.0

        max_hp = 2516608000
        min_hp = 2516582400

        if max_hp == min_hp:
            return 0.0

        normalized = ((raw_value - min_hp) / (max_hp - min_hp)) * 100.0
        return max(0.0,normalized)

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
        logger.info("Thread asynchrone démarré")

    def _async_read_loop(self):
        """
        Boucle de lecture continue (thread séparé)
        """
        read_interval = 1.0 / self.read_frequency

        while self._async_running:
            try:
                # Lire state (bloque CE thread, pas le principal)
                state = self._read_game_state_internal()

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
        Récupère la dernière state (NON-BLOQUANT en mode async)

        Returns:
            Dict avec game state
        """
        if not self.async_mode:
            # Mode synchrone classique
            return self.read_game_state()

        # Mode async : récupérer depuis queue
        try:
            state = self._state_queue.get(block=False)
            return state
        except queue.Empty:
            # Queue vide : retourner dernière state valide
            if self._last_valid_state is not None:
                return self._last_valid_state.copy()
            else:
                # Fallback : lecture synchrone
                return self.read_game_state()

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

    def _read_game_state_internal(self) -> dict:
        """
        État complet
        """
        state = {}

        # ===================================================================
        # PARTIE 0 : DÉTECTION FIN DE QUÊTE (PRIORITAIRE)
        # ===================================================================

        # Lire CURRENT_MAP pour détecter écran de fin
        current_map = self.read_value('CURRENT_MAP')
        state['current_map'] = current_map

        # Si CURRENT_MAP = 45, on est sortie de la quete
        if current_map == 45:
            state['quest_ended'] = True
            state['on_reward_screen'] = True
            logger.info(f"FIN DE QUÊTE DÉTECTÉE (CURRENT_MAP = 45)")
            # Logger les autres valeurs pour debug
            logger.info(f"HP: {state.get('player_hp', '?')}")
            logger.info(f"Zone: {state.get('current_zone', '?')}")
            logger.info(f"Deaths: {state.get('death_count', '?')}")
            logger.info(f"Quest time: {state.get('quest_time', '?')}")
        else:
            state['quest_ended'] = False
            state['on_reward_screen'] = False

        # ===================================================================
        # PARTIE 1 : STATS DE BASE
        # ===================================================================

        stamina_raw = self.read_value('PLAYER_CURRENT_STAMINA')
        hp_raw = self.read_value('PLAYER_CURRENT_HP')
        hp_rec_raw = self.read_value('PLAYER_RECOVERABLE_HP')
        orientation_ns_raw = self.read_value('PLAYER_NS_ORIENTATION')
        orientation_ew_raw = self.read_value('PLAYER_EW_ORIENTATION')

        state['player_stamina_raw'] = stamina_raw
        state['player_hp_raw'] = hp_raw
        state['player_hp_recoverable_raw'] = hp_rec_raw

        state['player_stamina'] = self.normalize_stamina(stamina_raw)
        state['player_hp'] = self.normalize_hp(hp_raw)
        state['player_hp_recoverable'] = self.normalize_hp(hp_rec_raw) if hp_rec_raw else 0.0

        state['stamina_low'] = state['player_stamina'] < 25 if state['player_stamina'] else False

        # Position
        player_x = self.read_value('PLAYER_X')
        player_y = self.read_value('PLAYER_Y')
        player_z = self.read_value('PLAYER_Z')

        state['player_x'] = player_x
        state['player_y'] = player_y
        state['player_z'] = player_z

        # Orientation en degrés
        orientation_deg = self._convert_orientation_to_degrees(
            orientation_ns_raw,
            orientation_ew_raw
        )

        state['player_orientation'] = orientation_deg

        # Autres base
        state['current_zone'] = self.read_value('CURRENT_ZONE')
        state['damage_last_hit'] = self.read_value('DAMAGE_RECEIVE_LAST_HIT')
        state['money'] = self.read_value('PLAYER_MONEY')
        state['death_count'] = self.read_value('DEATH_COUNTER')
        state['player_stamina_max'] = self.read_value('PLAYER_STAMINA_MAX')

        # ===================================================================
        # PARTIE 2 : FEATURES QUETES
        # ===================================================================

        # QUEST TIME (converti en secondes)
        quest_time_raw = self.read_value('QUEST_TIME_SPENT')
        if quest_time_raw is not None:
            state['quest_time'] = int(quest_time_raw / 30)  # Frames → secondes
        else:
            state['quest_time'] = None

        # ATTACK & DEFENSE (valeur brute combinée)
        state['attack_defense_value'] = self.read_value('ATTACK_AND_DEFENSE_VALUE')

        # SHARPNESS (valeur brute)
        state['sharpness'] = self.read_value('SHARPNESS')

        # IN GAME MENU (valeur brute - 0 ou 1)
        in_menu_raw = self.read_value('IN_GAME_MENU_IS_OPEN')
        state['in_game_menu'] = (in_menu_raw == 1) if in_menu_raw is not None else False

        # ITEM SELECTED (slot sélectionné 0-23, ou 24 si rien)
        item_selected_raw = self.read_value('ITEM_SELECTED')
        if item_selected_raw is not None:
            # Valeur brute: 0-23 = slots 1-24, 24 = rien
            state['item_selected'] = item_selected_raw
        else:
            state['item_selected'] = 24  # Défaut = rien de sélectionné

        # ===================================================================
        # PARTIE 3 : HP MONSTRES (BRUT) + DÉTECTION
        # ===================================================================

        # Lire HP de tous les small monsters (1-5)
        state['smonster1_hp'] = self.read_value('SMONSTER1_HP')
        state['smonster2_hp'] = self.read_value('SMONSTER2_HP')
        state['smonster3_hp'] = self.read_value('SMONSTER3_HP')
        state['smonster4_hp'] = self.read_value('SMONSTER4_HP')
        state['smonster5_hp'] = self.read_value('SMONSTER5_HP')

        # Large monster (si implémenté)
        state['lmonster1_hp'] = self.read_value('LMONSTER1_HP')

        # ===================================================================
        # PARTIE 4 : OXYGÈNE
        # ===================================================================

        oxygen_raw = self.read_value('TIME_SPENT_UNDERWATER')

        if oxygen_raw is not None:
            try:
                oxygen_value = int(oxygen_raw)

                if 0 <= oxygen_value <= 200:
                    state['time_underwater'] = oxygen_value
                    state['oxygen_valid'] = True
                    state['oxygen_low_warning'] = oxygen_value < 25
                    state['oxygen_critical_warning'] = oxygen_value < 10
                else:
                    state['time_underwater'] = None
                    state['oxygen_valid'] = False
                    state['oxygen_error'] = f'out_of_range_{oxygen_value}'

            except (ValueError, TypeError) as oxygen_value:
                state['time_underwater'] = None
                state['oxygen_valid'] = False
                state['oxygen_error'] = f'conversion_error_{oxygen_value}'
        else:
            state['time_underwater'] = None
            state['oxygen_valid'] = False
            state['oxygen_error'] = 'read_failed'

        # ===================================================================
        # PARTIE 5 : INVENTAIRE COMPLET (24 SLOTS)
        # ===================================================================

        state['inventory_items'] = self.read_inventory()

        return state

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
            return self._read_game_state_internal()

    @staticmethod
    def _convert_orientation_to_degrees(orientation_ns: float, orientation_ew: float) -> float:
        """
        Convertit orientation NS/EW (-1 à 1) en degrés (0-360°)

        Convention:
        - NS: -1 = Nord, +1 = Sud
        - EW: -1 = Ouest, +1 = Est
        - Résultat: 0° = Nord, 90° = Est, 180° = Sud, 270° = Ouest

        Args:
            orientation_ns: Valeur Nord-Sud (-1 à 1)
            orientation_ew: Valeur Est-Ouest (-1 à 1)

        Returns:
            Angle en degrés (0-360°, 1 décimale)
        """
        # PROTECTION : Si valeurs None, retourner 0.0
        if orientation_ns is None or orientation_ew is None:
            return 0.0

        import math

        # Calculer l'angle avec atan2
        # atan2(y, x) où y=EW et x=-NS (négatif car NS est inversé)
        angle_rad = math.atan2(orientation_ew, -orientation_ns)

        # Convertir en degrés
        angle_deg = math.degrees(angle_rad)

        # Normaliser 0-360° (atan2 retourne -180 à 180)
        if angle_deg < 0:
            angle_deg += 360.0

        # 1 décimale
        return round(angle_deg, 1)

    def get_training_features(self) -> dict:
        """Features pour l'IA"""
        state = self.read_game_state()
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
            'inventory_vector': self.get_inventory_vector(),
            'inventory_items': self.read_inventory(),
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

    def print_state(self, include_inventory=True):
        """
        Affiche état complet
        """
        state = self.read_game_state()

        logger.info("" + "=" * 70)
        logger.info("ÉTAT COMPLET DU JEU")
        logger.info("=" * 70)

        # HP/SANTÉ
        logger.info("💚 HP/SANTÉ :")
        logger.info("-" * 70)

        hp_raw = state['player_hp_raw']
        hp_norm = state['player_hp']

        if hp_raw is not None:
            logger.info(f"HP actuel :")
            logger.info(f"Brut : {hp_raw}")
            logger.info(f"Normalisé : {hp_norm:.1f}/100")
        else:
            logger.warning(f"HP : Non disponible")

        # STAMINA
        logger.info("⚡ ÉNERGIE/STAMINA :")
        logger.info("-" * 70)

        stam_raw = state['player_stamina_raw']
        stam_norm = state['player_stamina']

        if stam_raw is not None:
            logger.info(f"Stamina actuelle : {stam_norm:.1f}/100")
        else:
            logger.warning(f"Stamina : Non disponible")

        # OXYGÈNE AVEC DEBUG
        logger.info("💨 OXYGÈNE (SOUS L'EAU) :")
        logger.info("-" * 70)

        oxygen = state.get('time_underwater')
        oxygen_valid = state.get('oxygen_valid', False)
        oxygen_error = state.get('oxygen_error')

        if oxygen_valid and oxygen is not None:
            logger.info(f"Niveau d'oxygène : {oxygen}/100")

            if oxygen < 25:
                logger.info(f"Oxygène bas ({oxygen} < 25)")
            if oxygen < 10:
                logger.info(f"Risque de noyade ({oxygen} < 10)")

            if state.get('oxygen_low_warning'):
                logger.info(f"flag activé")
            if state.get('oxygen_critical_warning'):
                logger.info(f"Critical flag activé")

        elif oxygen is not None and not oxygen_valid:
            logger.warning(f"Valeur invalide : {oxygen}")
            if oxygen_error:
                logger.error(f"Erreur : {oxygen_error}")
        else:
            logger.warning(f"Lecture de l'oxygene échouée")
            if oxygen_error:
                logger.error(f"Erreur : {oxygen_error}")

        # QUEST TIME
        logger.info("⏱️ TEMPS QUÊTE :")
        logger.info("-" * 70)
        quest_time = state.get('quest_time')
        if quest_time is not None:
            minutes = quest_time // 60
            seconds = quest_time % 60
            logger.info(f"Temps restant : {minutes}:{seconds:02d} ({quest_time}s)")

            if quest_time <= 60:
                logger.info(f"DANGER : Moins d'1 minute !")
        else:
            logger.warning(f"Temps : Non disponible")

        # MENU STATUS
        logger.info("🎮 MENU STATUS :")
        logger.info("-" * 70)

        # Lecture directe depuis mémoire
        in_menu = state.get('in_game_menu', False)
        if in_menu:
            logger.info(f"📋 MENU DU JEU OUVERT")
        else:
            logger.info(f"✅ Menu fermé")

        # ITEM SELECTED (avec nom de l'item)
        logger.info("🎯 ITEM SÉLECTIONNÉ :")
        logger.info("-" * 70)
        item_selected = state.get('item_selected')

        if item_selected is not None:
            if item_selected == 24:
                logger.info(f"Aucun item sélectionné")
            else:
                # Chercher l'item dans l'inventaire
                inventory = state.get('inventory_items', [])
                item_name = None
                item_id = None
                quantity = None

                # item_selected = index 0-23, slot = 1-24
                target_slot = item_selected + 1

                for item in inventory:
                    if item.get('slot') == target_slot:
                        item_name = item.get('name', f"Item ID {item.get('item_id')}")
                        item_id = item.get('item_id')
                        quantity = item.get('quantity')
                        break

                if item_name:
                    logger.info(f"Slot {target_slot} sélectionné : {item_name}")
                    if item_id and quantity:
                        logger.info(f"→ ID: {item_id}, Quantité: x{quantity}")
                else:
                    logger.info(f"Slot {target_slot} sélectionné (vide ou non trouvé)")
                    logger.info(f"Index: {item_selected}")
        else:
            logger.warning(f"Non disponible")

        # SHARPNESS
        logger.info("🔪 TRANCHANT :")
        logger.info("-" * 70)
        sharpness = state.get('sharpness')
        if sharpness is not None:
            logger.info(f"Tranchant : {sharpness}")

            if sharpness == -2:
                logger.info(f"REBOND détecté !")
        else:
            logger.warning(f"Tranchant : Non disponible")

        # POSITION
        logger.info("🗺️ POSITION & ORIENTATION :")
        logger.info("-" * 70)

        x = state['player_x']
        y = state['player_y']
        z = state['player_z']
        orientation = state['player_orientation']
        zone = state['current_zone']

        if x is not None and y is not None and z is not None:
            logger.info(f"Position 3D : ({x:.2f}, {y:.2f}, {z:.2f})")
        else:
            logger.warning(f"Position : Non disponible")

        if orientation is not None:
            logger.info(f"Orientation : {orientation:.2f}°")

        if zone is not None:
            logger.info(f"Zone actuelle : {zone}")

        # HP MONSTRES
        logger.info("🐉 HP MONSTRES :")
        logger.info("-" * 70)

        for i in range(1, 5):
            hp = state.get(f'smonster{i}_hp')
            if hp is not None and hp > 0:
                logger.info(f"Petit monstre {i} : {hp} HP")

        has_monsters = any(
            state.get(f'smonster{i}_hp') is not None and state.get(f'smonster{i}_hp') > 0
            for i in range(1, 5)
        )

        if not has_monsters:
            logger.info(f"Aucun monstre détecté dans la zone")

        # INVENTAIRE
        if include_inventory:
            logger.info("🎒 INVENTAIRE COMPLET :")
            logger.info("-" * 70)

            inventory = self.read_inventory()

            if inventory:
                logger.info(f"{len(inventory)} item(s) :")

                for item in inventory:
                    slot = item['slot']
                    item_id = item['item_id']
                    quantity = item['quantity']
                    name = item['name']

                    logger.info(f"Slot {slot} : {name:30s} x{quantity:2d} (ID: {item_id})")
            else:
                logger.warning(f"Inventaire vide ou non détecté")
                logger.warning(f"💡 Vérifie que tu es EN JEU (pas au village)")

        logger.info("" + "=" * 70)


# Test
if __name__ == "__main__":
    print("TEST MEMORY READER v1.0.5\n")

    if not DME_AVAILABLE:
        print("dolphin-memory-engine non installé")
        exit(1)

    try:
        reader = MemoryReader(force_quest_mode=True)

        print("\n" + "=" * 70)
        print("AFFICHAGE DE L'ÉTAT COMPLET DU JEU")
        print("=" * 70)

        reader.print_state(include_inventory=True)
        print("\nTest réussi!")

    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()