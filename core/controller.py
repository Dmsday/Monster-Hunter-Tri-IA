"""
Contrôleur avec inputs CIBLÉS sur la fenêtre Dolphin
Les touches sont envoyées uniquement à Dolphin, pas globalement
"""

# ============================================================================
# IMPORTS STANDARD PYTHON
# ============================================================================
import time                   # Délais entre actions

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from utils.module_logger import get_module_logger
logger = get_module_logger('controller')

# ============================================================================
# CLAVIER GLOBAL (fallback si Dolphin non trouvé)
# ============================================================================
try:
    import pynput.keyboard as kb
    PYNPUT_AVAILABLE = True
except ImportError:
    kb = None
    PYNPUT_AVAILABLE = False
    logger.error("⚠️ pynput non installé.")

# ============================================================================
# MANETTE VIRTUELLE (optionnel)
# ============================================================================
try:
    import vgamepad as vg
    VGAMEPAD_AVAILABLE = True
except ImportError:
    vg = None
    VGAMEPAD_AVAILABLE = False
    logger.error("⚠️ vgamepad non installé. Mode manette désactivé.")


class WiiController:
    """
    Contrôleur avec actions complètes :

    NOUVELLES ACTIONS :
    - use_object (Y/Triangle) : Utiliser l'objet sélectionné
    - select_item_left/right : Choisir un item (R1 + stick droite)
    """

    def __init__(
            self,
            debug=False,
            use_controller=False,
            instance_id=0,
    ):
        """
        Initialise le contrôleur

        Args:
            debug : Mode debug (affichage uniquement)
            use_controller : Si False, utilise clavier (pynput) - DÉFAUT
                             Si True, utilise manette virtuelle (vgamepad)
            instance_id : ID de l'instance (pour multi-instance)

        MODE CLAVIER (--keyboard):
            - Envoie les touches directement au système d'exploitation
            - Peut interférer avec d'autres applications ouvertes
            - Nécessite que Dolphin soit la fenêtre active
            - Mode manette (défaut) est recommandé
        """
        self.debug = debug
        self.use_controller = use_controller
        self.instance_id = instance_id
        # Logs pour débug multi-instance
        if instance_id > 0:
            logger.debug(f"🎮 Contrôleur instance #{instance_id}")

        self.gamepad = None
        self.keyboard = None
        self.is_connected = False
        self._gamepad_initialized = False

        if self.use_controller:
            # MODE MANETTE
            if not VGAMEPAD_AVAILABLE:
                logger.error("vgamepad non disponible - Fallback clavier")
                self.use_controller = False
                # Continue vers l'initialisation clavier
            else:
                try:
                    self.gamepad = vg.VX360Gamepad()
                    self._gamepad_initialized = True
                    self.is_connected = True
                    logger.info("Manette Xbox 360 virtuelle créée")
                    self.reset_all()
                    return  # Sortir après succès manette
                except Exception as initialize_vigembus_error:
                    logger.error(f"Erreur ViGEmBus: {initialize_vigembus_error}")
                    logger.error("   Fallback vers clavier")
                    self.use_controller = False
                    # Continue vers l'initialisation clavier

        # MODE CLAVIER (exécuté si use_controller=False OU si fallback)
        if not PYNPUT_AVAILABLE:
            logger.error("pynput non disponible - Mode simulation")
            self.debug = True
            self.is_connected = False
            return

        try:
            self.keyboard = kb.Controller()
            self.is_connected = True
            logger.info("Contrôleur clavier virtuel créé")
            self.reset_all()
        except Exception as initialize_pynput_error:
            logger.error(f"Erreur pynput: {initialize_pynput_error}")
            self.debug = True
            self.is_connected = False

    def reset_all(self):
        """
        Reset tous les inputs
        """
        if self.debug:
            return

        if self.use_controller and self.gamepad:
            # MODE MANETTE
            try:
                self.gamepad.reset()
                self.gamepad.update()
                time.sleep(0.05)
            except Exception as controller_reset_error:
                logger.error(f"⚠️ Erreur reset manette: {controller_reset_error}")

        elif not self.use_controller and self.keyboard:
            # MODE CLAVIER
            try:
                keys_to_release = [
                    'w', 'a', 's', 'd',  # Mouvement
                    kb.Key.up, kb.Key.down, kb.Key.left, kb.Key.right,  # Caméra
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'e',  # tous les boutons
                    kb.Key.shift_l, kb.Key.shift_r,  # Shift droit
                    kb.Key.ctrl_l, kb.Key.ctrl_r,  # Ctrl droit
                    kb.Key.alt_l, kb.Key.alt_r,  # Alt au cas où
                ]
                for key in keys_to_release:
                    try:
                        self.keyboard.release(key)
                    except (AttributeError, ValueError, OSError):
                        pass  # Continue avec les autres touches
                time.sleep(0.1)  # 0.1s pour laisser le temps au système
            except Exception as keyboard_reset_error:
                logger.error(f"⚠️ Erreur reset clavier: {keyboard_reset_error}")

    def cleanup(self):
        """
        Nettoyage final du contrôleur (à appeler avant fermeture)
        Force le relâchement de TOUTES les touches possibles
        """
        if self.debug:
            return

        logger.info("🧹 Nettoyage contrôleur...")

        if self.use_controller and self.gamepad and self._gamepad_initialized:
            # MODE MANETTE
            try:
                # Vérifier que vgamepad n'est pas déjà en cours de shutdown
                if vg is not None and hasattr(self.gamepad, 'reset'):
                    self.gamepad.reset()
                    self.gamepad.update()
                    time.sleep(0.1)

                # Nettoyer le handle interne de manière plus sûre
                try:
                    if hasattr(self.gamepad, '_XInputDevice__device'):
                        device = getattr(self.gamepad, '_XInputDevice__device', None)
                        if device is not None and hasattr(device, 'close'):
                            device.close()
                        self.gamepad._XInputDevice__device = None
                except (AttributeError, TypeError, RuntimeError):
                    pass

                self.gamepad = None  # Libérer la référence
                self._gamepad_initialized = False  # Marquer comme non initialisé

                logger.info("   Manette réinitialisée et libérée")
            except Exception as gamepad_cleanup_error:
                logger.error(f"   ⚠️ Erreur cleanup manette: {gamepad_cleanup_error}")

        elif not self.use_controller and self.keyboard:
            # MODE CLAVIER - Release exhaustif
            try:
                # Vérifier que pynput n'est pas déjà en cours de shutdown
                if kb is not None and self.keyboard is not None:
                    # Liste EXHAUSTIVE de toutes les touches utilisées
                    all_keys = [
                        # Mouvement
                        'w', 'a', 's', 'd',
                        # Caméra
                        kb.Key.up, kb.Key.down, kb.Key.left, kb.Key.right,
                        # Chiffres
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        # Lettres utilisées
                        'q', 'e',
                        # Modificateurs (gauche ET droite)
                        kb.Key.shift_l, kb.Key.shift_r,
                        kb.Key.ctrl_l, kb.Key.ctrl_r,
                        kb.Key.alt_l, kb.Key.alt_r,
                    ]

                    # Relâcher DEUX FOIS pour être sûr
                    for _ in range(2):
                        for key in all_keys:
                            try:
                                self.keyboard.release(key)
                            except (AttributeError, ValueError, OSError):
                                pass
                        time.sleep(0.05)

                logger.info("   Clavier réinitialisé")

            except Exception as keyboard_cleanup_error:
                logger.error(f"   ⚠️ Erreur cleanup clavier: {keyboard_cleanup_error}")

        self.is_connected = False

    def __del__(self):
        """
        Destructeur Python - nettoyage automatique lors de la destruction de l'objet
        Évite l'erreur 'NoneType' object is not callable de vgamepad
        """
        try:
            # Vérifier que Python n'est pas en cours de shutdown
            # Si les modules globaux sont None, c'est trop tard pour nettoyer
            if vg is None:
                return  # Sortir immédiatement si modules déjà nettoyés

            # Nettoyer la manette si elle existe encore
            if hasattr(self, '_gamepad_initialized') and self._gamepad_initialized:
                if hasattr(self, 'gamepad') and self.gamepad is not None:
                    try:
                        # Vérifier que les méthodes existent encore
                        if hasattr(self.gamepad, 'reset') and callable(self.gamepad.reset):
                            self.gamepad.reset()
                        if hasattr(self.gamepad, 'update') and callable(self.gamepad.update):
                            self.gamepad.update()

                        # Nettoyer le handle interne
                        if hasattr(self.gamepad, '_XInputDevice__device'):
                            try:
                                device = getattr(self.gamepad, '_XInputDevice__device', None)
                                if device is not None and hasattr(device, 'close'):
                                    device.close()
                            except (AttributeError, TypeError, RuntimeError):
                                pass
                            self.gamepad._XInputDevice__device = None
                    except (AttributeError, TypeError, RuntimeError):
                        pass  # Ignorer si déjà détruit

                    self.gamepad = None
                    self._gamepad_initialized = False
        except (AttributeError, TypeError):
            pass  # Ignorer si les attributs n'existent pas

    # ============================================================
    # ENVOI D'INPUTS (MANETTE UNIQUEMENT)
    # ============================================================

    def _send_input(self):
        """
        Envoie les inputs à la manette virtuelle

        Note : Utilisé uniquement en mode manette (vgamepad)
               Le mode clavier n'a pas besoin d'update()
        """
        if self.debug:
            return

        # Seulement pour la manette
        if self.use_controller and self.gamepad:
            try:
                self.gamepad.update()
                time.sleep(0.016)  # ~60 FPS
            except Exception as send_input_error:
                logger.error(f"⚠️ Erreur envoi input: {send_input_error}")

    # ============================================================
    # ACTIONS DE BASE
    # ============================================================

    def move(self, direction: str, duration: float = 0.5):
        """
        Déplace le joueur
        """
        # DEBUG
        if self.debug:
            logger.debug(f"MOVE {direction.upper()} ({duration}s)")
            time.sleep(duration)
            return

        if self.use_controller and self.gamepad:
            # MODE MANETTE - Joystick gauche
            try:
                x, y = 0.0, 0.0
                if direction == 'forward':
                    y = 1.0
                elif direction == 'backward':
                    y = -1.0
                elif direction == 'left':
                    x = -1.0
                elif direction == 'right':
                    x = 1.0

                self.gamepad.left_joystick_float(x_value_float=x, y_value_float=y)
                self.gamepad.update()
                time.sleep(duration)

                self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
                self.gamepad.update()
                time.sleep(0.05)

            except Exception as movement_with_controller_error:
                logger.error(f"⚠️ Erreur mouvement manette: {movement_with_controller_error}")

        elif not self.use_controller and self.keyboard:
            # MODE CLAVIER - WASD
            try:
                key_map = {
                    'forward': 'w',
                    'backward': 's',
                    'left': 'a',
                    'right': 'd'
                }

                if direction not in key_map:
                    return

                key = key_map[direction]
                self.keyboard.press(key)
                time.sleep(duration)
                self.keyboard.release(key)
                time.sleep(0.05)

            except Exception as movement_with_keyboard_error:
                logger.error(f"⚠️ Erreur mouvement clavier: {movement_with_keyboard_error}")

    def rotate_camera(self, direction: str, duration: float = 0.3):
        """
        Tourne la caméra
        """
        if self.debug:
            logger.debug(f"CAMERA {direction.upper()} ({duration}s)")
            time.sleep(duration)
            return

        if self.use_controller and self.gamepad:
            # MODE MANETTE - Joystick droit
            try:
                x, y = 0.0, 0.0
                if direction == 'up':
                    y = 1.0
                elif direction == 'down':
                    y = -1.0
                elif direction == 'left':
                    x = -1.0
                elif direction == 'right':
                    x = 1.0

                self.gamepad.right_joystick_float(x_value_float=x, y_value_float=y)
                self.gamepad.update()
                time.sleep(duration)

                self.gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
                self.gamepad.update()
                time.sleep(0.05)

            except Exception as use_camera_with_controller_error:
                logger.error(f"⚠️ Erreur caméra manette: {use_camera_with_controller_error}")

        elif not self.use_controller and self.keyboard:
            # MODE CLAVIER - Flèches directionnelles
            try:
                key_map = {
                    'up': kb.Key.up,
                    'down': kb.Key.down,
                    'left': kb.Key.left,
                    'right': kb.Key.right
                }

                if direction not in key_map:
                    return

                key = key_map[direction]
                self.keyboard.press(key)
                time.sleep(duration)
                self.keyboard.release(key)
                time.sleep(0.05)

            except Exception as use_camera_with_keyboard_error:
                logger.error(f"⚠️ Erreur caméra clavier: {use_camera_with_keyboard_error}")

    def press_button_simple(self, button_name: str, duration: float = 0.2):
        """
        Appuie sur un bouton

        CLAVIER (défaut):
        - draw_weapon : 3
        - attack1 : 4
        - attack2 : 0
        - dodge : 1
        - use_object : Q
        - start : E
        - z_target : LCTRL
        - c_button : LSHIFT

        MANETTE (--controller):
        - draw_weapon : R2
        - attack1 : X
        - attack2 : A
        - dodge : B
        - use_object : Y
        - start : Start
        - z_target : R1
        - c_button : L1
        """
        if self.debug:
            logger.debug(f"BUTTON {button_name.upper()} ({duration}s)")
            time.sleep(duration)
            return

        if self.use_controller and self.gamepad:
            # MODE MANETTE
            button_map = {
                'draw_weapon': None,  # R2 - géré séparément
                'attack1': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
                'attack2': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                'dodge': vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                'use_object': vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
                'start': vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
                'z_target': vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
                'c_button': vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
            }

            # Cas spécial : R2
            if button_name == 'draw_weapon':
                try:
                    self.gamepad.right_trigger(value=255)
                    self.gamepad.update()
                    time.sleep(duration)
                    self.gamepad.right_trigger(value=0)
                    self.gamepad.update()
                    time.sleep(0.05)
                except Exception as R2_error:
                    logger.error(f"⚠️ Erreur R2: {R2_error}")
                return

            if button_name not in button_map:
                logger.error(f"⚠️ Bouton manette inconnu: {button_name}")
                return

            try:
                button = button_map[button_name]
                self.gamepad.press_button(button=button)
                self.gamepad.update()
                time.sleep(duration)
                self.gamepad.release_button(button=button)
                self.gamepad.update()
                time.sleep(0.05)
            except Exception as controller_pressbutton_error:
                logger.error(f"⚠️ Erreur bouton manette: {controller_pressbutton_error}")

        elif not self.use_controller and self.keyboard:
            # MODE CLAVIER
            button_map = {
                'draw_weapon': '3',
                'attack1': '4',
                'attack2': '0',
                'dodge': '1',
                'use_object': 'q',
                'start': 'e',
                'z_target': kb.Key.ctrl_l,
                'c_button': kb.Key.shift_l,
            }

            if button_name not in button_map:
                logger.error(f"⚠️ Bouton clavier inconnu: {button_name}")
                return

            try:
                key = button_map[button_name]
                self.keyboard.press(key)
                time.sleep(duration)
                self.keyboard.release(key)
                time.sleep(0.05)
            except Exception as keyboard_pressbutton_error:
                logger.error(f"⚠️ Erreur bouton clavier: {keyboard_pressbutton_error}")

    # Sélection d'items
    def select_item(self, direction: str, count: int = 1,
                    hold_trigger_frames: int = 30,
                    stick_hold_frames: int = 36,
                    pause_between_frames: int = 18):
        """
        Sélectionne un item dans la barre d'action

        CLAVIER: LSHIFT + Flèches Gauche/Droite
        MANETTE: L1 + Stick Droit
        """
        if self.debug:
            logger.debug(f"SELECT ITEM {direction.upper()} x{count}")
            return

        if self.use_controller and self.gamepad:
            # MODE MANETTE - L1 + Stick droit
            try:
                # 1. MAINTENIR L1 initial
                for _ in range(hold_trigger_frames):
                    self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                    self.gamepad.update()
                    time.sleep(0.016)

                # 2. Déplacer stick EN MAINTENANT L1
                stick_x = -1.0 if direction == 'left' else 1.0

                for i in range(count):
                    for _ in range(stick_hold_frames):
                        self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                        self.gamepad.right_joystick_float(x_value_float=stick_x, y_value_float=0.0)
                        self.gamepad.update()
                        time.sleep(0.016)

                    for _ in range(pause_between_frames):
                        self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                        self.gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
                        self.gamepad.update()
                        time.sleep(0.016)

                # 3. Relâcher TOUT
                self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                self.gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
                self.gamepad.update()
                time.sleep(0.05)

            except Exception as select_item_with_controller_error:
                logger.error(f"⚠️ Erreur sélection item manette: {select_item_with_controller_error}")

        elif not self.use_controller and self.keyboard:
            # MODE CLAVIER - LSHIFT + Flèches
            try:
                arrow_key = kb.Key.left if direction == 'left' else kb.Key.right

                # 1. MAINTENIR LSHIFT
                self.keyboard.press(kb.Key.shift_l)
                time.sleep(hold_trigger_frames * 0.016)

                # 2. Appuyer sur flèche EN MAINTENANT LSHIFT
                for i in range(count):
                    self.keyboard.press(arrow_key)
                    time.sleep(stick_hold_frames * 0.016)
                    self.keyboard.release(arrow_key)
                    time.sleep(pause_between_frames * 0.016)

                # 3. Relâcher LSHIFT
                self.keyboard.release(kb.Key.shift_l)
                time.sleep(0.05)

            except Exception as select_item_with_keyboard_error:
                logger.error(f"⚠️ Erreur sélection item clavier: {select_item_with_keyboard_error}")


    # ============================================================
    # INTERFACE POUR L'IA - ACTIONS DISCRÈTES (ÉTENDUE)
    # ============================================================

    def execute_action(self, action_id: int, frames: int = 10):
        """
        Exécute une action pour l'IA

        ACTIONS COMPLÈTES (0-18) :
        0 : Rien
        1-4 : Déplacements (forward, backward, left, right)
        5-8 : Caméra (up, down, left, right)
        9 : Attaque 1 (X/Carré)
        10 : Esquive (B/Rond)
        11 : (Attaque) Dégainer / rengainer / (R2)
        12 : Attaque 2 (A/Croix)
        13 : Start (menu pause)
        14 : Z target (R1 - ciblage)
        15 : C button (L1)
        16 : Utiliser objet / rengainer (Y/Triangle)
        17 : Sélectionner item gauche QUICK (L1 + stick droite ← | 0.4s)
        18 : Sélectionner item droite QUICK (L1 + stick droite → | 0.4s)

        Args:
            action_id: ID de l'action (0-24)
            frames: Durée en frames
        """
        duration = frames * 0.016

        # Action 0 : Rien
        if action_id == 0:
            self.reset_all()
            time.sleep(duration)

        # Déplacements (1-4)
        elif action_id == 1:
            self.move('forward', duration)
        elif action_id == 2:
            self.move('backward', duration)
        elif action_id == 3:
            self.move('left', duration)
        elif action_id == 4:
            self.move('right', duration)

        # Caméra (5-8)
        elif action_id == 5:
            self.rotate_camera('up', duration)
        elif action_id == 6:
            self.rotate_camera('down', duration)
        elif action_id == 7:
            self.rotate_camera('left', duration)
        elif action_id == 8:
            self.rotate_camera('right', duration)

        # Boutons principaux (9-16)
        elif action_id == 9:
            self.press_button_simple('attack1', duration)
        elif action_id == 10:
            self.press_button_simple('dodge', duration)
        elif action_id == 11:
            self.press_button_simple('draw_weapon', duration)
        elif action_id == 12:
            self.press_button_simple('attack2', duration)
        elif action_id == 13:
            self.press_button_simple('start', duration)
        elif action_id == 14:
            self.press_button_simple('z_target', duration)
        elif action_id == 15:
            self.press_button_simple('c_button', duration)

        elif action_id == 16:
            # Utiliser objet (Y/Triangle)
            self.press_button_simple('use_object', duration)

        elif action_id == 17:
            # Sélectionner item à gauche QUICK (0.4s)
            self.select_item('left', count=1,
                             hold_trigger_frames=12,  # 0.2s
                             stick_hold_frames=4,  # 0.4s relatif / 6 ticks rate (vitesse emulation en mode illimite = 600%)
                             pause_between_frames=6)  # 0.1s

        elif action_id == 18:
            # Sélectionner item à droite QUICK (0.4s)
            self.select_item('right', count=1,
                             hold_trigger_frames=12,  # 0.2s
                             stick_hold_frames=4,  # 0.4s relatif
                             pause_between_frames=6)  # 0.1s

        else:
            if self.debug:
                logger.debug(f"Action {action_id} inconnue")
            time.sleep(duration)

        return frames


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("🎮 TEST CONTRÔLEUR AMÉLIORÉ\n")
    print("Mode par défaut : Manette virtuelle (vgamepad)")
    print("Pour tester le clavier : WiiController(use_controller=False)\n")

    controller = WiiController(debug=False, use_controller=True)  # Manette par défaut

    if not controller.is_connected:
        print("Contrôleur non connecté")
    else:
        print("⏸️ Pause 3s...")
        time.sleep(3)

        print("\n🧪 Test sélection d'items...")

        print("  1. Sélectionner item gauche")
        controller.select_item('left', count=1)
        time.sleep(1)

        print("  2. Sélectionner item droite x2")
        controller.select_item('right', count=2)
        time.sleep(1)

        print("  3. Utiliser objet (Y/Triangle)")
        controller.press_button_simple('use_object', 0.5)
        time.sleep(2)

        controller.reset_all()
        print("\nTest terminé!")