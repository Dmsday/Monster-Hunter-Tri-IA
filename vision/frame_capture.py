"""
Capture de frames depuis la fenêtre Dolphin
Priorité à la fenêtre avec "Monster Hunter" dans le titre
"""

# ============================================================================
# IMPORTS STANDARD PYTHON
# ============================================================================
import time                   # Gestion FPS et rate limiting
import traceback              # Affichage détaillé des erreurs
import ctypes                 # Pour appeler PrintWindow directement

# ============================================================================
# TRAITEMENT D'IMAGES
# ============================================================================
import numpy as np            # Arrays pour stocker les images
import cv2                    # OpenCV - Conversion couleurs (BGRA->RGB)

# ============================================================================
# WIN32 API (WINDOWS)
# ============================================================================
import win32gui               # Manipulation fenêtres Windows
import win32ui                # Interface utilisateur Win32
import win32con               # Constantes Win32 (SRCCOPY, etc.)
# Ces 3 modules permettent de :
# - Trouver la fenêtre Dolphin (win32gui.EnumWindows)
# - Capturer son contenu (BitBlt)
# - Convertir en image exploitable

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
# --- Logging ---
from utils.module_logger import get_module_logger
logger = get_module_logger('frame_capture')


class FrameCapture:
    def __init__(self, window_name="Dolphin", target_fps=30, instance_id=0, force_printwindow=False,
                 expected_window_title=None):
        """
        Args:
            window_name: Nom de la fenêtre Dolphin (deprecated, use expected_window_title)
            target_fps: FPS cible pour la capture
            instance_id: ID de l'instance (pour renommage multi-instance)
            force_printwindow: Forcer PrintWindow pour rtvision
            expected_window_title: Titre exact attendu (ex: "MHTri-0") - PRIORITY over instance_id
        """
        self.window_name = window_name
        self.instance_id = instance_id
        self.expected_window_title = expected_window_title
        self.hwnd = None
        self.target_fps = target_fps
        self.force_printwindow = force_printwindow
        self.frame_delay = 1.0 / target_fps
        self.last_capture_time = 0
        self._shutdown = False  # Flag to stop reconnection attempts during cleanup

        # Variables pour objets GDI persistants
        self._hwnd_dc = None
        self._mfc_dc = None
        self._save_dc = None
        self._save_bitmap = None
        self._bitmap_width = 0
        self._bitmap_height = 0
        self._gdi_initialized = False
        self._reinit_attempted = False

        self.find_window()

        # Compteur pour anti-spam
        self.identical_frame_count = 0  # Nombre de frames identiques consécutives
        self.last_warning_step = 0      # Dernier step où on a affiché un warning
        self.warning_interval = 600      # Afficher warning tous les 600 frames (10s)

        # Priorité de recherche si instance_id > 0
        if instance_id > 0:
            # Chercher d'abord MHTri-{N}
            self.window_patterns = [
                f"MHTri-{instance_id}",  # Priorité 1
                f"Monster Hunter Tri - {instance_id}",  # Priorité 2
                "Monster Hunter Tri",  # Fallback
                "Dolphin"  # Dernier recours
            ]
        else:
            # Instance 0 : chercher MHTri ou Monster Hunter Tri classique
            self.window_patterns = [
                "MHTri",
                "Monster Hunter Tri",
                "Dolphin"
            ]

    def _init_gdi_objects(self, width, height):
        """
         Initialiser objets GDI UNE SEULE FOIS
         """
        # Si déjà initialisé avec bonnes dimensions, skip
        if (self._gdi_initialized and
                self._bitmap_width == width and
                self._bitmap_height == height):
            return

        # Nettoyer anciens objets si redimensionnement
        if self._gdi_initialized:
            self._cleanup_gdi()

        try:
            # Créer objets GDI (une seule fois)
            self._hwnd_dc = win32gui.GetWindowDC(self.hwnd)
            self._mfc_dc = win32ui.CreateDCFromHandle(self._hwnd_dc)
            self._save_dc = self._mfc_dc.CreateCompatibleDC()

            self._save_bitmap = win32ui.CreateBitmap()
            self._save_bitmap.CreateCompatibleBitmap(self._mfc_dc, width, height)
            self._save_dc.SelectObject(self._save_bitmap)

            self._bitmap_width = width
            self._bitmap_height = height
            self._gdi_initialized = True

        except Exception as init_GDI_error:
            logger.error(f"Erreur init GDI: {init_GDI_error}")
            self._gdi_initialized = False
            raise

    def find_window(self):
        """
        PRIORITY MULTI-INSTANCE:
        0. EXACT MATCH if expected_window_title provided (NEW - HIGHEST PRIORITY)
        1. MHTri-1, MHTri-2, ..., MHTri-100 (numbered)
        2. Monster Hunter Tri (full title)
        3. MHTri (without number)
        4. Dolphin with game loaded
        5. Generic Dolphin

        EXCLUSIONS :
            - PyCharm, Visual Studio, VSCode
            - Navigateurs web (Chrome, Firefox, Edge)
            - Explorateur Windows
        """

        def callback(h, wins):
            if win32gui.IsWindowVisible(h):
                game_title = win32gui.GetWindowText(h)
                title_lower = game_title.lower()

                # Skip empty windows
                if not game_title:
                    return True

                # PRIORITY 0 : EXACT MATCH (if expected_window_title provided)
                if self.expected_window_title:
                    if game_title == self.expected_window_title:
                        # Exact match found - HIGHEST PRIORITY
                        wins.append((h, game_title, 100000))  # Priority >> 10000
                        return True
                    else:
                        # Not our window, skip
                        return True

                # FILTRE EXCLUSION : Ignorer fenêtres IDEs et navigateurs
                excluded_patterns = [
                    'pycharm', 'visual studio', 'vscode', 'vs code',
                    'chrome', 'firefox', 'edge', 'brave', 'opera',
                    'explorer', 'explorateur', 'notepad', 'bloc-notes',
                    'cmd', 'powershell', 'terminal', 'console',
                ]

                if any(excluded in title_lower for excluded in excluded_patterns):
                    return True  # Ignorer cette fenêtre

                # Détection multi-critères
                has_dolphin = "dolphin" in title_lower
                has_mh_full = "monster hunter" in title_lower
                has_mhtri = title_lower.startswith("mhtri")

                # Détection MHTri-X (X = 1-100)
                is_mhtri_numbered = False
                mhtri_number = 1000  # Par défaut très élevé

                if has_mhtri and "-" in game_title:
                    try:
                        # Extraire le numéro après "MHTri-"
                        parts = game_title.split("-")
                        if len(parts) >= 2:
                            # Prendre premier mot après le tiret
                            num_str = parts[1].split()[0] if parts[1] else ""
                            window_num = int(num_str)
                            if 1 <= window_num <= 100:
                                is_mhtri_numbered = True
                                mhtri_number = window_num
                    except (ValueError, IndexError):
                        pass

                # Calculer score de priorité
                window_priority = 0

                if is_mhtri_numbered:
                    # Si on cherche MHTri-2 et qu'on trouve MHTri-2 = priorité MAX
                    if mhtri_number == self.instance_id:
                        window_priority = 10000  # Priorité absolue
                    else:
                        # Autre MHTri numéroté = basse priorité (éviter collision)
                        window_priority = 100 - mhtri_number
                elif has_mh_full:
                    window_priority = 900  # "Monster Hunter Tri"
                elif has_mhtri:
                    window_priority = 800  # "MHTri" simple
                elif has_dolphin:
                    window_priority = 100  # Dolphin générique

                if window_priority > 0:
                    wins.append((h, game_title, window_priority))

            return True

        windows = []
        win32gui.EnumWindows(callback, windows)

        if not windows:
            raise ValueError(
                f"Fenêtre de jeu non trouvée!\n"
                f"💡 Vérifications:\n"
                f"   1. Dolphin est-il lancé?\n"
                f"   2. Monster Hunter Tri est-il chargé?\n"
                f"   3. La fenêtre est-elle visible (pas minimisée)?\n"
            )

        # Sort by priority (descending)
        windows.sort(key=lambda x: x[2], reverse=True)

        # Take highest priority window
        self.hwnd, found_title, priority = windows[0]

        # Logging according to window found
        if priority >= 100000:
            logger.info(f"Exact window match found: '{found_title}'")
            logger.info(f"  Expected: '{self.expected_window_title}'")
            logger.info(f"  HWND: {self.hwnd}")
            logger.info(f"  Priorité: {priority}")
        elif priority >= 1000:
            num = 1000 - priority
            logger.warning(f"Instance {self.instance_id} : Fenêtre MHTri-{num} trouvée (ATTENDU: MHTri-{self.instance_id})")
            logger.warning(f"  Titre: '{found_title}'")
            logger.warning(f"  RISQUE DE COLLISION - Vérifie renommage Dolphin")
        elif priority >= 900:
            logger.info(f"Fenêtre Monster Hunter Tri trouvée !")
            logger.info(f" Titre: '{found_title}'")
        elif priority >= 800:
            logger.info(f"Fenêtre MHTri trouvée !")
            logger.info(f" Titre: '{found_title}'")
        else:
            logger.info(f"Fenêtre Dolphin générique trouvée")
            logger.info(f" Titre: '{found_title}'")
            logger.info(f"💡 Charge Monster Hunter Tri pour de meilleurs résultats")

        logger.info(f"HWND: {self.hwnd}")

        # Afficher autres fenêtres trouvées (top 3)
        if len(windows) > 1:
            logger.info(f"\n{len(windows)} fenêtres détectées (Top 3):")
            for k, (hwnd, game_title, prio) in enumerate(windows[:3], 1):
                marker = "👉" if k == 1 else "  "
                logger.info(f"{marker} {k}. {game_title} (priorité: {prio})")

    def _get_black_frame(self):
        """
        Retourne une frame noire en cas d'erreur
        """
        # Utiliser les dimensions du bitmap si disponible, sinon 640x480 par défaut
        height = self._bitmap_height if self._bitmap_height > 0 else 480
        width = self._bitmap_width if self._bitmap_width > 0 else 640
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _reinit_gdi(self):
        """
        Réinitialise les objets GDI en cas d'erreur
        """
        try:
            logger.warning("Réinitialisation GDI...")

            # Nettoyer anciens objets
            self._cleanup_gdi()

            # Recréer avec les dimensions actuelles
            if self._bitmap_width > 0 and self._bitmap_height > 0:
                self._init_gdi_objects(self._bitmap_width, self._bitmap_height)
                logger.warning("GDI réinitialisé")
                return True
            else:
                logger.warning("Dimensions invalides pour réinit")
                return False

        except Exception as reinit_error:
            logger.error(f"Échec réinitialisation: {reinit_error}")
            return False

    def capture_frame(
            self,
            crop_region: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """
        Capture a frame from the Dolphin window with robust error handling

        Args:
            crop_region: Tuple (x, y, width, height) for a specific area

        Returns:
            numpy.ndarray: RGB image (H, W, 3)
        """
        # Check if shutting down --> stop trying to reconnect
        if self._shutdown:
            logger.debug("Frame capture shutdown, returning black frame")
            return self._get_black_frame()

        # Check if the window exists
        if self.hwnd is None:
            if self._shutdown:  # Double-check before attempting reconnection
                return self._get_black_frame()
            try:
                self.find_window()
            except Exception as find_error:
                logger.error(f"Unable to find Dolphin: {find_error}")
                return self._get_black_frame()

        # Check if the window still exists
        if not win32gui.IsWindow(self.hwnd):
            if self._shutdown:  # Don't try to reconnect if shutting down
                logger.debug("Window closed during shutdown, skipping reconnection")
                return self._get_black_frame()

            logger.warning("Dolphin window closed - searching for a new window...")
            try:
                self.find_window()
            except (ValueError, RuntimeError, OSError):
                return self._get_black_frame()

        # Dimensions fenêtre
        try:
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            width = right - left
            height = bottom - top
        except Exception as rect_error:
            logger.error(f"Erreur lecture dimensions: {rect_error}")
            return self._get_black_frame()

        if width <= 0 or height <= 0:
            logger.warning(f"Dimensions invalides: {width}x{height}")
            return self._get_black_frame()

        # Initialiser objets GDI (une seule fois)
        try:
            self._init_gdi_objects(width, height)
        except Exception as init_error:
            logger.error(f"Erreur init GDI: {init_error}")
            return self._get_black_frame()

        # Vérifier que les objets GDI sont bien initialisés
        if not self._gdi_initialized or self._save_dc is None or self._save_bitmap is None:
            logger.error("Objets GDI non initialisés")
            return self._get_black_frame()

        # Choisir methode selon mode
        capture_success = False

        # Si force_printwindow active, sauter BitBlt
        if not self.force_printwindow:
            try:
                self._save_dc.BitBlt(
                    (0, 0), (width, height),
                    self._mfc_dc, (0, 0),
                    win32con.SRCCOPY
                )
                capture_success = True

            except Exception as bitblt_error:
                logger.warning(f"BitBlt echec: {bitblt_error}")
                capture_success = False
        else:
            # Mode force_printwindow : sauter BitBlt directement
            logger.debug("Force PrintWindow actif, BitBlt ignore")

        # RECUPERATION DES BITS DU BITMAP (COMMUN AUX DEUX METHODES)
        try:
            # Verifier que l'objet existe avant GetBitmapBits
            if self._save_bitmap is None:
                logger.warning("save_bitmap est None - tentative reinit")

                if not self._reinit_attempted:
                    self._reinit_attempted = True
                    if self._reinit_gdi():
                        logger.info("Reinit reussie, nouvelle tentative...")
                        return self.capture_frame(crop_region)

                return self._get_black_frame()

            # Recuperer les bits
            bmpstr = self._save_bitmap.GetBitmapBits(True)

        except AttributeError as bitmap_attr_error:
            # Erreur specifique : objet GDI devenu None
            logger.warning(f"Objet GDI invalide: {bitmap_attr_error}")

            if not self._reinit_attempted:
                self._reinit_attempted = True
                if self._reinit_gdi():
                    logger.info("Reinit apres AttributeError...")
                    return self.capture_frame(crop_region)

            return self._get_black_frame()

        except Exception as get_bits_error:
            logger.warning(f"GetBitmapBits echec: {get_bits_error}")
            return self._get_black_frame()

        # Verifier que les bits ne sont pas None
        if bmpstr is None:
            logger.warning("GetBitmapBits retourne None")
            return self._get_black_frame()

        # SI BITBLT A ECHOUE, FALLBACK PRINTWINDOW
        if not capture_success:
            # Log uniquement si c'est un echec inattendu (pas en mode force)
            if not self.force_printwindow:
                logger.warning("Tentative PrintWindow apres echec BitBlt...")
            else:
                # Mode force : comportement normal, pas de warning
                logger.debug("Utilisation PrintWindow (mode force_printwindow)")

            hdc = self._save_dc.GetSafeHdc()

            if hdc == 0 or hdc is None:
                logger.warning("HDC invalide pour PrintWindow")
                if not self._reinit_attempted:
                    self._reinit_attempted = True
                    if self._reinit_gdi():
                        logger.info("Nouvelle tentative apres reinit...")
                        return self.capture_frame(crop_region)
                return self._get_black_frame()

            print_window = getattr(ctypes.windll.user32, 'PrintWindow')
            print_window.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
            print_window.restype = ctypes.c_bool
            result = print_window(
                int(self.hwnd),
                int(hdc),
                2  # PW_RENDERFULLCONTENT
            )

            if not result:
                # Log selon contexte
                if self.force_printwindow:
                    logger.debug("PrintWindow retourne False (fenetre minimisee?)")
                else:
                    logger.warning("PrintWindow aussi a echoue")

            # Re-recuperer les bits apres PrintWindow
            try:
                if self._save_bitmap is None:
                    logger.warning("save_bitmap None apres PrintWindow")
                    return self._get_black_frame()

                bmpstr = self._save_bitmap.GetBitmapBits(True)

                if bmpstr is None:
                    logger.warning("GetBitmapBits None apres PrintWindow")
                    return self._get_black_frame()

            except Exception as printwindow_bits_error:
                logger.warning(f"GetBitmapBits apres PrintWindow: {printwindow_bits_error}")
                return self._get_black_frame()

        # CONVERSION EN IMAGE (COMMUN)
        try:
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img = img.reshape((height, width, 4))

            # Conversion couleur optimisee
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

            # Crop si demande
            if crop_region:
                x, y, w, h = crop_region
                if x >= 0 and y >= 0 and x + w <= img.shape[1] and y + h <= img.shape[0]:
                    img = img[y:y + h, x:x + w]
                else:
                    logger.warning(f"Crop invalide: {crop_region} pour image {img.shape}")

            # Reset du flag de reinit si succes
            self._reinit_attempted = False

            return img

        except Exception as conversion_error:
            logger.error(f"Erreur conversion image: {conversion_error}")

            # Tentative de reinitialisation (1 fois max par cycle)
            if not self._reinit_attempted:
                self._reinit_attempted = True
                if self._reinit_gdi():
                    logger.info("Nouvelle tentative apres reinit...")
                    return self.capture_frame(crop_region)

            return self._get_black_frame()

    def _cleanup_gdi(self):
        """
        Nettoyer objets GDI dans le bon ordre
        """
        try:
            # ÉTAPE 1 : Désélectionner le bitmap du DC (CRUCIAL)
            # Vérifier que les objets sont VRAIMENT valides
            if (self._save_dc is not None and
                    self._save_bitmap is not None and
                    hasattr(self._save_dc, 'SelectObject')):
                try:
                    # Créer un bitmap vide pour remplacer celui sélectionné
                    empty_bitmap = win32ui.CreateBitmap()
                    empty_bitmap.CreateCompatibleBitmap(self._mfc_dc, 1, 1)
                    self._save_dc.SelectObject(empty_bitmap)
                    try:
                        empty_bitmap.DeleteObject()  # type: ignore[attr-defined]
                    except (AttributeError, RuntimeError, OSError, ValueError):
                        pass  # Ignorer erreur DeleteObject
                except (AttributeError, RuntimeError, OSError, ValueError):
                    pass  # Si ça échoue, on continue quand même

            # ÉTAPE 2 : Supprimer les objets GDI dans l'ordre inverse de création
            if self._save_bitmap is not None:
                try:
                    win32gui.DeleteObject(self._save_bitmap.GetHandle())
                except (AttributeError, RuntimeError, OSError, ValueError):
                    pass

            if self._save_dc is not None:
                try:
                    self._save_dc.DeleteDC()
                except (AttributeError, RuntimeError, OSError, ValueError):
                    pass

            if self._mfc_dc is not None:
                try:
                    self._mfc_dc.DeleteDC()
                except (AttributeError, RuntimeError, OSError, ValueError):
                    pass

            if self._hwnd_dc is not None:
                try:
                    win32gui.ReleaseDC(self.hwnd, self._hwnd_dc)
                except (AttributeError, RuntimeError, OSError, ValueError):
                    pass

        except Exception as cleanup_GDI_error:
            # Erreur globale (ne devrait pas arriver avec les try individuels)
            logger.debug(f"Erreur lors du nettoyage GDI: {cleanup_GDI_error}")
        finally:
            # Toujours réinitialiser les variables
            self._hwnd_dc = None
            self._mfc_dc = None
            self._save_dc = None
            self._save_bitmap = None
            self._gdi_initialized = False

    def close(self):
        """
        Fermeture propre
        """
        self._cleanup_gdi()

    def shutdown(self):
        """
        Signal frame capture to stop reconnection attempts.
        Called during training cleanup to prevent spam during Dolphin shutdown.
        """
        self._shutdown = True
        logger.info("Frame capture shutdown - stopping reconnection attempts")

    def capture_game_area(self):
        """
        Capture uniquement la zone de jeu (exclut bordures Dolphin)
        À ajuster selon ta config
        """
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bottom - top

        # Zone de jeu approximative (à ajuster)
        game_crop_region = (
            10,  # x offset
            40,  # y offset (barre titre + menu)
            width - 20,  # largeur
            height - 50  # hauteur
        )

        # noinspection PyTypeChecker
        return self.capture_frame(crop_region=game_crop_region)


# Test
if __name__ == "__main__":
    print("🎥 Test de capture Dolphin (version corrigée)\n")

    try:
        # Créer 'capturer'
        print("=" * 70)
        print("🔍 RECHERCHE DES FENÊTRES MONSTER HUNTER TRI")
        print("=" * 70)

        capturer = FrameCapture()

        print("")
        print("=" * 70)
        print("📊 RÉSUMÉ DE LA DÉTECTION")
        print("=" * 70)


        # Énumérer toutes les fenêtres pour le rapport
        def find_all_mh_windows():
            """Trouve toutes les fenêtres MH pour le rapport"""
            found = []

            def callback(h, wins):
                if win32gui.IsWindowVisible(h):
                    title = win32gui.GetWindowText(h)
                    title_lower = title.lower()

                    if not title:
                        return True

                    # Détection des fenêtres MH
                    if any(keyword in title_lower for keyword in ["mhtri", "monster hunter", "dolphin"]):
                        # Calculer priorité
                        priority = 0
                        window_type = "Autre"

                        if "mhtri-" in title_lower:
                            try:
                                num = int(title.split("-")[1].split()[0])
                                if 1 <= num <= 100:
                                    priority = 1000 - num
                                    window_type = f"MHTri-{num}"
                            except (ValueError, IndexError):
                                pass
                        elif "monster hunter" in title_lower:
                            priority = 900
                            window_type = "Monster Hunter Tri"
                        elif title_lower.startswith("mhtri"):
                            priority = 800
                            window_type = "MHTri"
                        elif "dolphin" in title_lower:
                            priority = 100
                            window_type = "Dolphin générique"

                        wins.append({
                            'hwnd': h,
                            'title': title,
                            'priority': priority,
                            'type': window_type,
                            'selected': h == capturer.hwnd
                        })

                return True

            win32gui.EnumWindows(callback, found)
            return sorted(found, key=lambda x: x['priority'], reverse=True)


        all_windows = find_all_mh_windows()

        # Compter par type
        mh_windows = [w for w in all_windows if w['priority'] >= 800]
        dolphin_generic = [w for w in all_windows if w['priority'] == 100]

        print(f"Fenêtres Monster Hunter trouvées : {len(mh_windows)}")
        print(f"Fenêtres Dolphin génériques : {len(dolphin_generic)}")
        print(f"Total : {len(all_windows)}")
        print("")

        # Afficher toutes les fenêtres avec indicateur de sélection
        if all_windows:
            print("Liste complète des fenêtres détectées :")
            print("-" * 70)

            for i, window in enumerate(all_windows, 1):
                marker = "✅ SÉLECTIONNÉE" if window['selected'] else "  "
                print(f"{marker} #{i} [{window['type']}]")
                print(f"      Titre : {window['title']}")
                print(f"      HWND  : {window['hwnd']}")
                print(f"      Priorité : {window['priority']}")
                print("")

        print("=" * 70)
        print("🎯 FENÊTRE SÉLECTIONNÉE POUR LA CAPTURE")
        print("=" * 70)

        selected = next((w for w in all_windows if w['selected']), None)
        if selected:
            print(f"Type : {selected['type']}")
            print(f"Titre : {selected['title']}")
            print(f"HWND : {selected['hwnd']}")
            print(f"Priorité : {selected['priority']}")

        print("")
        print("=" * 70)
        print("📸 TEST DE CAPTURE (5 FRAMES)")
        print("=" * 70)

        # Capture 5 frames
        for frame_idx in range(5):
            captured_frame = capturer.capture_frame()
            print(f"Frame {frame_idx + 1}/5 : {captured_frame.shape} - {captured_frame.dtype}")

            # Sauvegarder la première frame
            if frame_idx == 0:
                cv2.imwrite("test_frame.png", cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR))
                print("   💾 Frame sauvegardée : test_frame.png")

            time.sleep(0.5)

        print("")
        print("=" * 70)
        print("✅ TEST RÉUSSI")
        print("=" * 70)
        print("Vérifications effectuées :")
        print(f"   ✓ {len(mh_windows)} fenêtre(s) Monster Hunter détectée(s)")
        print(f"   ✓ Fenêtre sélectionnée : {selected['type'] if selected else 'Aucune'}")
        print(f"   ✓ Captures effectuées : 5/5")
        print(f"   ✓ Frame test sauvegardée : test_frame.png")

    except Exception as frame_capture_individual_test_error:
        print("")
        print("=" * 70)
        print("❌ ERREUR DURANT LE TEST")
        print("=" * 70)
        print(f"Erreur : {frame_capture_individual_test_error}")
        print("")
        traceback.print_exc()