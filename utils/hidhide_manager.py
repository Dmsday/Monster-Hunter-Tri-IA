"""
Gestionnaire HidHide pour isolation des manettes virtuelles
"""

import subprocess
import os
import logging
from typing import Optional
import vgamepad as vg

logger = logging.getLogger('hidhide_manager')


def is_admin() -> bool:
    """
    Verify if script runs with admin rights

    Returns:
        True if administrator, False otherwise
    """
    try:
        import ctypes
        import ctypes.wintypes

        # Access Shell32.dll and check admin status
        # IsUserAnAdmin is a Windows API function that returns BOOL (1 if admin, 0 otherwise)
        # PyCharm can't infer this dynamically loaded function, but it exists on Windows
        # noinspection PyUnresolvedReferences
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except (AttributeError, OSError) as admin_check_error:
        # AttributeError: windll.shell32 not available (non-Windows)
        # OSError: DLL access failed
        logger.debug(f"Admin check failed: {admin_check_error}")
        return False


class HidHideManager:
    """
    Gère la configuration HidHide pour isoler les manettes virtuelles
    par instance Dolphin
    """

    def __init__(self, hidhide_cli_path: Optional[
        str] = "C:/Program Files/Nefarius Software Solutions/HidHide/x64/HidHideCLI.exe"):
        """
        Args:
            hidhide_cli_path: Chemin vers HidHideCLI.exe
                             Si None, cherche dans PATH
        """
        if not is_admin():
            logger.warning("Script non lancé en administrateur")
            logger.warning("HidHide nécessite des droits élevés")
            logger.warning("💡 Relance PyCharm/terminal en tant qu'admin")

        self.cli_path = hidhide_cli_path or "HidHideCLI.exe"
        self.configured_devices = []

        logger.info("🔍 Vérification HidHide...")
        logger.debug(f"   Chemin CLI : {self.cli_path}")

        if not self._check_hidhide_installed():
            logger.error("❌ HidHide n'est pas installé ou inaccessible!")
            logger.error(f"   Chemin vérifié : {self.cli_path}")
            logger.error("")
            logger.error("💡 Solutions:")
            logger.error("   1. Télécharge HidHide: https://github.com/ViGEm/HidHide/releases")
            logger.error("   2. Installe en tant qu'administrateur")
            logger.error("   3. Vérifie que le chemin est correct:")
            logger.error(f"      {self.cli_path}")
            logger.error("   4. Alternative : passe le chemin custom:")
            logger.error("      HidHideManager(hidhide_cli_path='C:/ton/chemin/HidHideCLI.exe')")

            raise RuntimeError("HidHide requis pour multi-instance")

        logger.info("HidHide détecté et fonctionnel")

        # Vérifier version (optionnel)
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            if result.stdout:
                version_line = result.stdout.strip().split('\n')[0]
                logger.info(f"Version: {version_line}")

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as hidhide_version_error:
            # Version check is non-critical, continue without it
            logger.debug(f"Could not retrieve HidHide version: {hidhide_version_error}")

    def _check_hidhide_installed(self) -> bool:
        """Vérifie si HidHide est installé"""

        # Méthode 1 : Vérifier que le CLI existe
        if not os.path.exists(self.cli_path):
            logger.debug(f"HidHideCLI.exe non trouvé : {self.cli_path}")
            return False

        # Méthode 2 : Tester l'exécution
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=2,  # Timeout avant d'annoncé echec
                creationflags=subprocess.CREATE_NO_WINDOW  # Pas de fenêtre console
            )

            # Accepter codes de retour 0 OU 1 (certaines versions retournent 1)
            if result.returncode in [0, 1]:
                logger.debug("HidHide CLI répond correctement")
                return True

            # Si code différent, afficher pour debug
            logger.debug(f"HidHide returncode inattendu : {result.returncode}")
            logger.debug(f"Stdout: {result.stdout}")
            logger.debug(f"Stderr: {result.stderr}")

            # Considérer comme installé si on a eu une réponse
            return True

        except FileNotFoundError:
            logger.debug(f"Fichier non trouvé : {self.cli_path}")
            return False

        except subprocess.TimeoutExpired:
            logger.warning("HidHide CLI timeout (3s dépassé)")
            # Considérer comme installé si le fichier existe
            return os.path.exists(self.cli_path)

        except Exception as check_error:
            logger.debug(f"Erreur vérification HidHide : {check_error}")
            # Fallback : vérifier juste que le fichier existe
            return os.path.exists(self.cli_path)

    def configure_device_for_exe(
            self,
            device_instance_path: str,
            allowed_exe: str
    ):
        r"""
        Configure HidHide pour masquer un device de tous les exe
        SAUF celui spécifié

        Args:
            device_instance_path: Chemin d'instance du device (ex: HID\VID_045E&PID_028E...)
            allowed_exe: Chemin complet vers l'exe autorisé (ex: C:\Dolphin\dolphin1.exe)
        """
        try:
            # Étape 1: Cacher le device globalement
            subprocess.run(
                [self.cli_path, "--dev-hide", device_instance_path],
                check=True,
                capture_output=True
            )

            # Étape 2: Whitelister l'exe spécifique
            subprocess.run(
                [self.cli_path, "--app-reg", allowed_exe],
                check=True,
                capture_output=True
            )

            # Étape 3: Permettre à cet exe de voir ce device
            subprocess.run(
                [
                    self.cli_path,
                    "--app-allow", allowed_exe,
                    "--dev-gaming", device_instance_path
                ],
                check=True,
                capture_output=True
            )

            self.configured_devices.append({
                'device': device_instance_path,
                'exe': allowed_exe
            })

            logger.info(f"Device configuré pour {os.path.basename(allowed_exe)}")

        except subprocess.CalledProcessError as configure_device_error:
            logger.error(f"Erreur configuration HidHide: {configure_device_error}")

            # Display stdout/stderr if available
            if configure_device_error.stdout:
                try:
                    logger.error(f"   Stdout: {configure_device_error.stdout.decode('utf-8', errors='ignore')}")
                except (AttributeError, UnicodeDecodeError) as decode_error:
                    # AttributeError: stdout is not bytes
                    # UnicodeDecodeError: invalid encoding
                    logger.debug(f"Could not decode stdout: {decode_error}")

            if configure_device_error.stderr:
                try:
                    logger.error(f"   Stderr: {configure_device_error.stderr.decode('utf-8', errors='ignore')}")
                except (AttributeError, UnicodeDecodeError) as decode_error:
                    logger.debug(f"Could not decode stderr: {decode_error}")

            raise

    def get_vgamepad_device_path(self, gamepad_index: int = 0) -> Optional[str]:
        r"""
        Récupère le device instance path d'un vGamepad

        Cette méthode doit être appelée APRÈS création des manettes

        Format : HID\VID_045E&PID_028E&IG_XX\...

        Args:
            gamepad_index: Index du gamepad (0 pour le premier créé)

        Returns:
            Device instance path (format HID\VID_...) ou None
        """

        # Retry logic : 3 tentatives avec timeout croissant
        timeouts = [5, 10, 20]  # 5s, puis 10s, puis 20s

        for attempt, timeout in enumerate(timeouts, 1):
            try:
                logger.debug(
                    f"Recherche device ViGEm #{gamepad_index} (tentative {attempt}/{len(timeouts)}, timeout={timeout}s)...")

                result = subprocess.run(
                    [self.cli_path, "--dev-list"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )

                vigem_devices = []

                # Parser la sortie ligne par ligne
                for line in result.stdout.split('\n'):
                    line_clean = line.strip()

                    # Détecter devices Xbox 360 (ViGEm)
                    if 'HID\\VID_045E&PID_028E' in line_clean:
                        if line_clean.startswith('HID\\'):
                            vigem_devices.append(line_clean)
                            logger.debug(f"   Device trouvé : {line_clean[:50]}...")

                # Trier pour ordre cohérent
                vigem_devices.sort()

                logger.info(f"Total devices ViGEm détectés : {len(vigem_devices)}")

                if vigem_devices:
                    logger.debug("Devices ViGEm trouvés :")
                    for j, dev in enumerate(vigem_devices):
                        logger.debug(f"   [{j}] {dev[:80]}...")

                if gamepad_index < len(vigem_devices):
                    selected = vigem_devices[gamepad_index]
                    logger.info(f"Device sélectionné pour gamepad #{gamepad_index}")
                    logger.debug(f"Path: {selected}")
                    return selected
                else:
                    logger.warning(f"Gamepad #{gamepad_index} non trouvé")
                    logger.warning(f"Demandé : #{gamepad_index}, Disponibles : {len(vigem_devices)}")

                    if len(vigem_devices) == 0:
                        logger.error("No ViGEm device detected!")
                        logger.error("Possible causes:")
                        logger.error("   1. Virtual controllers not created yet")
                        logger.error("   2. ViGEmBus not installed")
                        logger.error("   3. ViGEm drivers not loaded")

                    return None

            except subprocess.CalledProcessError as list_error:
                logger.error(f"HidHide device listing error: {list_error}")
                if list_error.stderr:
                    logger.error(f"   Stderr: {list_error.stderr}")

                # If it's the last attempt, give up
                if attempt == len(timeouts):
                    return None

                # Otherwise, retry
                logger.warning(f"Attempt {attempt} failed, retrying in 2s...")
                time.sleep(2)
                continue

            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout tentative {attempt} ({timeout}s dépassé)")

                # If it's the last attempt, give up
                if attempt == len(timeouts):
                    logger.error("Failed after 3 attempts")
                    logger.error("💡 Possible solutions:")
                    logger.error("   1. HidHide CLI too slow → Restart your PC")
                    logger.error("   2. Too many devices → Unplug unnecessary USB devices")
                    logger.error("   3. Corrupted cache → Reinstall HidHide")
                    return None

                # Otherwise, retry with a longer timeout
                logger.warning(f"Retrying with a longer timeout ({timeouts[attempt]}s)...")
                time.sleep(1)
                continue

            except Exception as get_device_error:
                logger.error(f"Unexpected error: {get_device_error}")

                if attempt == len(timeouts):
                    import traceback
                    traceback.print_exc()
                    return None

                time.sleep(2)
                continue

        return None

    def reset_all(self):
        """Réinitialise toutes les configurations HidHide"""
        try:
            subprocess.run(
                [self.cli_path, "--clr-all"],
                check=True,
                capture_output=True
            )
            self.configured_devices.clear()
            logger.info("Configuration HidHide réinitialisée")
        except subprocess.CalledProcessError as HidHide_reset_all_error:
            logger.error(f"Erreur reset HidHide: {HidHide_reset_all_error}")

    def cleanup(self):
        """Nettoyage à la fermeture"""
        logger.info("🧹 Nettoyage HidHide...")
        self.reset_all()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import time

    print("🧪 TEST HIDHIDE MANAGER")
    print("=" * 70)
    print()

    try:
        # ===================================================================
        # ÉTAPE 1 : CRÉATION DES MANETTES (AVANT HidHide)
        # ===================================================================
        print("📋 ÉTAPE 1/4 : Création des manettes virtuelles")
        print("-" * 70)

        gamepads = []

        try:
            num_gamepads = 2
            print(f"Création de {num_gamepads} manettes...")

            for i in range(num_gamepads):
                gamepad = vg.VX360Gamepad()
                gamepads.append(gamepad)
                print(f"   ✅ Manette #{i} créée")
                time.sleep(0.3)

            print()
            print("⏳ Attente détection par le système (2s)...")
            time.sleep(2.0)
            print("✅ Manettes créées et prêtes")
            print()

        except ImportError:
            print("❌ vgamepad non installé")
            print("💡 Installe avec: pip install vgamepad")
            print()
            exit(1)

        # ===================================================================
        # ÉTAPE 2 : INITIALISATION HIDHIDE
        # ===================================================================
        print("📋 ÉTAPE 2/4 : Initialisation HidHide")
        print("-" * 70)

        manager = HidHideManager()
        print("✅ HidHide initialisé")
        print()

        # ===================================================================
        # ÉTAPE 3 : LISTAGE DES DEVICES
        # ===================================================================
        print("📋 ÉTAPE 3/4 : Détection des manettes virtuelles")
        print("-" * 70)

        found_devices = []

        for i in range(num_gamepads):
            print(f"Recherche manette #{i}...")
            device = manager.get_vgamepad_device_path(i)

            if device:
                found_devices.append(device)
                print(f"   ✅ Trouvée : {device[:70]}...")
            else:
                print(f"   ❌ Non trouvée")

        print()
        print(f"📊 Résumé : {len(found_devices)}/{num_gamepads} manettes détectées")
        print()

        if len(found_devices) == 0:
            print("⚠️  Aucune manette détectée - Impossible de continuer")
            print()
            print("💡 Solutions :")
            print("   1. Vérifie que ViGEmBus est installé")
            print("   2. Redémarre le PC")
            print("   3. Réinstalle vgamepad")
            raise RuntimeError("Aucune manette détectée")

        # ===================================================================
        # ÉTAPE 4 : STATISTIQUES
        # ===================================================================
        print("📋 ÉTAPE 4/4 : Statistiques")
        print("-" * 70)
        print(f"Devices configurés  : {len(manager.configured_devices)}")
        print(f"Manettes créées     : {len(gamepads)}")
        print(f"Manettes détectées  : {len(found_devices)}")
        print()

        # ===================================================================
        # NETTOYAGE
        # ===================================================================
        print("=" * 70)
        print()
        input("✅ Test terminé ! Appuie sur ENTRÉE pour nettoyer et quitter...")
        print()

        print("🧹 Nettoyage...")

        # Cleanup gamepads
        if gamepads:
            for i, gp in enumerate(gamepads):
                try:
                    gp.reset()
                    gp.update()
                except (AttributeError, RuntimeError) as gamepad_cleanup_error:
                    # AttributeError: gamepad object invalid
                    # RuntimeError: ViGEm bus disconnected
                    print(f"   ⚠️  Gamepad #{i} cleanup failed: {gamepad_cleanup_error}")
            print(f"   ✅ {len(gamepads)} gamepad(s) cleaned")

        # Réinitialiser HidHide
        manager.reset_all()
        print("   ✅ HidHide réinitialisé")

        print()
        print("✅ Nettoyage terminé")

    except RuntimeError as test_error:
        print()
        print(f"❌ ERREUR: {test_error}")
        print()
        print("💡 Solutions:")
        print("   1. Installer HidHide:")
        print("      https://github.com/ViGEm/HidHide/releases")
        print("   2. Redémarrer après installation")
        print("   3. Lancer ce script en administrateur")

    except Exception as e:
        print()
        print(f"❌ Erreur inattendue: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("👋 Fin du test")