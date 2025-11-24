import vgamepad as vg
import time
import pyautogui
import pymem
import pymem.process
from typing import Dict, List, Set, Tuple, Optional
import struct
import ctypes
from datetime import datetime


class DolphinMemoryScanner:
    """Gère le scan optimisé de la mémoire de Dolphin"""

    def __init__(self, process_name: str = "Dolphin.exe"):
        self.process_name = process_name
        self.pm = None
        self.mem1_base = None
        self.mem1_size = 0x02000000  # 32MB - taille standard de MEM1
        self.scan_size = 0x01000000  # Scanner seulement les 16 premiers MB (inputs généralement ici)

    def attach(self) -> bool:
        """Attache le scanner au processus Dolphin"""
        try:
            self.pm = pymem.Pymem(self.process_name)
            print(f"✓ Attaché au processus {self.process_name} (PID: {self.pm.process_id})")

            self.mem1_base = self._find_mem1_base()
            if self.mem1_base:
                print(f"✓ MEM1 trouvée à l'adresse: 0x{self.mem1_base:016X}")
                print(f"✓ Zone de scan: {self.scan_size // 1024 // 1024}MB (optimisé)")
                return True
            else:
                print("✗ Impossible de trouver MEM1")
                return False
        except Exception as e:
            print(f"✗ Erreur lors de l'attachement: {e}")
            return False

    def _find_mem1_base(self) -> Optional[int]:
        """Trouve l'adresse de base de MEM1 dans Dolphin"""
        try:
            # Méthode 1: Chercher dans les modules
            for module in self.pm.list_modules():
                if any(x in module.name.lower() for x in ['system32', 'windows', 'program files']):
                    continue

                if module.SizeOfImage >= self.mem1_size:
                    try:
                        test_read = self.pm.read_bytes(module.lpBaseOfDll, 4)
                        return module.lpBaseOfDll
                    except:
                        continue

            # Méthode 2: Scanner la mémoire
            print("Recherche de MEM1 par scan mémoire...")
            mbi = pymem.memory.MEMORY_BASIC_INFORMATION()
            address = 0

            while address < 0x7FFFFFFF:
                if pymem.memory.virtual_query(self.pm.process_handle, address, ctypes.byref(mbi)):
                    if (mbi.State == pymem.memory.MEM_COMMIT and
                            mbi.RegionSize >= self.mem1_size and
                            mbi.Type == pymem.memory.MEM_PRIVATE):
                        return mbi.BaseAddress
                    address = mbi.BaseAddress + mbi.RegionSize
                else:
                    address += 0x1000

            return None
        except Exception as e:
            print(f"Erreur lors de la recherche de MEM1: {e}")
            return None

    def read_value(self, address: int, data_type: str) -> Optional[int]:
        """Lit une valeur à une adresse donnée"""
        try:
            if data_type == 'u32':
                bytes_data = self.pm.read_bytes(self.mem1_base + address, 4)
                return struct.unpack('>I', bytes_data)[0]
            elif data_type == 'u16':
                bytes_data = self.pm.read_bytes(self.mem1_base + address, 2)
                return struct.unpack('>H', bytes_data)[0]
            elif data_type == 'u8':
                return self.pm.read_uchar(self.mem1_base + address)
        except:
            return None

    def initial_scan(self, data_type: str) -> Dict[int, int]:
        """Scan initial rapide avec lecture par blocs"""
        print(f"  📊 Scan initial ({data_type})...")
        memory_snapshot = {}

        block_size = 0x100000  # 1MB par bloc
        step = {'u8': 1, 'u16': 2, 'u32': 4}[data_type]
        total_blocks = self.scan_size // block_size

        for block_idx in range(total_blocks):
            offset = block_idx * block_size

            if block_idx % 2 == 0:
                progress = (block_idx / total_blocks) * 100
                print(f"    Progression: {progress:.0f}%", end='\r')

            try:
                chunk = self.pm.read_bytes(self.mem1_base + offset, block_size)

                for i in range(0, len(chunk) - step + 1, step):
                    addr = offset + i
                    if data_type == 'u32':
                        value = struct.unpack('>I', chunk[i:i + 4])[0]
                    elif data_type == 'u16':
                        value = struct.unpack('>H', chunk[i:i + 2])[0]
                    else:
                        value = chunk[i]

                    memory_snapshot[addr] = value
            except:
                continue

        print(f"    ✓ {len(memory_snapshot):,} adresses")
        return memory_snapshot

    def filter_by_value_change(self, addresses: Set[int], old_values: Dict[int, int],
                               data_type: str, condition: str = 'increased') -> Set[int]:
        """
        Filtre optimisé par changement de valeur
        condition: 'increased', 'decreased', 'changed', 'unchanged'
        """
        result = set()

        # Grouper par blocs pour lecture efficace
        addresses_by_block = {}
        block_size = 0x100000

        for addr in addresses:
            block_start = (addr // block_size) * block_size
            if block_start not in addresses_by_block:
                addresses_by_block[block_start] = []
            addresses_by_block[block_start].append(addr)

        step = {'u8': 1, 'u16': 2, 'u32': 4}[data_type]

        for block_start, block_addresses in addresses_by_block.items():
            try:
                chunk = self.pm.read_bytes(self.mem1_base + block_start, block_size)

                for addr in block_addresses:
                    offset = addr - block_start

                    if offset + step > len(chunk) or addr not in old_values:
                        continue

                    # Lire la nouvelle valeur
                    if data_type == 'u32':
                        new_value = struct.unpack('>I', chunk[offset:offset + 4])[0]
                    elif data_type == 'u16':
                        new_value = struct.unpack('>H', chunk[offset:offset + 2])[0]
                    else:
                        new_value = chunk[offset]

                    old_value = old_values[addr]

                    # Appliquer la condition
                    if condition == 'increased' and new_value > old_value:
                        result.add(addr)
                    elif condition == 'decreased' and new_value < old_value:
                        result.add(addr)
                    elif condition == 'changed' and new_value != old_value:
                        result.add(addr)
                    elif condition == 'unchanged' and new_value == old_value:
                        result.add(addr)
            except:
                continue

        return result

    def scan_addresses_values(self, addresses: Set[int], data_type: str) -> Dict[int, int]:
        """Lit les valeurs actuelles d'un ensemble d'adresses"""
        values = {}

        # Grouper par blocs
        addresses_by_block = {}
        block_size = 0x100000

        for addr in addresses:
            block_start = (addr // block_size) * block_size
            if block_start not in addresses_by_block:
                addresses_by_block[block_start] = []
            addresses_by_block[block_start].append(addr)

        step = {'u8': 1, 'u16': 2, 'u32': 4}[data_type]

        for block_start, block_addresses in addresses_by_block.items():
            try:
                chunk = self.pm.read_bytes(self.mem1_base + block_start, block_size)

                for addr in block_addresses:
                    offset = addr - block_start

                    if offset + step > len(chunk):
                        continue

                    if data_type == 'u32':
                        value = struct.unpack('>I', chunk[offset:offset + 4])[0]
                    elif data_type == 'u16':
                        value = struct.unpack('>H', chunk[offset:offset + 2])[0]
                    else:
                        value = chunk[offset]

                    values[addr] = value
            except:
                continue

        return values


class DolphinInputScanner:
    """Scanner d'inputs avec séquences complexes"""

    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        self.memory_scanner = DolphinMemoryScanner()

        # Mapping des inputs
        self.test_inputs = {
            'A_Button': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A),
            'B_Button': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B),
            'X_Button': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X),
            'Y_Button': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y),
            'Start': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_START),
            'L_Trigger': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER),
            'R_Trigger': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER),
            'DPad_Up': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP),
            'DPad_Down': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN),
            'DPad_Left': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT),
            'DPad_Right': lambda: self.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT),
            'Stick_X_Pos': lambda: self.move_stick_x(32767),
            'Stick_X_Neg': lambda: self.move_stick_x(-32768),
            'Stick_Y_Pos': lambda: self.move_stick_y(32767),
            'Stick_Y_Neg': lambda: self.move_stick_y(-32768),
        }

        self.results = {'u8': {}, 'u16': {}, 'u32': {}}

    def press_button(self, button):
        """Presse un bouton"""
        self.gamepad.press_button(button=button)
        self.gamepad.update()
        time.sleep(0.08)
        self.gamepad.release_button(button=button)
        self.gamepad.update()

    def hold_button(self, button, duration: float = 0.3):
        """Maintient un bouton enfoncé"""
        self.gamepad.press_button(button=button)
        self.gamepad.update()
        time.sleep(duration)
        self.gamepad.release_button(button=button)
        self.gamepad.update()

    def spam_button(self, button, count: int = 5):
        """Spam un bouton rapidement"""
        for _ in range(count):
            self.gamepad.press_button(button=button)
            self.gamepad.update()
            time.sleep(0.05)
            self.gamepad.release_button(button=button)
            self.gamepad.update()
            time.sleep(0.05)

    def move_stick_x(self, value):
        """Déplace le stick X"""
        self.gamepad.left_joystick(x_value=value, y_value=0)
        self.gamepad.update()
        time.sleep(0.08)
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.update()

    def move_stick_y(self, value):
        """Déplace le stick Y"""
        self.gamepad.left_joystick(x_value=0, y_value=value)
        self.gamepad.update()
        time.sleep(0.08)
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.update()

    def reset_gamepad(self):
        """Remet la manette à l'état neutre"""
        self.gamepad.reset()
        self.gamepad.update()

    def load_save_state(self):
        """Simule F5 pour charger la save state"""
        print("\n⟳ Chargement de la save state (F5)...")
        pyautogui.press('f5')
        time.sleep(2.0)

    def complex_input_sequence(self, input_name: str, input_func, data_type: str,
                               initial_snapshot: Dict[int, int]) -> Set[int]:
        """
        Séquence d'inputs complexe pour mieux filtrer les adresses

        Étapes:
        1. Neutre (vérif baseline)
        2. Single press (détection changement)
        3. Retour neutre (vérif retour)
        4. Hold (détection valeur maintenue)
        5. Retour neutre
        6. Spam (détection pattern répétitif)
        7. Retour neutre final
        """
        print(f"\n{'─' * 50}")
        print(f"🎮 Testing: {input_name} ({data_type})")

        # Convertir le snapshot initial en Set d'adresses
        all_addresses = set(initial_snapshot.keys())
        candidates = all_addresses.copy()

        # ÉTAPE 1: Baseline neutre
        print(f"  [1/7] 📍 Baseline neutre...")
        time.sleep(0.2)
        baseline_values = initial_snapshot

        # ÉTAPE 2: Single press - détecter augmentation
        print(f"  [2/7] 👆 Single press...")
        input_func()
        time.sleep(0.1)

        print(f"        → Recherche valeurs augmentées...")
        candidates = self.memory_scanner.filter_by_value_change(
            candidates, baseline_values, data_type, 'increased'
        )
        print(f"        → {len(candidates):,} adresses ont augmenté")

        if len(candidates) == 0:
            print(f"  ❌ Aucune adresse trouvée")
            return set()

        # ÉTAPE 3: Retour neutre - vérifier retour à baseline
        print(f"  [3/7] ⏹ Retour neutre...")
        self.reset_gamepad()
        time.sleep(0.2)

        print(f"        → Vérification retour baseline...")
        candidates = self.memory_scanner.filter_by_value_change(
            candidates, baseline_values, data_type, 'unchanged'
        )
        print(f"        → {len(candidates):,} adresses revenues au neutre")

        if len(candidates) == 0:
            print(f"  ❌ Aucune adresse stable")
            return set()

        # Capturer les valeurs actuelles (baseline)
        current_values = self.memory_scanner.scan_addresses_values(candidates, data_type)

        # ÉTAPE 4: Hold - maintenir enfoncé
        print(f"  [4/7] 🔒 Hold (maintien)...")

        # Obtenir le bouton approprié pour hold
        if 'Button' in input_name or 'Trigger' in input_name:
            button_map = {
                'A_Button': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                'B_Button': vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                'X_Button': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
                'Y_Button': vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
                'Start': vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
                'L_Trigger': vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
                'R_Trigger': vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
                'DPad_Up': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
                'DPad_Down': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
                'DPad_Left': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
                'DPad_Right': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
            }
            if input_name in button_map:
                self.hold_button(button_map[input_name], 0.3)
        else:
            input_func()

        time.sleep(0.1)

        print(f"        → Vérification valeur différente...")
        candidates = self.memory_scanner.filter_by_value_change(
            candidates, current_values, data_type, 'changed'
        )
        print(f"        → {len(candidates):,} adresses ont changé")

        if len(candidates) == 0:
            print(f"  ❌ Aucune adresse réactive au hold")
            return set()

        # ÉTAPE 5: Retour neutre
        print(f"  [5/7] ⏹ Retour neutre...")
        self.reset_gamepad()
        time.sleep(0.2)

        current_values = self.memory_scanner.scan_addresses_values(candidates, data_type)

        # ÉTAPE 6: Spam - pattern répétitif
        print(f"  [6/7] ⚡ Spam (détection pattern)...")

        # Scanner pendant le spam pour voir les changements en temps réel
        spam_samples = []
        if 'Button' in input_name or 'Trigger' in input_name:
            button_map = {
                'A_Button': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                'B_Button': vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                'X_Button': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
                'Y_Button': vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
                'Start': vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
                'L_Trigger': vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
                'R_Trigger': vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
                'DPad_Up': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
                'DPad_Down': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
                'DPad_Left': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
                'DPad_Right': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
            }
            if input_name in button_map:
                for i in range(3):
                    self.gamepad.press_button(button_map[input_name])
                    self.gamepad.update()
                    time.sleep(0.05)
                    # Scanner pendant l'appui
                    spam_values = self.memory_scanner.scan_addresses_values(candidates, data_type)
                    spam_samples.append(spam_values)

                    self.gamepad.release_button(button_map[input_name])
                    self.gamepad.update()
                    time.sleep(0.05)
        else:
            for i in range(3):
                input_func()
                spam_values = self.memory_scanner.scan_addresses_values(candidates, data_type)
                spam_samples.append(spam_values)
                time.sleep(0.05)

        # Filtrer les adresses qui ont changé pendant chaque spam
        spam_candidates = set()
        for sample in spam_samples:
            for addr in candidates:
                if addr in current_values and addr in sample:
                    if sample[addr] != current_values[addr]:
                        spam_candidates.add(addr)

        candidates = spam_candidates
        print(f"        → {len(candidates):,} adresses réactives au spam")

        if len(candidates) == 0:
            print(f"  ❌ Aucune adresse réactive au spam")
            return set()

        # ÉTAPE 7: Retour neutre final
        print(f"  [7/7] ⏹ Retour neutre final...")
        self.reset_gamepad()
        time.sleep(0.2)

        print(f"        → Vérification stabilité...")
        candidates = self.memory_scanner.filter_by_value_change(
            candidates, baseline_values, data_type, 'unchanged'
        )

        print(f"  ✅ {len(candidates)} adresses candidates finales")

        return candidates

    def run_full_scan(self, num_iterations: int = 2):
        """Exécute le scan complet avec séquences complexes"""
        print("=" * 60)
        print("  SCANNER AVANCÉ D'INPUTS DOLPHIN - MH TRI")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  • Inputs à tester: {len(self.test_inputs)}")
        print(f"  • Itérations: {num_iterations}")
        print(f"  • Types: u8, u16, u32")
        print(f"  • Séquences: 7 étapes par input")
        print("\n⚠ Prérequis:")
        print("  1. Dolphin + Monster Hunter Tri")
        print("  2. Save state sur F5")
        print("  3. Manette virtuelle active")
        print("  4. Dans le jeu (pas menu)")

        print("\n" + "─" * 60)
        if not self.memory_scanner.attach():
            print("\n✗ Échec attachement Dolphin!")
            return

        print("\n⏸ Appuyez sur Entrée pour démarrer...")
        input()

        # Scanner pour chaque type de données
        for data_type in ['u16', 'u8', 'u32']:  # u16 en premier (plus probable pour boutons)
            print("\n" + "=" * 60)
            print(f"  SCAN TYPE: {data_type.upper()}")
            print("=" * 60)

            for iteration in range(1, num_iterations + 1):
                print(f"\n🔄 Itération {iteration}/{num_iterations}")

                # Snapshot initial
                print("\n📸 Création snapshot mémoire...")
                memory_snapshot = self.memory_scanner.initial_scan(data_type)

                # Tester chaque input
                for input_name, input_func in self.test_inputs.items():
                    try:
                        candidates = self.complex_input_sequence(
                            input_name, input_func, data_type, memory_snapshot
                        )

                        if iteration == 1:
                            if candidates:
                                self.results[data_type][input_name] = candidates
                        else:
                            if input_name in self.results[data_type]:
                                # Intersection avec résultats précédents
                                self.results[data_type][input_name] &= candidates
                                if not self.results[data_type][input_name]:
                                    del self.results[data_type][input_name]

                    except Exception as e:
                        print(f"  ✗ Erreur {input_name}: {e}")
                        continue

                # Save state pour prochaine itération
                if iteration < num_iterations:
                    self.load_save_state()
                    time.sleep(1.0)

        # Résultats
        self.save_results()

    def save_results(self):
        """Sauvegarde les résultats en format texte simple"""
        print("\n" + "=" * 60)
        print("  📊 RÉSULTATS FINAUX")
        print("=" * 60)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"input_addresses_{timestamp}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ADRESSES D'INPUTS - MONSTER HUNTER TRI\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base MEM1: 0x{self.memory_scanner.mem1_base:016X}\n\n")

            has_results = False

            for data_type in ['u8', 'u16', 'u32']:
                if not self.results[data_type]:
                    continue

                has_results = True
                print(f"\n{'─' * 60}")
                print(f"TYPE: {data_type.upper()}")
                print(f"{'─' * 60}")

                f.write(f"\n{'=' * 60}\n")
                f.write(f"TYPE: {data_type.upper()}\n")
                f.write(f"{'=' * 60}\n\n")

                # Trier par qualité
                sorted_inputs = sorted(
                    self.results[data_type].items(),
                    key=lambda x: len(x[1])
                )

                for input_name, addresses in sorted_inputs:
                    count = len(addresses)
                    quality = "🟢 EXCELLENT" if count <= 5 else "🟡 BON" if count <= 20 else "🔴 NOMBREUX"

                    print(f"\n{input_name}: {quality} ({count} addr)")

                    if count <= 50:
                        addr_list = ', '.join(f"0x{addr:08X}" for addr in sorted(addresses))
                        print(f"  {addr_list}")
                        f.write(f"{input_name}: {addr_list}\n")
                    else:
                        f.write(f"{input_name}: {count} adresses (trop nombreuses)\n")

            if not has_results:
                print("\n❌ Aucun résultat trouvé!")
                f.write("\nAucun résultat trouvé.\n")
            else:
                print(f"\n{'=' * 60}")
                print(f"✅ Résultats sauvegardés: {output_file}")
                print(f"{'=' * 60}")

                # Stats
                total = sum(len(addrs) for dt in self.results.values() for addrs in dt.values())
                excellent = sum(1 for dt in self.results.values()
                                for addrs in dt.values() if len(addrs) <= 5)

                print(f"\n📈 Statistiques:")
                print(f"  Total d'adresses uniques trouvées: {total}")
                print(f"  Résultats excellents (≤5 addr): {excellent}")


def main():
    scanner = DolphinInputScanner()

    try:
        scanner.run_full_scan(num_iterations=2)
    except KeyboardInterrupt:
        print("\n\n⏹ Scan interrompu")
    except Exception as e:
        print(f"\n\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nNettoyage...")
        scanner.reset_gamepad()
        print("✅ Terminé!")


if __name__ == "__main__":
    main()