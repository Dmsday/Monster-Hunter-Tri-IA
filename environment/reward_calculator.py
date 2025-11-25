"""
Reward Calculator v2
"""

import numpy as np
import time

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from utils.module_logger import get_module_logger
logger = get_module_logger('reward_calculator')

from environment.exploration_tracker import ExplorationTracker


class MonsterHunterRewardCalculator:
    """
    Calculateur de reward
    """

    def __init__(self):
        # État précédent
        self.prev_hp = None
        self.prev_stamina = None
        self.hp_recovery_given = False  # Flag pour bonus HP recovery
        self.stamina_recovery_given = False  # Flag pour bonus stamina recovery
        self.hp_recovery_accumulated = 0.0  # accumulateur HP
        self.stamina_recovery_accumulated = 0.0  # accumulateur stamina

        self._combat_log_count = 0
        self._debug_marker_check_count = 0
        self.prev_damage_flag = None
        self.prev_death_count = 0

        self.prev_position = None
        self.prev_orientation = None
        self.prev_zone = None

        self.prev_sharpness = None

        # Oxygène
        self.prev_oxygen = None
        self.oxygen_low_start_time = None
        self.last_oxygen_penalty_time = None
        self.oxygen_penalty_count = 0

        # Cartographie
        self.cube_size = 650 # Taille d'un cube
        self.max_cubes = 250 # Nombre max avant compression
        self.exploration_tracker = ExplorationTracker(
            cube_size=self.cube_size,
            max_cubes=self.max_cubes,
            compression_target=0.85  # Cible de compression (85%)
        )

        # MENU DU JEU (Start toggle)
        self.game_menu_entry_time = None  # Timestamp entrée menu
        self.game_menu_total_time = 0.0  # Temps total dans le menu
        self.last_menu_penalty_time = None  # Dernier malus appliqué
        self.game_menu_open_count = 0
        self.prev_in_menu = False

        # Timer pour mise à jour marqueurs
        self.last_marker_update = time.time()
        self.marker_update_interval = 1.0  # Mettre à jour toutes les secondes

        #---------------------
        # Constantes de reward
        #---------------------
        # Exploration
        self.BONUS_NEW_ZONE_DISCOVERED = 2.0
        self.BONUS_NEW_AREA_DISCOVERED = 0.6
        self.PENALTY_REVISIT_AREA = 0.1

        # Menu du jeu
        self.PENALTY_MENU_THRESHOLD = 5.0  # Seuil avant malus (secondes)
        self.PENALTY_MENU_BASE = 0.6  # Malus de base
        self.PENALTY_MENU_RECURRING = 0.2  # Malus récurrent (par 3s)
        self.MENU_RECURRING_DELAY = 3.0  # Délai entre malus récurrents

        # Tracking présence monstres
        self.zone_has_monsters = False  # Zone actuelle a des monstres ?
        self.prev_zone_had_monsters = False  # Zone précédente avait des monstres ?
        self.prev_in_combat = False  # État combat précédent pour détecter changements
        self.frames_in_monster_zone = 0  # Compteur de frames en zone avec monstres
        self.left_monster_zone_count = 0  # Nombre de fois qu'on quitte une zone avec monstres
        self._last_monster_count = 0  # Compteur interne pour GUI
        self._monsters_detected_logged = False

        # Tracking coups donnés
        self.total_damage_dealt = 0
        self.hit_count = 0
        self.consecutive_hits = 0

        # Tracking immobilité
        self.frames_stationary = 0
        self.idle_start_time = None

        # Exploration
        self.total_distance_traveled = 0.0
        self.frames_stationary = 0

        # Camp avec reset après mort
        self.camp_entry_time = None
        self.camp_total_time = 0.0
        self.camp_penalty_triggered = False
        self.last_camp_penalty_period = 0

        # Sortie camp
        self.first_camp_exit = False
        self.just_died = False
        self.camp_exit_after_death = False

        # Zone cooldown
        self.last_zone_change_time = None
        self.zone_change_cooldown = 7.0

        # Tracking dégâts pour réduction malus mort
        self.prev_small_monsters_hp = {}
        self.prev_large_monsters_hp = {}
        self.monsters_hit_count = 0
        self.total_monster_damage = 0
        self.monsters_killed_count = 0
        self.monster_damage_since_zone_change = 0

        # Breakdown
        self.reward_breakdown = {}
        self.reward_breakdown_detailed = {}
        self.last_reward_details = {}

        # Multiplicateurs de base
        self.REWARD_SURVIVAL = 0.005
        self.REWARD_HIT_BASE = 1.0
        self.REWARD_HIT_MULTIPLIER = 0.02
        self.REWARD_ATTACK_ATTEMPT = 0.1

        # Penalite HP dynamique
        self.PENALTY_DAMAGE_BASE = 0.1
        self.PENALTY_BIG_HIT = 0.4
        self.PENALTY_LOW_HP = 0.04
        self.PENALTY_CRITICAL_HP = 0.1
        self.PENALTY_LOW_STAMINA = 0.02
        self.BONUS_GOOD_HEALTH = 0.02
        self.PENALTY_IDLE = 0.005
        self.REWARD_VICTORY = 200.0
        self.PENALTY_DEATH_BASE = 30.0
        self.PENALTY_QUEST_FAILED = 60.0

        # Oxygène
        self.PENALTY_OXYGEN_INITIAL = 0.7
        self.PENALTY_OXYGEN_RECURRING = 0.2
        self.BONUS_OXYGEN_RECOVERY = 0.6
        self.OXYGEN_LOW_THRESHOLD = 25
        self.OXYGEN_SAFE_THRESHOLD = 50
        self.OXYGEN_PENALTY_DELAY = 10.0
        self.OXYGEN_RECURRING_DELAY = 2.0

        # Constantes curiosité
        self.BONUS_NEW_ZONE_DISCOVERED = 2.0  # Gros bonus pour nouvelle map
        self.BONUS_NEW_AREA_DISCOVERED = 0.6  # Bonus moyen pour nouvelle zone de la map
        self.PENALTY_REVISIT_AREA = 0.04  # Petit malus si on revient trop souvent au même endroit
        self.REVISIT_THRESHOLD = 3  # Nombre de visites avant malus

        # Constantes de présence monstres
        self.BONUS_IN_MONSTER_ZONE = 0.02  # Petit bonus constant par frame en zone avec monstres
        self.PENALTY_LEFT_MONSTER_ZONE = 1.5  # Malus si on quitte une zone avec monstres
        self.BONUS_MONSTER_ZONE_PERSISTENCE = 0.005  # Bonus supplémentaire pour rester longtemps

        # Timer pour combat
        self.last_damage_time = 0.0
        self.combat_timeout = 10.0  # 10 secondes sans dégât = fin combat

        # Bonus actions defensives
        self.BONUS_BLOCK_ATTEMPT = 0.04
        self.BONUS_DODGE_ATTEMPT = 0.06
        self.BONUS_ATTACK_ATTEMPT = 0.05

        # Exploration (avec bruit)
        self.BONUS_EXPLORATION_BASE = 0.01 # Inutilisé, à garder
        self.EXPLORATION_NOISE_SCALE = 0.005 # Inutilisé, à garder
        self.PENALTY_STATIONARY = 0.004

        # Bonus zone change
        self.BONUS_ZONE_CHANGE = 1.0
        self.BONUS_FIRST_CAMP_EXIT = 3.0
        self.BONUS_CAMP_EXIT_AFTER_DEATH = 2.0

        # Camp pénalités
        self.PENALTY_CAMP_THRESHOLD = 30.0
        self.PENALTY_CAMP_BASE = 0.5
        self.PENALTY_CAMP_INCREMENT = 0.4
        self.PENALTY_CAMP_MAX = 3.0

        # Sharpness
        self.PENALTY_LOW_SHARPNESS = 0.1
        self.PENALTY_BOUNCED = 1.0

        # Monster HP
        self.REWARD_MONSTER_HIT = 2.0
        self.REWARD_MONSTER_DAMAGE_MULT = 0.03

        # Bonus kill
        self.BONUS_KILL_SMALL_MONSTER = 6.0
        self.BONUS_KILL_LARGE_MONSTER = 20.0

    @staticmethod
    def _calculate_hp_penalty_multiplier(current_hp: float) -> float:
        """Calcul pénalité HP exponentielle"""
        if current_hp <= 0:
            return 2.0

        multiplier = np.exp(-current_hp / 50.0)
        return np.clip(multiplier, 0.1, 2.0)

    def _check_curiosity_rewards(self, current_state: dict, reward: float, info: dict) -> float:
        """
        Calcule les récompenses de curiosité
        """
        current_zone = current_state.get('current_zone', 0) or 0
        x = current_state.get('player_x')
        z = current_state.get('player_z')
        y = current_state.get('player_y')

        curiosity_reward = 0.0

        # Vérifier que les positions sont valides
        if x is None or y is None or z is None:
            return reward

        # UTILISER LE TRACKER D'EXPLORATION
        # Note: on passe last_action depuis les infos pour la détection d'obstacles
        exploration_result = self.exploration_tracker.update_position(
            x, y, z,
            zone_id=current_zone,
            action=info.get('last_action'),  # Optionnel pour détection obstacles
        )

        # Récompense de découverte
        discovery_reward = exploration_result['discovery_reward']
        max_discovery_reward_per_step = 2.0  # Max 2.0 de reward discovery par step

        # Si dans le camp (zone 0), diviser par 20
        if current_zone == 0:
            if hasattr(self, 'just_died') and self.just_died:
                # Aucune reward d'exploration dans le camp après mort
                discovery_reward = 0.0
                info['camp_exploration_blocked'] = True
            else:
                # Sinon, diviser par 20
                discovery_reward /= 20.0

            if discovery_reward > 0:
                info['exploration_camp_penalty'] = True

        # Clipper la reward
        discovery_reward = min(discovery_reward, max_discovery_reward_per_step)
        # Bonus clippe
        curiosity_reward += discovery_reward

        # Détail
        if discovery_reward > 0:
            if exploration_result.get('new_cube'):
                self.reward_breakdown_detailed[
                    'exploration.new_cube_bonus'] = discovery_reward
            else:
                self.reward_breakdown_detailed['exploration.new_zone_bonus'] = discovery_reward

        # Malus de revisit (déjà calculé par le tracker)
        revisit_penalty = exploration_result.get('revisit_penalty', 0.0)
        curiosity_reward -= revisit_penalty

        # Détail
        if revisit_penalty > 0:
            self.reward_breakdown_detailed['exploration.revisit_penalty'] = -revisit_penalty

        # Infos pour logging
        if exploration_result['new_cube']:
            info['new_cube_discovered'] = True

        if exploration_result.get('cube_created'):
            info['cube_created'] = True

        if exploration_result['visit_count'] > 0:
            info['cube_visit_count'] = exploration_result['visit_count']

        if revisit_penalty > 0:
            info['revisit_penalty'] = revisit_penalty

        # Ajouter stats d'exploration aux infos
        tracker_stats = self.exploration_tracker.get_stats()
        info['total_cubes'] = tracker_stats['total_cubes']
        info['zones_discovered'] = tracker_stats['zones_discovered']
        info['exploration_visits'] = tracker_stats['total_visits']

        # Ajouter au breakdown
        if 'exploration' not in self.reward_breakdown:
            self.reward_breakdown['exploration'] = 0.0

        self.reward_breakdown['exploration'] += curiosity_reward
        reward += curiosity_reward

        return reward

    def _check_monster_zone_rewards(self, current_state: dict, reward: float, info: dict, zone_just_changed: bool) -> float:
        """
        Récompenses liées à la présence de monstres

        Logique :
        1. Détecte si la zone actuelle a des monstres (SMONSTER_HP > 0)
        2. Bonus constant pour rester en zone avec monstres
        3. Malus si on quitte une zone avec monstres pour aller en zone vide

        Args:
            current_state: État actuel
            reward: Reward actuelle
            info: Dict d'infos

        Returns:
            Reward ajustée
        """
        current_zone = current_state.get('current_zone', 0) or 0
        monster_reward = 0.0

        # Récupérer le nombre de monstres depuis info
        monster_count = info.get('monster_count', 0)

        # Utiliser in_combat depuis info (déjà calculé)
        in_combat = getattr(self, 'prev_in_combat', False)  # Fallback à False si pas défini

        # 1. BONUS CONSTANT POUR ÊTRE EN ZONE AVEC MONSTRES
        if self.zone_has_monsters and current_zone != 0:  # Pas dans le camp
            self.frames_in_monster_zone += 1

            # Bonus de base (seulement si en combat ou frames < 100)
            if in_combat or self.frames_in_monster_zone < 100:
                monster_reward += self.BONUS_IN_MONSTER_ZONE

            # Détail
            self.reward_breakdown_detailed['monster_zone.in_zone_bonus'] = self.BONUS_IN_MONSTER_ZONE

            # Pas de bonus si immobile
            if self.frames_stationary < 30:  # Seulement si on bouge
                monster_reward += self.BONUS_IN_MONSTER_ZONE
            else:
                info['monster_zone_bonus_blocked'] = True

            # Bonus supplémentaire pour persistance (encourage à rester)
            if self.frames_in_monster_zone > 100:  # Après ~3 secondes
                persistence_bonus = self.BONUS_MONSTER_ZONE_PERSISTENCE
                monster_reward += persistence_bonus

                # Détail
                self.reward_breakdown_detailed['monster_zone.persistence_bonus'] = persistence_bonus

            info['in_monster_zone'] = True
            info['monster_count'] = monster_count
            info['frames_in_monster_zone'] = self.frames_in_monster_zone

        # 2. MALUS SI ON QUITTE UNE ZONE AVEC MONSTRES
        # Condition : transition True → False ET ce n'est PAS un changement de zone
        if self.prev_zone_had_monsters and not self.zone_has_monsters and current_zone != 0 and not zone_just_changed:
            # On est passé d'une zone avec monstres à une zone sans
            monster_reward -= self.PENALTY_LEFT_MONSTER_ZONE
            self.left_monster_zone_count += 1

            # Détail
            self.reward_breakdown_detailed['monster_zone.left_zone_penalty'] = -self.PENALTY_LEFT_MONSTER_ZONE

            info['left_monster_zone'] = True
            info['left_monster_zone_count'] = self.left_monster_zone_count
            logger.debug(f"QUITTÉ ZONE AVEC MONSTRES ! Pénalité: -{self.PENALTY_LEFT_MONSTER_ZONE:.1f}")

            # Reset compteur de frames
            self.frames_in_monster_zone = 0

        # 3. BONUS SI ON ENTRE EN ZONE AVEC MONSTRES
        # Condition : transition False → True ET ce n'est PAS un changement de zone
        elif not self.prev_zone_had_monsters and self.zone_has_monsters and not zone_just_changed:
            self.frames_in_monster_zone = 0
            info['entered_monster_zone'] = True
            logger.debug(f"ENTRÉE EN ZONE AVEC MONSTRES ! ({monster_count} monstres détectés)")

        # Sauvegarder pour le prochain step
        self.prev_zone_had_monsters = self.zone_has_monsters

        # Ajouter au breakdown
        if 'monster_zone' not in self.reward_breakdown:
            self.reward_breakdown['monster_zone'] = 0.0

        self.reward_breakdown['monster_zone'] += monster_reward
        reward += monster_reward

        return reward

    def _handle_oxygen_penalties(self, current_oxygen: int, reward: float, info: dict) -> float:
        """
        Gestion des pénalités/bonus oxygène avec vérifications robustes
        Système :
        - Dès <25 oxygène : malus/s immédiatement
        - Augmente progressivement
        """
        # Vérifications de sécurité
        if current_oxygen is None:
            # Pas de gestion oxygène si valeur None
            info['oxygen_status'] = 'none_detected'
            return reward

        # S'assurer que c'est un nombre valide
        try:
            current_oxygen = int(current_oxygen)
        except (ValueError, TypeError):
            info['oxygen_status'] = 'invalid_value'
            return reward

        # Vérifier que la valeur est dans un range raisonnable
        if current_oxygen < 0 or current_oxygen > 200:
            info['oxygen_status'] = f'out_of_range_{current_oxygen}'
            return reward

        current_time = time.time()
        oxygen_penalty = 0.0
        oxygen_bonus = 0.0

        # Debug : Signaler l'état actuel
        info['oxygen_level'] = current_oxygen
        info['oxygen_status'] = 'monitoring'

        # MARQUER CUBE EAU DEBUG
        if current_oxygen < 100:
            if hasattr(self, 'exploration_tracker'):
                current_cube = self.exploration_tracker.current_cube
                if current_cube:
                    self.exploration_tracker.marker_system.mark_water(current_cube)

        # Pénalité dès <25, progressivement plus sévère
        if current_oxygen < self.OXYGEN_LOW_THRESHOLD:
            # Initialiser timer si première fois
            if self.oxygen_low_start_time is None:
                self.oxygen_low_start_time = current_time
                self.last_oxygen_penalty_time = None
                info['oxygen_low_started'] = True
                logger.info(f"OXYGÈNE BAS DÉTECTÉ ! Niveau: {current_oxygen}/100")

            # CALCUL PROGRESSIF DE LA PÉNALITÉ
            # Base : -3/s
            base_penalty_per_second = 0.6

            # Multiplicateur basé sur l'oxygène restant (0-25)
            # 25 oxygène = x1.0
            # 15 oxygène = x1.5
            # 5 oxygène = x2.5
            # 1 oxygène = x3.3
            oxygen_multiplier = 1.0 + (2.3 * (1.0 - current_oxygen / 25.0))

            # Pénalité par seconde ajustée
            penalty_per_second = base_penalty_per_second * oxygen_multiplier

            # Appliquer pénalité toutes les secondes
            if self.last_oxygen_penalty_time is None:
                self.last_oxygen_penalty_time = current_time
            time_since_last = current_time - self.last_oxygen_penalty_time

            if time_since_last >= 1.0:  # Toutes les 1 seconde
                oxygen_penalty = penalty_per_second
                self.oxygen_penalty_count += 1
                self.last_oxygen_penalty_time = current_time

                # Détail
                self.reward_breakdown_detailed['oxygen.oxygen_progressive'] = -oxygen_penalty

                info['oxygen_progressive_penalty'] = True
                info['oxygen_penalty_count'] = self.oxygen_penalty_count
                info['oxygen_penalty_rate'] = penalty_per_second
                info['oxygen_status'] = f'low_oxygen_{self.oxygen_penalty_count}'

                logger.info(f"💨 OXYGÈNE CRITIQUE ({current_oxygen}) - Pénalité: -{penalty_per_second:.1f}/s (x{self.oxygen_penalty_count})")
        else:
            # Oxygène au-dessus du seuil bas
            if self.oxygen_low_start_time is not None and current_oxygen >= self.OXYGEN_SAFE_THRESHOLD:
                # Récupération !
                if self.oxygen_penalty_count > 0:
                    oxygen_bonus += self.BONUS_OXYGEN_RECOVERY

                    # Détail
                    self.reward_breakdown_detailed['oxygen.oxygen_recovery'] = self.BONUS_OXYGEN_RECOVERY

                    info['oxygen_recovery_bonus'] = True
                    info['oxygen_status'] = 'recovered'
                    logger.info(f"💨 OXYGÈNE RÉCUPÉRÉ ! Niveau: {current_oxygen} - Bonus: +{self.BONUS_OXYGEN_RECOVERY:.1f}")

                # Reset compteurs
                self.oxygen_low_start_time = None
                self.last_oxygen_penalty_time = None
                self.oxygen_penalty_count = 0
            elif self.oxygen_low_start_time is not None:
                # Entre les deux seuils (bas et safe)
                info['oxygen_status'] = 'recovering'

        # Appliquer rewards
        reward -= oxygen_penalty
        reward += oxygen_bonus

        if 'oxygen' not in self.reward_breakdown:
            self.reward_breakdown['oxygen'] = 0.0

        self.reward_breakdown['oxygen'] += oxygen_bonus - oxygen_penalty

        # Infos détaillées
        info['oxygen_penalty'] = oxygen_penalty
        info['oxygen_bonus'] = oxygen_bonus
        info['oxygen_penalty_count'] = self.oxygen_penalty_count

        return reward

    @staticmethod
    def _detect_monsters_in_zone(current_state: dict) -> tuple:
        """
        Détecte la présence de monstres dans la zone actuelle

        Returns:
            tuple: (monsters_present: bool, monster_count: int)
        """
        monsters_present = False
        monster_count = 0

        # Small monsters
        for i in range(1, 5):
            hp = current_state.get(f'smonster{i}_hp')
            if hp is not None and hp > 0:
                monsters_present = True
                monster_count += 1

        # Large monsters
        for i in range(1, 2):
            hp = current_state.get(f'lmonster{i}_hp')
            if hp is not None and hp > 0:
                monsters_present = True
                monster_count += 1

        return monsters_present, monster_count

    def calculate(
            self,
            prev_state: dict,
            current_state: dict,
            action: int,
            info: dict = None,
            took_damage: bool = False,
    ) -> float:
        """
        Calcule la reward avec équilibrage update
        """
        reward = 0.0
        info = info or {}

        # PROTECTION : Si premier step d'épisode, ignorer prev_state
        episode_steps = info.get('episode_steps', 0)
        if episode_steps == 1:
            # Premier step = pas de prev_state valide
            prev_state = None
            logger.debug("Premier step - prev_state ignoré")

        # ===================================================================
        # PROTECTION : Vérifier si l'épisode est déjà en fin
        # ===================================================================
        current_map = current_state.get('current_map')
        death_count = current_state.get('death_count', 0) or 0
        quest_time = current_state.get('quest_time', 5400)

        # Conditions d'invalidité
        is_invalid_state = (
                current_map == 45 or  # Retourner au village (quete terminer)
                (current_map == 100 and quest_time is None) or  # Toujours dans la map de quete mais temps ecoule
                current_map not in [45, 100]  # map bizarre
        )

        if is_invalid_state:
            logger.warning(f"ÉTAT INVALIDE détecté dans reward_calculator")
            logger.warning(f"MAP={current_map}, deaths={death_count}, time={quest_time}")
            logger.warning(f"→ Retour reward nulle, AUCUN calcul")

            info['invalid_state_detected'] = True
            info['current_map'] = current_map
            info['death_count'] = death_count
            info['quest_time'] = quest_time

            # NE PAS APPELER self.reset() ici (sera fait dans env.reset())
            return 0.0

        # ===================================================================
        # DOUBLE VÉRIFICATION : Flags additionnels
        # ===================================================================
        #  Vérifier aussi les flags quest_ended/on_reward_screen
        if current_state.get('quest_ended') or current_state.get('on_reward_screen'):
            logger.info(f"⚠️ Flag quest_ended détecté dans reward_calculator")
            info['quest_ended_flag_in_calc'] = True

            # Nettoyer états internes
            self.reset()

            return 0.0

        # Initialisation : S'assurer que prev_state existe
        if prev_state is None:
            prev_state = {}

        # Mise à jour marqueurs dynamiques (toutes les secondes)
        current_time = time.time()
        if current_time - self.last_marker_update >= self.marker_update_interval:
            if hasattr(self, 'exploration_tracker'):
                self.exploration_tracker.marker_system.update_dynamic_markers()
                self.last_marker_update = current_time

        # Sauvegarder l'action pour le tracker d'exploration
        info['last_action'] = action

        # Reset breakdown
        self.reward_breakdown = {
            'survival': 0.0,
            'combat': 0.0,
            'health': 0.0,
            'exploration': 0.0,
            'penalties': 0.0,
            'zone_change': 0.0,
            'defensive_actions': 0.0,
            'oxygen': 0.0,
            'monster_zone': 0.0,
            'death': 0.0,
            'damage_taken': 0.0,
            'monster_hit': 0.0,
            'hit': 0.0,
            'camp_penalty': 0.0,
            'menu_penalty': 0.0,
            'other': 0.0
        }

        # Reset du breakdown détaillé
        self.reward_breakdown_detailed = {}

        # 1. SURVIE
        survival_reward = self.REWARD_SURVIVAL
        reward += survival_reward
        self.reward_breakdown['survival'] = survival_reward
        # Pas de détail ici (simple)

        # 2. DÉGÂTS REÇUS AVEC PÉNALITÉ EXPONENTIELLE
        current_hp = current_state.get('player_hp', 0) or 0

        if prev_state and self.prev_hp is not None:
            damage_taken = self.prev_hp - current_hp

            # PROTECTION : Si delta HP aberrant (>150), c'est un reset
            if abs(damage_taken) >= 100:
                logger.debug(f"⚠️ Delta HP aberrant : {damage_taken} (prev={self.prev_hp}, current={current_hp})")
                logger.debug("→ Probable corruption après reset - delta ignoré")
                damage_taken = 0  # Ignorer complètement

            if damage_taken > 0:
                # Clipper damage_taken AVANT calcul (max 99 HP de dégâts par step)
                damage_taken = min(damage_taken, 99)
                hp_multiplier = self._calculate_hp_penalty_multiplier(current_hp)
                damage_penalty = damage_taken * self.PENALTY_DAMAGE_BASE * hp_multiplier
                reward -= damage_penalty
                self.reward_breakdown['damage_taken'] -= damage_penalty

                # Détail
                self.reward_breakdown_detailed['health.damage_penalty'] = -damage_penalty

                current_zone = current_state.get('current_zone', 0) or 0
                # MARQUER ZONE MONSTRE - mark_monster_area
                if hasattr(self, 'exploration_tracker'):
                    current_cube = self.exploration_tracker.current_cube
                    if current_cube:
                        # Récupérer cubes environnants
                        zone_cubes = self.exploration_tracker.cubes_by_zone.get(current_zone, [])

                        self.exploration_tracker.marker_system.mark_monster_area(
                            current_cube,
                            zone_cubes,
                            max_distance=3.0
                        )

                if damage_taken > 20:
                    big_hit_penalty = self.PENALTY_BIG_HIT * hp_multiplier
                    reward -= big_hit_penalty
                    self.reward_breakdown['damage_taken'] -= big_hit_penalty

                    # Détail
                    self.reward_breakdown_detailed['health.big_hit_penalty'] = -big_hit_penalty

                info['damage_taken'] = damage_taken
                info['hp_multiplier'] = hp_multiplier

        self.prev_hp = current_hp

        # 2B. BONUS RÉCUPÉRATION HP
        if prev_state and self.prev_hp is not None:
            hp_gain = current_hp - self.prev_hp

            if hp_gain > 0:
                # Bonus = 80% des HP gagnés (DONNÉ UNE SEULE FOIS)
                if not self.hp_recovery_given:
                    hp_recovery_bonus = hp_gain * 0.8
                    reward += hp_recovery_bonus
                    self.reward_breakdown['health'] += hp_recovery_bonus
                    self.reward_breakdown_detailed['health.hp_recovery'] = hp_recovery_bonus
                    info['hp_recovered'] = hp_gain
                    self.hp_recovery_given = True  # Marquer comme donné
            else:
                # Reset flag si HP baisse (permet de redonner le bonus après dégâts)
                self.hp_recovery_given = False

        # 3. DÉGÂTS REÇUS - FLAG
        damage_flag = current_state.get('damage_last_hit')

        if damage_flag is not None and self.prev_damage_flag is not None:
            if damage_flag != self.prev_damage_flag:
                flag_penalty = 1.0
                reward -= flag_penalty
                self.reward_breakdown['damage_taken'] -= flag_penalty

                # Détail
                self.reward_breakdown_detailed['health.hit_flag_penalty'] = -flag_penalty

                info['hit_received_flag'] = True

        self.prev_damage_flag = damage_flag

        # 4. HIT DÉTECTÉ (MONSTRES HP)
        monster_hit_reward, monster_damage = self._check_monster_damage(current_state)
        if monster_hit_reward > 0:
            reward += monster_hit_reward
            self.reward_breakdown['monster_hit'] += monster_hit_reward

            # Détail
            self.reward_breakdown_detailed['monster_hit.monster_damage'] = monster_hit_reward

            self.monsters_hit_count += 1
            self.hit_count += 1
            self.total_monster_damage += monster_damage
            self.monster_damage_since_zone_change += monster_damage
            info['monster_hit'] = True

        # 5. ACTIONS (attaque, bloque, esquive)
        # Détection zone actuelle
        current_zone = current_state.get('current_zone', 0) or 0
        # Détecter changement de zone avant calcul combat
        zone_just_changed = (self.prev_zone is not None and current_zone != self.prev_zone)

        # Calculer zone_has_monsters ici (avant de l'utiliser)
        monsters_present_now, monster_count_now = self._detect_monsters_in_zone(current_state)

        # Forcer "nombre de monstres" à 0 et "monstre dans zone" à false si dans le camp (zone 0)
        if current_zone == 0:
            monsters_present_now = False
            monster_count_now = 0

        self._last_monster_count = monster_count_now

        # 1. Calculer time_since_damage AVANT toute modification de last_damage_time
        # Vérifier si last_damage_time a été initialisé (> 0)
        if self.last_damage_time > 0:
            time_since_damage = current_time - self.last_damage_time
        else:
            time_since_damage = float('inf')  # Jamais de dégât = infini

        # 2. Déterminer in_combat
        if zone_just_changed:
            # Reset complet si changement de zone
            in_combat = False
            logger.info(f"Changement zone {self.prev_zone} → {current_zone} : Reset état combat")

        elif took_damage:
            # Vient de prendre des dégâts = combat actif
            was_in_combat = self.prev_in_combat
            in_combat = True
            self.last_damage_time = current_time  # Mettre à jour le timestamp

            # Logger uniquement sur transition False → True
            if not was_in_combat:
                info['combat_started'] = True
                logger.info(f"⚔️ COMBAT ACTIF (dégâts pris)")
                logger.debug(f"DÉBUT COMBAT détecté (Zone {current_zone})")
                logger.debug(f"Monstres présents: {monster_count_now}")

        elif time_since_damage < self.combat_timeout:
            # Moins de 10s depuis dernier dégât = combat toujours actif
            in_combat = True

        else:
            # Plus de 10s sans dégât = fin combat
            in_combat = False
            if self.last_damage_time > 0:  # Éviter log au démarrage
                info['combat_ended'] = True
                logger.info(f"Fin du combat (timeout: {time_since_damage:.1f}s)")
                logger.debug(f"Raison : Timeout ({time_since_damage:.1f}s sans dégât)")
            self.last_damage_time = 0.0

        # DEBUG : Logger l'état AVANT le calcul
        total_steps = info.get('total_steps', 0)
        if total_steps > 0 and total_steps % 300 == 0:
            logger.debug(f"🔍 État combat (step {total_steps}):")
            logger.debug(f"   took_damage={took_damage}")
            logger.debug(f"   in_combat={in_combat}")
            logger.debug(f"   last_damage_time={self.last_damage_time:.1f}s")
            logger.debug(f"   time_since_damage={time_since_damage:.1f}s")
            logger.debug(f"   zone_has_monsters={self.zone_has_monsters}")
            logger.debug(f"   monsters_present_now={monsters_present_now}")
            logger.debug(f"   monster_count={self._last_monster_count}")

        # Si zone avec monstres MAIS aucun dégât depuis 15s → Ignorer
        # Règles :
        # 1. Activation si dégâts pris
        # 2. Désactivation si timeout 15s ET last_damage_time valide
        # 3. Une fois activé, rester actif jusqu'au timeout
        # 4. Ignorer monsters_present_now une fois activé

        if took_damage:
            # ACTIVATION : Dégâts pris = zone confirmée
            if not self.zone_has_monsters:
                logger.debug(f"ENTRÉE EN ZONE AVEC MONSTRES ! ({monster_count_now} monstres détectés)")
            self.zone_has_monsters = True

        elif self.zone_has_monsters:
            # Zone déjà activée : vérifier timeout
            # Condition : last_damage_time valide (> 0) ET timeout dépassé
            if self.last_damage_time > 0 and time_since_damage > 15.0:
                logger.info(f"🚫 Zone monstre DÉSACTIVÉE (pas de dégâts depuis {time_since_damage:.1f}s)")
                self.zone_has_monsters = False
                info['monster_zone_ignored_no_combat'] = True
            # Sinon : rester actif (ne pas suivre monsters_present_now)

        else:
            # Pas encore activé : détecter monstres mais NE PAS activer
            # On attend les premiers dégâts pour confirmation
            if monsters_present_now:
                # Logger uniquement la première détection (éviter spam)
                if not hasattr(self, '_monsters_detected_logged'):
                    self._monsters_detected_logged = True
                    logger.debug(f"DÉTECTION : {monster_count_now} monstres présents (attente confirmation)")
                info['monsters_detected_awaiting_damage'] = True
            else:
                # Reset flag si plus de monstres
                if hasattr(self, '_monsters_detected_logged'):
                    delattr(self, '_monsters_detected_logged')

        # Mettre a jour l'etat immediatement
        info['monsters_present'] = self.zone_has_monsters
        info['monster_count'] = monster_count_now
        info['in_monster_zone'] = self.zone_has_monsters
        info['in_combat'] = in_combat

        # DEBUG : Logger UNIQUEMENT les changements d'état combat
        if in_combat != self.prev_in_combat:
            if in_combat:
                logger.debug(f"DÉBUT COMBAT détecté (Zone {current_zone})")
                logger.debug(f"Monstres présents: {monster_count_now}")
                logger.debug(f"Temps depuis dernier dégât: {time_since_damage:.1f}s")
            else:
                logger.debug(f"FIN COMBAT (Zone {current_zone})")
                if self.zone_has_monsters:
                    logger.debug(f"Raison : Timeout ({time_since_damage:.1f}s sans dégât)")
                elif current_zone == 0:
                    logger.debug(f"Raison : Retour au camp")
                else:
                    logger.debug(f"Raison : Plus de monstres détectés")

            self.prev_in_combat = in_combat  # Sauvegarder nouvel état

        # Log périodique pour debug
        if not hasattr(self, '_combat_log_count') or self._combat_log_count is None:
            self._combat_log_count += 0
        else:
            self._combat_log_count = 1

        # Forcer "en combat" sur false si dans le camp (zone 0)
        if current_zone == 0:
            in_combat = False

        # Ajouter aux infos
        info['in_combat'] = in_combat

        # CONDITIONNER TOUTES LES ACTIONS AU COMBAT
        if action in [9, 12] and self.zone_has_monsters:  # Attaque
            if monster_hit_reward > 0:
                attack_success_bonus = 0.4 * (2.0 if in_combat else 1.0)
                reward += attack_success_bonus
                self.reward_breakdown['hit'] += attack_success_bonus
                self.reward_breakdown_detailed['hit.hit_success'] = attack_success_bonus
                info['attack_success'] = True
            else:
                # Donner reward SEULEMENT si en combat actif
                if in_combat:
                    attack_reward = self.BONUS_ATTACK_ATTEMPT * 2.0
                    reward += attack_reward
                    self.reward_breakdown['defensive_actions'] += attack_reward
                    self.reward_breakdown_detailed['defensive_actions.attack_attempt'] = attack_reward
                    info['attack_attempt'] = True
                # SINON : Aucune reward (pas de monstres ou pas en combat)

        elif action == 10 and in_combat and self.zone_has_monsters: # Bloc
            block_bonus = self.BONUS_BLOCK_ATTEMPT
            reward += block_bonus
            self.reward_breakdown['defensive_actions'] += block_bonus
            self.reward_breakdown_detailed['defensive_actions.block'] = block_bonus
            info['block_attempt'] = True

        elif action == 11 and in_combat and self.zone_has_monsters:  # esquive
            dodge_bonus = self.BONUS_DODGE_ATTEMPT
            reward += dodge_bonus
            self.reward_breakdown['defensive_actions'] += dodge_bonus
            self.reward_breakdown_detailed['defensive_actions.dodge'] = dodge_bonus
            info['dodge_attempt'] = True

        # 6. COMBO
        if self.consecutive_hits > 1:
            combo_bonus = min(self.consecutive_hits * 0.5, 5.0)
            reward += combo_bonus
            self.reward_breakdown['hit'] += combo_bonus

            # Détail
            self.reward_breakdown_detailed['hit.combo_bonus'] = combo_bonus

            info['combo'] = self.consecutive_hits

        # 7. TRACKING POSITION (pour autres calculs)
        x = current_state.get('player_x')
        y = current_state.get('player_y')
        z = current_state.get('player_z')

        if x is not None and y is not None and z is not None:
            if self.prev_position is not None:
                prev_x, prev_y, prev_z = self.prev_position

                distance = np.sqrt(
                    ((x - prev_x) ** 2) +
                    ((y - prev_y) ** 2) +
                    ((z - prev_z) ** 2)
                )

                # PROTECTION : Si distance aberrante (>5000)
                if distance > 5000:
                    logger.debug(f"⚠️ Distance aberrante : {distance:.0f}")
                    logger.debug(f"  Prev: ({prev_x:.0f}, {prev_y:.0f}, {prev_z:.0f})")
                    logger.debug(f"  Current: ({x:.0f}, {y:.0f}, {z:.0f})")
                    logger.debug("→ Ignoré (probable reset/téléport)")
                    distance = 0  # Ignorer

                if distance is not None and distance < 1000:  # Double check
                    self.total_distance_traveled += distance

                    # Initialiser exploration dans breakdown
                    if 'exploration' not in self.reward_breakdown:
                        self.reward_breakdown['exploration'] = 0.0

            self.prev_position = (x, y, z)

        # 8. CURIOSITÉ (nouvelles zones, nouvelles positions)
        reward = self._check_curiosity_rewards(current_state, reward, info)

        # 8B. CALCULER CHANGEMENT DE ZONE (utilisé par section 9 et 10)
        zone_just_changed = (self.prev_zone is not None and current_zone != self.prev_zone)

        # 9. PRÉSENCE DE MONSTRES
        reward = self._check_monster_zone_rewards(current_state, reward, info, zone_just_changed)

        # 10. CHANGEMENT DE ZONE
        if zone_just_changed:
            current_time = time.time()

            # Pause création des cubes après changement zone
            self.exploration_tracker.pause_creation(duration=2.0)

            # Reset tracking combat
            self.monster_damage_since_zone_change = 0
            self.zone_has_monsters = False  # Reset détection monstres
            self.frames_in_monster_zone = 0

            # Reset aussi le flag de détection
            if hasattr(self, '_monsters_detected_logged'):
                delattr(self, '_monsters_detected_logged')

            if self.last_zone_change_time is None:
                can_reward = True
            else:
                elapsed = current_time - self.last_zone_change_time
                can_reward = elapsed >= self.zone_change_cooldown

            if can_reward:
                zone_bonus = self.BONUS_ZONE_CHANGE

                if self.prev_zone == 0 and current_zone != 0:
                    if not self.first_camp_exit:
                        zone_bonus += self.BONUS_FIRST_CAMP_EXIT
                        self.first_camp_exit = True
                        info['first_camp_exit'] = True
                        logger.info(f"🎯 PREMIÈRE SORTIE DU CAMP ! Bonus: +{zone_bonus:.1f}")

                    elif self.just_died:
                        zone_bonus += self.BONUS_CAMP_EXIT_AFTER_DEATH
                        self.just_died = False
                        info['camp_exit_after_death'] = True
                        logger.info(f"💪 Sortie du camp après mort ! Bonus: +{zone_bonus:.1f}")

                reward += zone_bonus
                self.reward_breakdown['zone_change'] += zone_bonus

                # Détail
                if self.prev_zone == 0 and current_zone != 0:
                    if not self.first_camp_exit:
                        self.reward_breakdown_detailed['zone_change.first_exit'] = self.BONUS_FIRST_CAMP_EXIT
                        self.reward_breakdown_detailed['zone_change.zone_bonus'] = self.BONUS_ZONE_CHANGE
                    elif self.just_died:
                        self.reward_breakdown_detailed[
                            'zone_change.exit_after_death'] = self.BONUS_CAMP_EXIT_AFTER_DEATH
                        self.reward_breakdown_detailed['zone_change.zone_bonus'] = self.BONUS_ZONE_CHANGE
                else:
                    self.reward_breakdown_detailed['zone_change.zone_bonus'] = zone_bonus

                info['zone_changed'] = True
                self.last_zone_change_time = current_time

        # Sauvegarder la zone pour le prochain step
        self.prev_zone = current_zone

        # 11. PÉNALITÉ CAMP
        if current_zone == 0:
            current_time = time.time()

            if self.camp_entry_time is None:
                self.camp_entry_time = current_time
                self.last_camp_penalty_period = 0
            else:
                time_in_camp = current_time - self.camp_entry_time
                self.camp_total_time = time_in_camp

                if time_in_camp >= self.PENALTY_CAMP_THRESHOLD:
                    time_over_threshold = time_in_camp - self.PENALTY_CAMP_THRESHOLD
                    current_period = int(time_in_camp / 30.0)

                    if current_period > self.last_camp_penalty_period:
                        periods_30s = int(time_over_threshold / 30.0)
                        camp_penalty = self.PENALTY_CAMP_BASE + (periods_30s * self.PENALTY_CAMP_INCREMENT)
                        camp_penalty = min(camp_penalty, self.PENALTY_CAMP_MAX)

                        reward -= camp_penalty
                        self.reward_breakdown['camp_penalty'] -= camp_penalty

                        # Détail
                        self.reward_breakdown_detailed['penalties.camp_base'] = -self.PENALTY_CAMP_BASE
                        if periods_30s > 0:
                            self.reward_breakdown_detailed['penalties.camp_periods'] = -(
                                        periods_30s * self.PENALTY_CAMP_INCREMENT)

                        info['camp_penalty'] = camp_penalty
                        info['time_in_camp'] = time_in_camp

                        self.last_camp_penalty_period = current_period
        else:
            if self.camp_entry_time is not None:
                self.camp_entry_time = None
                self.camp_penalty_triggered = False
                self.last_camp_penalty_period = 0

        # 12. GESTION OXYGÈNE
        current_oxygen = current_state.get('time_underwater')
        if current_oxygen is not None:
            reward = self._handle_oxygen_penalties(current_oxygen, reward, info)
            # Détail (déjà géré dans _handle_oxygen_penalties)

        self.prev_oxygen = current_oxygen

        # 13. GESTION HP
        health_penalty = 0.0
        health_bonus = 0.0

        if current_hp < 30:
            health_penalty += self.PENALTY_LOW_HP
            # Détail
            self.reward_breakdown_detailed['health.low_hp_penalty'] = -self.PENALTY_LOW_HP
        if current_hp < 15:
            health_penalty += self.PENALTY_CRITICAL_HP
            # Détail
            self.reward_breakdown_detailed['health.critical_hp_penalty'] = -self.PENALTY_CRITICAL_HP

        # Bonus santé excellente (>100 HP = objet utilisé)
        if current_hp > 100 and current_zone != 0:  # Bonus si HP buffés
            health_bonus += self.BONUS_GOOD_HEALTH * 1.5  # Bonus renforcé
            self.reward_breakdown_detailed['health.buffed_hp_bonus'] = self.BONUS_GOOD_HEALTH * 1.5
        if current_hp > 80 and current_zone != 0: # Bonus normal si >80%
            health_bonus += self.BONUS_GOOD_HEALTH
            # Détail
            self.reward_breakdown_detailed['health.good_health_bonus'] = self.BONUS_GOOD_HEALTH

        reward -= health_penalty
        reward += health_bonus
        self.reward_breakdown['health'] = health_bonus - health_penalty

        # 14. GESTION STAMINA
        stamina = current_state.get('player_stamina', 0) or 0

        # PROTECTION : Vérifier delta stamina si prev_stamina existe
        if self.prev_stamina is not None:
            stamina_delta = stamina - self.prev_stamina

            # Si delta aberrant (>150), reset détecté
            if abs(stamina_delta) > 100:
                logger.warning(f"⚠️ Delta stamina aberrant : {stamina_delta}")
                logger.warning("→ Ignoré (probable reset)")
                # Ne pas calculer de bonus/malus basé sur ce delta

        self.prev_stamina = stamina

        if stamina < 22:
            stamina_penalty = self.PENALTY_LOW_STAMINA
            reward -= stamina_penalty

            # Détail
            self.reward_breakdown_detailed['penalties.stamina_low'] = -stamina_penalty

            self.reward_breakdown['penalties'] -= stamina_penalty
            info['stamina_low'] = True

        # BONUS si stamina buffée (>100 = objet utilisé)
        if stamina > 100:
            stamina_bonus = 0.02  # Petit bonus constant si stamina buffée
            reward += stamina_bonus
            self.reward_breakdown['other'] += stamina_bonus
            self.reward_breakdown_detailed['other.buffed_stamina_bonus'] = stamina_bonus
            info['stamina_buffed'] = True

        # 15. DEATH PENALTY
        death_count = current_state.get('death_count', 0) or 0

        if prev_state and self.prev_death_count is not None:
            # Apply penalty ONLY if death_count has CHANGED
            if death_count > self.prev_death_count:
                # Compute base penalty
                base_death_penalty = self.PENALTY_DEATH_BASE * (1 + 0.5 * death_count)

                # Reduction based on damage dealt
                damage_reduction_factor = self._calculate_death_penalty_reduction()
                final_death_penalty = base_death_penalty * (1.0 - damage_reduction_factor)

                # Apply immediately (BEFORE 3rd death)
                reward -= final_death_penalty
                self.reward_breakdown['death'] -= final_death_penalty
                # Details
                self.reward_breakdown_detailed['penalties.death_penalty'] = -final_death_penalty

                info['player_died'] = True
                info['death_number'] = death_count
                info['death_penalty'] = final_death_penalty

                # Signal to environment that episode should terminate after 3rd death
                if death_count >= 3:
                    info['death_count'] = death_count
                    info['critical_death'] = True
                    logger.info(f"💀💀💀 3RD DEATH - EPISODE WILL TERMINATE")
                    logger.info(f"Final penalty applied: -{final_death_penalty:.2f}")
                else:
                    logger.info(f"💀 PLAYER DIED #{death_count}")
                    logger.info(f"Base penalty: {base_death_penalty:.2f}")
                    logger.info(f"Reduction: {damage_reduction_factor:.1%}")
                    logger.info(f"Final penalty: -{final_death_penalty:.2f}")

                # Always update death_count for the next step
                info['death_count'] = death_count

                # Reset camp stats
                self.camp_entry_time = None
                self.camp_penalty_triggered = False
                self.last_camp_penalty_period = 0
                self.just_died = True
        else:
            # Initialize if first read
            self.prev_death_count = death_count

        # 16. COMPORTEMENT - MALUS IMMOBILITÉ
        # Détecter immobilité (action 0 OU pas de mouvement)
        current_pos = (current_state.get('player_x'), current_state.get('player_y'), current_state.get('player_z'))

        is_stationary = False

        # Vérifier que la position actuelle est valide
        if current_pos[0] is not None and current_pos[1] is not None and current_pos[2] is not None:
            if self.prev_position and self.prev_position[0] is not None:
                # Calculer distance 3D
                dx = current_pos[0] - self.prev_position[0]
                dy = current_pos[1] - self.prev_position[1]
                dz = current_pos[2] - self.prev_position[2]
                distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                # considérer immobile seulement si distance < 0.1 ET action = 0 (pas de mouvement intentionnel)
                if distance < 0.1 and action == 0:  # Seuil de mouvement minimal
                    is_stationary = True
                else:
                    # Reset si mouvement détecté
                    is_stationary = False

            else:
                # Pas de position précédente, considérer en mouvement
                is_stationary = False

        if is_stationary:
            self.frames_stationary += 1

            # Malus après 3 secondes (3s * 30fps = 90 frames)
            if self.frames_stationary > 90:
                # Temps d'immobilité en secondes
                idle_time = (self.frames_stationary - 90) / 30.0

                # Malus progressif, augmente avec le temps
                # Formule : -3 * (1 + idle_time/10)
                idle_multiplier = 1.0 + (idle_time / 10.0)
                idle_penalty = 0.6 * idle_multiplier / 30.0  # Par frame

                reward -= idle_penalty
                self.reward_breakdown['penalties'] -= idle_penalty
                self.reward_breakdown_detailed['penalties.idle'] = -idle_penalty

                info['idle_penalty'] = idle_penalty
                info['idle_time'] = idle_time
        else:
            # Reset compteur si mouvement
            self.frames_stationary = 0
            self.idle_start_time = None

        # 17. TRACKING ORIENTATION (pour autres calculs)
        orientation = current_state.get('player_orientation')
        self.prev_orientation = orientation

        # 18. MENU DU JEU
        menu_penalty = 0.0
        current_time = time.time()

        # Lecture directe depuis current_state
        in_menu = current_state.get('in_game_menu', False)

        # AJOUTER un flag pour éviter double comptage
        if not hasattr(self, 'prev_in_menu'):
            self.prev_in_menu = False

        # Détecter entrée dans le menu (transition False / True)
        if in_menu and not self.prev_in_menu:
            # Entrée dans le menu
            self.game_menu_entry_time = current_time
            self.last_menu_penalty_time = None
            self.game_menu_open_count += 1
            info['game_menu_opened'] = True
            info['game_menu_count'] = self.game_menu_open_count

        # Détecter sortie du menu (transition True / False)
        elif not in_menu and self.prev_in_menu:
            # Sortie du menu
            if self.game_menu_entry_time is not None:
                time_in_menu = current_time - self.game_menu_entry_time
                self.game_menu_total_time += time_in_menu
                info['game_menu_closed'] = True
                info['time_in_menu'] = time_in_menu
                self.game_menu_entry_time = None

        # Sauvegarder état actuel pour prochain step
        self.prev_in_menu = in_menu

        # Pénalité si menu ouvert trop longtemps
        if in_menu and self.game_menu_entry_time is not None:
            time_in_menu = current_time - self.game_menu_entry_time

            # Multiplicateur selon nombre d'ouvertures
            open_count_multiplier = 1.0 + (self.game_menu_open_count * 0.05)  # +5% par ouverture

            # Malus initial après 5s
            if time_in_menu >= self.PENALTY_MENU_THRESHOLD:
                if self.last_menu_penalty_time is None:
                    # Malus de base avec multiplicateur
                    base_menu_penalty = self.PENALTY_MENU_BASE * open_count_multiplier
                    menu_penalty += base_menu_penalty
                    self.last_menu_penalty_time = current_time
                    self.reward_breakdown_detailed['penalties.menu_initial'] = -base_menu_penalty
                    info['menu_initial_penalty'] = True

                    logger.info(f"⚠️ MENU ouvert >5s (ouverture #{self.game_menu_open_count})")
                    logger.info(f"Malus: -{base_menu_penalty:.2f} (x{open_count_multiplier:.1f})")
                else:
                    # Malus récurrent tous les 3s
                    time_since_last = current_time - self.last_menu_penalty_time
                    if time_since_last >= self.MENU_RECURRING_DELAY:
                        recurring_penalty = self.PENALTY_MENU_RECURRING * open_count_multiplier
                        menu_penalty += recurring_penalty
                        self.last_menu_penalty_time = current_time
                        self.reward_breakdown_detailed['penalties.menu_recurring'] = -recurring_penalty
                        info['menu_recurring_penalty'] = True

        # Appliquer pénalité
        reward -= menu_penalty
        if 'menu_penalty' not in self.reward_breakdown:
            self.reward_breakdown['menu_penalty'] = 0.0
        self.reward_breakdown['menu_penalty'] -= menu_penalty

        # Infos
        info['in_game_menu'] = in_menu
        info['game_menu_total_time'] = self.game_menu_total_time
        info['game_menu_open_count_multiplier'] = 1.0 + (self.game_menu_open_count * 0.5)

        # 19. FIN DE QUÊTE (victory/failure lorsque sera implémenté)


        # 20. STATS
        info['total_hits'] = self.hit_count
        info['hit_count'] = self.hit_count
        info['total_damage_dealt'] = self.total_damage_dealt
        info['total_monster_damage'] = self.total_monster_damage
        info['monsters_killed_count'] = self.monsters_killed_count
        info['current_hp'] = current_hp
        info['current_stamina'] = stamina
        info['death_count'] = death_count
        info['total_distance'] = self.total_distance_traveled
        info['camp_total_time'] = self.camp_total_time
        info['monsters_hit_count'] = self.monsters_hit_count
        info['quest_time'] = current_state.get('quest_time')
        info['oxygen_penalty_count'] = self.oxygen_penalty_count

        # S'assurer que les stats monstres ACTUELLES sont toujours présentes
        # (pas seulement quand elles changent)
        info['in_monster_zone'] = self.zone_has_monsters
        info['monsters_present'] = self.zone_has_monsters
        if not hasattr(self, '_last_monster_count'):
            self._last_monster_count = 0
        info['monster_count'] = self._last_monster_count
        info['in_combat'] = info.get('in_combat', False)  # Conserver si déjà défini

        self.last_reward_details = self.reward_breakdown.copy()
        info['reward_breakdown'] = self.reward_breakdown.copy()
        info['reward_breakdown_detailed'] = self.reward_breakdown_detailed.copy()

        # Ajouter explicitement les valeurs CURRENT (pas moyennes)
        info['reward_breakdown_current'] = self.reward_breakdown.copy()
        info['reward_breakdown_detailed_current'] = self.reward_breakdown_detailed.copy()

        # AGRÉGATION FINALE : S'assurer que toutes les catégories principales existent
        # Même si elles sont à 0, elles doivent être présentes dans reward_breakdown
        categories_to_ensure = [
            'survival', 'combat', 'health', 'exploration', 'penalties',
            'zone_change', 'defensive_actions', 'oxygen', 'monster_zone',
            'death', 'damage_taken', 'monster_hit', 'hit', 'camp_penalty',
            'sharpness_penalty', 'menu_penalty', 'other'
        ]

        for category in categories_to_ensure:
            if category not in self.reward_breakdown:
                self.reward_breakdown[category] = 0.0

        # Nettoyer les breakdowns (remplacer None par 0.0)
        info['reward_breakdown'] = {
            k: float(v) if v is not None else 0.0
            for k, v in info['reward_breakdown'].items()
        }
        # Nettoyer les breakdowns detaille (remplacer None par 0.0)
        info['reward_breakdown_detailed'] = {
            k: float(v) if v is not None else 0.0
            for k, v in info['reward_breakdown_detailed'].items()
        }

        # Toujours transmettre les HP des monstres pour le GUI
        for i in range(1, 6):  # Small monsters
            hp_key = f'smonster{i}_hp'
            info[hp_key] = current_state.get(hp_key, 0) or 0

        for i in range(1, 2):  # Large monsters
            hp_key = f'lmonster{i}_hp'
            info[hp_key] = current_state.get(hp_key, 0) or 0

        # Mise à jour death_count pour le prochain step
        self.prev_death_count = death_count

        # Vérifier qu'un cube peut avoir plusieurs marqueurs
        self._debug_marker_check_count += 1  # Incrémenter à chaque appel de calculate()

        if self._debug_marker_check_count % 1000 == 0:  # Tous les 1000 steps
            if hasattr(self, 'exploration_tracker'):
                tracker = self.exploration_tracker

                # Chercher un cube avec plusieurs marqueurs
                found_multi_marker = False
                for zone_id, cubes in tracker.cubes_by_zone.items():
                    for cube in cubes:
                        if len(cube.markers) > 1:
                            marker_names = [m.name for m in cube.markers.keys()]
                            logger.debug(f"Cube multi-marqueurs détecté (step {self._debug_marker_check_count}):")
                            logger.debug(f"Position: ({cube.center_x:.0f}, {cube.center_y:.0f}, {cube.center_z:.0f})")
                            logger.debug(f"Zone: {cube.zone_id}")
                            logger.debug(f"Marqueurs: {', '.join(marker_names)}")
                            found_multi_marker = True
                            break  # Afficher 1 seul exemple
                    if found_multi_marker:
                        break  # Sortir des 2 boucles

        # retourne 'reward' ('info' modifié par référence)
        return reward

    def _check_monster_damage(self, current_state: dict) -> tuple:
        """
        Détection dégâts monstres - Return (reward, damage_dealt)
        """
        reward = 0.0
        total_damage = 0

        # Small monsters
        for i in range(1, 6): # Pour 5 monstres
            key = f'smonster{i}_hp'
            current_hp = current_state.get(key)

            if current_hp is not None and current_hp >= 0:
                prev_hp = self.prev_small_monsters_hp.get(i)

                if prev_hp is not None and prev_hp > current_hp:
                    damage = prev_hp - current_hp

                    # Clipper le damage
                    if damage > 1000:  # Seuil max
                        logger.warning(f"Delta HP monstre aberrant : {damage}")
                        logger.warning(f"Monstre {i}: {prev_hp} → {current_hp}")
                        logger.warning(f"→ Probable reset/corruption, ignoré")
                        # Mettre à jour prev sans reward
                        self.prev_small_monsters_hp[i] = current_hp
                        continue  # Ignorer ce delta

                    # Clipper à 50 max (sécurité supplémentaire)
                    damage = min(damage, 50)
                    total_damage += damage

                    monster_reward = self.REWARD_MONSTER_HIT + (damage * self.REWARD_MONSTER_DAMAGE_MULT)

                    if current_hp == 0 and prev_hp > 0:
                        kill_bonus = self.BONUS_KILL_SMALL_MONSTER
                        monster_reward += kill_bonus
                        self.monsters_killed_count += 1
                        logger.info(f"💀 Petit monstre {i} TUÉ ! Bonus: +{kill_bonus:.1f}")

                    reward += monster_reward

                self.prev_small_monsters_hp[i] = current_hp

        # Large monsters
        for i in range(1, 2):
            key = f'lmonster{i}_hp'
            current_hp = current_state.get(key)

            if current_hp is not None and current_hp >= 0:
                prev_hp = self.prev_large_monsters_hp.get(i)

                if prev_hp is not None and prev_hp > current_hp:
                    damage = prev_hp - current_hp

                    # Clipper le damage
                    if damage > 5000:  # Boss = seuil plus élevé
                        logger.warning(f"⚠️ Delta HP boss aberrant : {damage}")
                        logger.warning(f"Boss {i}: {prev_hp} → {current_hp}")
                        logger.warning(f"→ Probable reset/corruption, ignoré")
                        self.prev_large_monsters_hp[i] = current_hp
                        continue

                    monster_reward = (self.REWARD_MONSTER_HIT * 2) + (damage * self.REWARD_MONSTER_DAMAGE_MULT)

                    if current_hp == 0 and prev_hp > 0:
                        kill_bonus = self.BONUS_KILL_LARGE_MONSTER
                        monster_reward += kill_bonus
                        self.monsters_killed_count += 1
                        logger.info(f"🐉 BOSS {i} VAINCU ! Bonus: +{kill_bonus:.1f}")

                    reward += monster_reward

                self.prev_large_monsters_hp[i] = current_hp

        return reward, total_damage

    def _calculate_death_penalty_reduction(self) -> float:
        """
        Calcule facteur de réduction du malus mort basé sur :
        - Dégâts infligés depuis changement zone (max 10%)
        - Monstres tués (max 20%)
        """
        if self.monster_damage_since_zone_change == 0:
            return 0.0

        # Max 10% de réduction pour dégâts
        damage_reduction = min(self.monster_damage_since_zone_change / 2000.0, 0.10) # 2000 = HP max theorique d'un monstre

        # Max 20% de réduction pour kills
        if self.monsters_killed_count > 0:
            damage_reduction += 0.20

        # Réduction totale max 30%
        return min(damage_reduction, 0.80)


    def get_reward_breakdown_summary(self) -> str:
        """Résumé textuel du breakdown"""
        if not self.last_reward_details:
            return "Aucune reward calculée"

        lines = []
        for category, value in self.last_reward_details.items():
            if abs(value) > 0.01:
                sign = "+" if value > 0 else ""
                lines.append(f"{category:20s}: {sign}{value:+7.2f}")

        return "".join(lines)

    def reset(self):
        """
        Reset pour un nouvel épisode
        """
        self.prev_hp = None
        self.prev_stamina = None
        self.prev_in_combat = False
        self.hp_recovery_given = False  # flag HP recovery
        self.stamina_recovery_given = False  # flag stamina recovery
        self.prev_death_count = 0
        self.prev_position = None
        self.prev_orientation = None
        self.prev_zone = None
        self.prev_sharpness = None
        self.prev_damage_flag = None

        # Reset tous les accumulateurs
        self.hp_recovery_accumulated = 0.0
        self.stamina_recovery_accumulated = 0.0

        # Reset oxygène
        self.prev_oxygen = None
        self.oxygen_low_start_time = None
        self.last_oxygen_penalty_time = None
        self.oxygen_penalty_count = 0

        # On garde les découvertes entre épisodes
        # Mais on reset les compteurs de visites pour éviter malus accumulés
        self.exploration_tracker.reset_episode()

        # Reset présence monstres
        self.zone_has_monsters = False
        self.prev_zone_had_monsters = False
        self.prev_in_combat = False  # Reset état combat précédent
        self.frames_in_monster_zone = 0
        self.left_monster_zone_count = 0

        self.total_damage_dealt = 0
        self.hit_count = 0
        self.consecutive_hits = 0

        self.total_distance_traveled = 0.0
        self.frames_stationary = 0

        # Reset compteur interne
        self._last_monster_count = 0

        # Reset compteur combat log
        self._combat_log_count = 0

        # Reset Tracking immobilité
        self.frames_stationary = 0
        self.idle_start_time = None

        # Reset camp
        self.camp_entry_time = None
        self.camp_penalty_triggered = False
        self.last_camp_penalty_period = 0

        # Reset sortie camp
        self.first_camp_exit = False
        self.just_died = False
        self.camp_exit_after_death = False

        self.last_zone_change_time = None

        # Reset tracking monstres
        self.prev_small_monsters_hp.clear()
        self.prev_large_monsters_hp.clear()
        self.monsters_hit_count = 0
        self.total_monster_damage = 0
        self.monsters_killed_count = 0
        self.monster_damage_since_zone_change = 0

        # Reset timer combat
        self.last_damage_time = 0.0

        # Reset l'exploration tracker (compteurs d'épisode uniquement)
        self.exploration_tracker.reset_episode()

        # S'assurer qu'aucun cube n'a été perdu
        for zone_id in self.exploration_tracker.cubes_by_zone.keys():
            if not self.exploration_tracker.verify_octree_integrity(zone_id):
                logger.warning(f"🚨 Incohérence détectée après reset - Zone {zone_id}")

        # Reset menu
        self.game_menu_entry_time = None
        self.game_menu_total_time = 0.0
        self.last_menu_penalty_time = None
        self.prev_in_menu = False
        self.game_menu_open_count = 0  # Reset compteur menu par épisode

        # Reset breakdown
        self.reward_breakdown = {}
        self.reward_breakdown_detailed = {}
        self.last_reward_details = {}

    def full_reset(self):
        """
        Reset COMPLET incluant les découvertes
        À utiliser quand on veut vraiment tout réinitialiser
        (par exemple, nouveau training run)
        """
        self.reset()

        # Recréer un tracker vierge
        self.exploration_tracker = ExplorationTracker(
            cube_size=self.cube_size,
            max_cubes=self.max_cubes,
            compression_target=0.85
        )

    def get_stats(self) -> dict:
        """Retourne les statistiques"""
        # Récupérer les stats d'exploration
        exploration_stats = self.exploration_tracker.get_stats()

        return {
            'hit_count': self.hit_count,
            'total_damage_dealt': self.total_damage_dealt,
            'total_monster_damage': self.total_monster_damage,
            'monsters_killed_count': self.monsters_killed_count,
            'consecutive_hits': self.consecutive_hits,
            'total_distance': self.total_distance_traveled,
            'camp_total_time': self.camp_total_time,
            'monsters_hit_count': self.monsters_hit_count,
            'oxygen_penalty_count': self.oxygen_penalty_count,
            'game_menu_total_time': self.game_menu_total_time,
            'game_menu_count': self.game_menu_open_count,
            'zones_discovered': exploration_stats['zones_discovered'],
            'areas_explored': exploration_stats['total_cubes'],
            'total_cubes': exploration_stats['total_cubes'],
            'exploration_visits': exploration_stats['total_visits'],
            'left_monster_zone_count': self.left_monster_zone_count,  
            'reward_breakdown': self.last_reward_details.copy(),
            # NE PAS retourner 'episode' ici (conflit SB3)
            'exploration_cubes': exploration_stats.get('exploration_cubes', {}),
        }