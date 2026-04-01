import math

def normalize_stamina(raw_value: int) -> float:
    """Normalise stamina"""
    if raw_value is None:
        return 0.0

    min_stamina = 787032
    max_stamina = 39322200

    normalized = ((raw_value - min_stamina) / (max_stamina - min_stamina)) * 100.0
    return max(0.0, normalized)

def normalize_hp(raw_value: int) -> float:
    """Normalise HP"""
    if raw_value is None:
        return 0.0

    max_hp = 2516608000
    min_hp = 2516582400

    normalized = ((raw_value - min_hp) / (max_hp - min_hp)) * 100.0
    return max(0.0, normalized)

def convert_orientation_to_degrees(orientation_ns: float, orientation_ew: float) -> float:
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
