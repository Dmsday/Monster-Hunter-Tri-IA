"""
Fonction utilitaire pour convertir des valeurs en float sécurisé
Évite les NaN/Inf qui peuvent causer des crashs
"""

import numpy as np


def safe_float(value, default=0.0, min_val=-1e6, max_val=1e6):
    """
    Convertit une valeur en float sécurisé

    Protections :
    - Remplace None par default
    - Remplace NaN/Inf par default
    - Clamp entre min_val et max_val

    Args:
        value: Valeur à convertir
        default: Valeur par défaut si invalide
        min_val: Valeur minimale autorisée
        max_val: Valeur maximale autorisée

    Returns:
        Float sécurisé
    """
    try:
        if value is None:
            return default

        val = float(value)

        # Vérifier NaN/Inf
        if np.isnan(val) or np.isinf(val):
            return default

        # Clamp dans les limites
        return float(np.clip(val, min_val, max_val))

    except (ValueError, TypeError):
        return default