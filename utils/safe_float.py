"""
Fonction utilitaire pour convertir des valeurs en float sécurisé.
Évite les NaN/Inf qui peuvent causer des crashs dans les vecteurs d'observation.

Aucune dépendance externe : utilise uniquement math de la stdlib.
"""

import math


def safe_float(value, default=0.0, min_val=-1e6, max_val=1e6):
    """
    Convertit une valeur en float sécurisé.

    Protections :
    - Remplace None par default
    - Remplace NaN/Inf par default
    - Clamp entre min_val et max_val

    Args:
        value:   Valeur à convertir (int, float, numpy scalar, str, None…)
        default: Valeur de remplacement si la valeur est invalide
        min_val: Borne inférieure autorisée (clamp)
        max_val: Borne supérieure autorisée (clamp)

    Returns:
        float sécurisé dans [min_val, max_val]
    """
    try:
        if value is None:
            return default

        val = float(value)

        # Rejeter NaN et Inf (math.isnan/isinf fonctionnent sur tout float Python)
        if math.isnan(val) or math.isinf(val):
            return default

        # Clamp sans numpy : max/min Python pur, retourne un float natif
        return max(min_val, min(max_val, val))

    except (ValueError, TypeError):
        return default