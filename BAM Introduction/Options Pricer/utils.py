import math


def norm_cdf(x):
    """
    Función de distribución acumulativa de la variable normal estándar.
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))