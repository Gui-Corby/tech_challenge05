import numpy as np
from src.drift_monitor import calculate_psi

def test_calculate_psi_small_when_distributions_equal():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)

    psi = calculate_psi(a, b, bins=10)
    assert psi < 0.05
