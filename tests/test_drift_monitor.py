import numpy as np
from src.drift_monitor import calculate_psi

def test_calculate_psi_small_when_distributions_equal():
    a = np.random.normal(0, 1, 2000)
    b = np.random.normal(0, 1, 2000)
    psi = calculate_psi(a, b, bins=10)
    assert psi < 0.1
