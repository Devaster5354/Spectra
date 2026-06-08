"""
Spectra - Physical Constants Module
Author: Abhinivesh Sharma
Description: Contains universal constants and RF reference values
"""

import numpy as np

# Speed of light (m/s)
C = 3e8  

# Boltzmann constant (J/K)
K_BOLTZMANN = 1.38064852e-23  

# Standard temperature (Kelvin)
STANDARD_TEMPERATURE = 290  

# Reference impedance (Ohms)
Z_0 = 50  

# Thermal noise density at 290K (dBm/Hz)
NOISE_DENSITY_DBM_PER_HZ = -174  

# Conversion helpers
def db_to_linear(db):
    return 10 ** (db / 10)

def linear_to_db(linear):
    return 10 * np.log10(linear)

def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)

def watts_to_dbm(watts):
    return 10 * np.log10(watts) + 30
