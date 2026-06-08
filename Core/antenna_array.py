"""
Spectra - Antenna Array Module
Supports ULA and UPA modeling with beam steering.
"""

import numpy as np
from .constants import C


class UniformLinearArray:
    def __init__(self, num_elements, spacing, frequency):
        """
        num_elements : Number of antenna elements
        spacing      : Element spacing (meters)
        frequency    : Carrier frequency (Hz)
        """
        self.N = num_elements
        self.d = spacing
        self.f = frequency
        self.lambda_ = C / frequency
        self.k = 2 * np.pi / self.lambda_

    def steering_vector(self, theta_deg):
        theta = np.radians(theta_deg)
        n = np.arange(self.N)
        return np.exp(1j * n * self.k * self.d * np.cos(theta))

    def array_factor(self, theta_scan_deg, steering_angle_deg=0):
        theta = np.radians(theta_scan_deg)
        steering_vec = self.steering_vector(steering_angle_deg)

        af = np.zeros_like(theta, dtype=complex)
        for idx, t in enumerate(theta):
            response = np.exp(1j * np.arange(self.N) *
                              self.k * self.d * np.cos(t))
            af[idx] = np.dot(response, steering_vec.conj())

        return np.abs(af) / self.N

    def estimate_beamwidth(self):
        """
        Approximate Half Power Beamwidth (HPBW)
        """
        return 2 * (self.lambda_ / (self.N * self.d))
