"""
Spectra - Multi-User Massive MIMO SINR Engine
Supports interference-aware SINR computation and system throughput.
"""

import numpy as np
from .constants import db_to_linear, linear_to_db


class MassiveMIMOSINR:
    def __init__(self, bandwidth_hz, noise_figure_db=5, temperature=290):
        self.bandwidth = bandwidth_hz
        self.noise_figure_db = noise_figure_db
        self.temperature = temperature

    def thermal_noise_power_linear(self):
        """
        Returns thermal noise power in linear Watts
        """
        k = 1.38e-23
        noise = k * self.temperature * self.bandwidth
        nf_linear = db_to_linear(self.noise_figure_db)
        return noise * nf_linear

    def compute_sinr(self, H_list, W):
        """
        H_list : list of user channel matrices [H1, H2, ..., HK]
                 each Hi shape (Nr x Nt)
        W      : precoding matrix (Nt x K)
        """

        K = len(H_list)
        noise_power = self.thermal_noise_power_linear()

        sinr_list = []

        for k in range(K):
            Hk = H_list[k]

            signal = np.linalg.norm(Hk @ W[:, k])**2

            interference = 0
            for j in range(K):
                if j != k:
                    interference += np.linalg.norm(Hk @ W[:, j])**2

            epsilon = 1e-12  # Small constant to avoid division by zero
            sinr = signal / (interference + noise_power + epsilon)
            sinr_list.append(sinr)

        return np.array(sinr_list)

    def sum_rate(self, sinr_linear):
        """
        Shannon capacity formula
        """
        return np.sum(np.log2(1 + sinr_linear))

    def sinr_db(self, sinr_linear):
        return linear_to_db(sinr_linear)
