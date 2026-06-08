"""
Spectra - Power Allocation Engine

Supports:
- Equal power allocation
- Water-filling allocation
- Weighted allocation
"""

import numpy as np


class PowerAllocator:

    def __init__(self, total_power_linear):
        self.total_power = total_power_linear

    # ----------------------------------------------------
    # Equal Power Allocation
    # ----------------------------------------------------
    def equal_power(self, K):
        return np.ones(K) * (self.total_power / K)

    # ----------------------------------------------------
    # Water-Filling Algorithm
    # ----------------------------------------------------
    def water_filling(self, channel_gains, noise_power):

        K = len(channel_gains)

        gains = np.array(channel_gains)
        inverse_snr = noise_power / gains

        sorted_indices = np.argsort(inverse_snr)
        sorted_inv = inverse_snr[sorted_indices]

        power = np.zeros(K)
        remaining_power = self.total_power

        for i in range(K):
            level = (remaining_power +
                     np.sum(sorted_inv[:i+1])) / (i+1)

            if i == K-1 or level <= sorted_inv[i+1]:
                water_level = level
                break

        for i in range(K):
            alloc = water_level - inverse_snr[i]
            power[i] = max(alloc, 0)

        # Normalize to total power
        power = power * (self.total_power / np.sum(power))

        return power

    # ----------------------------------------------------
    # Weighted Power Allocation
    # ----------------------------------------------------
    def weighted_power(self, weights):

        weights = np.array(weights)
        weights = weights / np.sum(weights)

        return weights * self.total_power
