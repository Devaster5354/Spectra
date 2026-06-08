"""
Spectra - Advanced Link Budget Engine
Includes RF losses, noise figure, modulation-aware SNR thresholds,
and coverage estimation.
"""

import numpy as np
from .constants import (
    db_to_linear,
    linear_to_db,
    dbm_to_watts,
    watts_to_dbm,
    NOISE_DENSITY_DBM_PER_HZ
)


class LinkBudget:
    def __init__(self,
                 tx_power_dbm,
                 tx_gain_db,
                 rx_gain_db,
                 frequency_hz,
                 bandwidth_hz,
                 noise_figure_db=5,
                 implementation_margin_db=2,
                 cable_loss_db=1):

        self.tx_power_dbm = tx_power_dbm
        self.tx_gain_db = tx_gain_db
        self.rx_gain_db = rx_gain_db
        self.frequency_hz = frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self.noise_figure_db = noise_figure_db
        self.implementation_margin_db = implementation_margin_db
        self.cable_loss_db = cable_loss_db

        self.modulation_thresholds = {
            "QPSK": 5,
            "16QAM": 10,
            "64QAM": 17,
            "256QAM": 23
        }

        self.spectral_efficiency = {
            "QPSK": 2,
            "16QAM": 4,
            "64QAM": 6,
            "256QAM": 8
        }

    def received_power(self, path_loss_db):
        """
        Computes received power in dBm
        """
        pr = (self.tx_power_dbm +
              self.tx_gain_db +
              self.rx_gain_db -
              path_loss_db -
              self.cable_loss_db)

        return pr

    def noise_power_dbm(self):
        """
        Thermal noise + noise figure
        """
        noise_dbm = NOISE_DENSITY_DBM_PER_HZ + \
                    10 * np.log10(self.bandwidth_hz)

        return noise_dbm + self.noise_figure_db

    def snr_db(self, received_power_dbm):
        """
        Compute SNR in dB
        """
        noise_dbm = self.noise_power_dbm()
        return received_power_dbm - noise_dbm - self.implementation_margin_db

    def select_modulation(self, snr_db):
        """
        Adaptive modulation selection
        """
        selected = "Out of Range"

        for mod, threshold in sorted(
                self.modulation_thresholds.items(),
                key=lambda x: x[1]):

            if snr_db >= threshold:
                selected = mod

        return selected

    def achievable_rate(self, snr_db):
        """
        Shannon capacity estimate
        """
        snr_linear = db_to_linear(snr_db)
        return self.bandwidth_hz * np.log2(1 + snr_linear)

    def coverage_radius(self, max_path_loss_db, scenario_exponent):
        """
        Estimate coverage distance based on allowable path loss
        using log-distance model.
        """

        d0 = 1
        pl0 = 32.44 + 20*np.log10(self.frequency_hz/1e6)

        d = d0 * 10 ** ((max_path_loss_db - pl0) /
                        (10 * scenario_exponent))

        return d
