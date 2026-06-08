"""
Spectra - 3GPP Inspired Channel Modeling Engine
Implements Large Scale and Small Scale fading models
based on 3GPP TR 38.901 concepts.
"""

import numpy as np
from .constants import C


class ChannelScenario:
    """
    Defines propagation environment parameters.
    """

    def __init__(self, scenario_type="UMa"):
        self.scenario_type = scenario_type
        self._configure()

    def _configure(self):
        if self.scenario_type == "UMa":  # Urban Macro
            self.pl_exponent = 3.5
            self.shadow_std = 6
            self.cluster_count = 12
        elif self.scenario_type == "UMi":  # Urban Micro
            self.pl_exponent = 3.0
            self.shadow_std = 4
            self.cluster_count = 10
        elif self.scenario_type == "InH":  # Indoor
            self.pl_exponent = 2.0
            self.shadow_std = 3
            self.cluster_count = 8
        else:
            raise ValueError("Unsupported scenario type")


class LargeScaleFading:
    """
    Handles path loss and shadowing
    """

    def __init__(self, scenario: ChannelScenario, rng=None):
        self.scenario = scenario
        self.rng = rng if rng is not None else np.random.default_rng()

    def path_loss(self, distance_m, frequency_hz):
        """
        Log-distance path loss model
        """
        d0 = 1  # reference distance (1m)
        lambda_ = C / frequency_hz

        fspl_d0 = 20 * np.log10(4 * np.pi * d0 / lambda_)
        pl = fspl_d0 + 10 * self.scenario.pl_exponent * np.log10(distance_m / d0)

        shadowing = self.rng.normal(0, self.scenario.shadow_std)

        return pl + shadowing


class SmallScaleFading:
    """
    Cluster-based stochastic multipath fading
    with exponential Power Delay Profile (PDP).
    """

    def __init__(self,
                 scenario: ChannelScenario,
                 rms_delay_spread=100e-9,   # 100 ns typical urban
                 num_taps=12,
                 rng=None):

        self.scenario = scenario
        self.rng = rng if rng is not None else np.random.default_rng()
        self.rms_delay_spread = rms_delay_spread
        self.num_taps = num_taps

    # ----------------------------------------------------
    # Generate Exponential Power Delay Profile
    # ----------------------------------------------------
    def _power_delay_profile(self):

        delays = np.linspace(0,
                             5 * self.rms_delay_spread,
                             self.num_taps)

        power = np.exp(-delays / self.rms_delay_spread)

        # Normalize total power
        power = power / np.sum(power)

        return delays, power

    # ----------------------------------------------------
    # Generate Multipath Taps
    # ----------------------------------------------------
    def generate_taps(self,
                      velocity_mps,
                      carrier_freq_hz,
                      time_s):

        delays, power = self._power_delay_profile()

        fd_max = velocity_mps * carrier_freq_hz / C

        taps = np.zeros(self.num_taps, dtype=complex)

        for i in range(self.num_taps):

            phase = self.rng.uniform(0, 2*np.pi)

            doppler = fd_max * np.cos(self.rng.uniform(0, 2*np.pi))

            fading = (self.rng.standard_normal() +
                      1j * self.rng.standard_normal()) / np.sqrt(2)

            taps[i] = np.sqrt(power[i]) * fading * \
                      np.exp(1j * (phase + 2*np.pi*doppler*time_s))

        return taps

    # ----------------------------------------------------
    # Cluster Channel Approximation (Collapsed)
    # ----------------------------------------------------
    def cluster_channel(self,
                        velocity_mps,
                        carrier_freq_hz,
                        time_s):

        taps = self.generate_taps(
            velocity_mps,
            carrier_freq_hz,
            time_s
        )

        # Collapse taps into single equivalent flat fading coefficient
        return np.sum(taps)



class MIMOChannel:
    """
    MIMO Channel Matrix Generator
    Supports optional spatial correlation modeling.
    """

    def __init__(self,
                 tx_antennas,
                 rx_antennas,
                 scenario: ChannelScenario,
                 spatial_corr=True,
                 tx_corr_factor=0.5,
                 rx_corr_factor=0.5,
                 rng=None):

        self.Nt = tx_antennas
        self.Nr = rx_antennas
        self.scenario = scenario
        self.rng = rng if rng is not None else np.random.default_rng()
        self.small_scale = SmallScaleFading(scenario, rng=self.rng)

        self.spatial_corr = spatial_corr
        self.tx_corr_factor = tx_corr_factor
        self.rx_corr_factor = rx_corr_factor

    # ---------------------------------------------
    # Exponential Correlation Model
    # ---------------------------------------------
    def _correlation_matrix(self, N, rho):
        R = np.zeros((N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                R[i, j] = rho ** abs(i - j)
        return R

    # ---------------------------------------------
    # Matrix Square Root (Stable)
    # ---------------------------------------------
    def _matrix_sqrt(self, R):
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

    # ---------------------------------------------
    # Channel Generation
    # ---------------------------------------------
    def generate(self,
                 velocity_mps,
                 carrier_freq_hz,
                 time_s):

        # i.i.d small-scale fading matrix
        H = np.zeros((self.Nr, self.Nt), dtype=complex)

        for r in range(self.Nr):
            for t in range(self.Nt):
                H[r, t] = self.small_scale.cluster_channel(
                    velocity_mps,
                    carrier_freq_hz,
                    time_s
                )

        if not self.spatial_corr:
            return H

        # Apply spatial correlation
        Rt = self._correlation_matrix(self.Nt, self.tx_corr_factor)
        Rr = self._correlation_matrix(self.Nr, self.rx_corr_factor)

        Rt_sqrt = self._matrix_sqrt(Rt)
        Rr_sqrt = self._matrix_sqrt(Rr)

        H_corr = Rr_sqrt @ H @ Rt_sqrt

        return H_corr
