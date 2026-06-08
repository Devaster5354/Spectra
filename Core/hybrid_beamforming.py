"""
Spectra - Hybrid Beamforming Module (Engineered Version)
Includes:
- ZF Precoding
- MMSE Precoding
- Condition number monitoring
- Numerical stabilization
"""

import numpy as np


class HybridBeamformer:

    def __init__(self, Nt, NRF, Ns, rng=None):
        self.Nt = Nt
        self.NRF = NRF
        self.Ns = Ns
        self.rng = rng if rng is not None else np.random.default_rng()

        if NRF < Ns:
            raise ValueError("Number of RF chains must be >= number of streams")

    # ----------------------------------------------------
    # Analog RF Precoder
    # ----------------------------------------------------
    def random_analog_precoder(self):
        phases = self.rng.uniform(0, 2*np.pi, (self.Nt, self.NRF))
        W_RF = np.exp(1j * phases)
        return W_RF / np.sqrt(self.Nt)

    # ----------------------------------------------------
    # Digital Precoder (ZF / MMSE)
    # ----------------------------------------------------
    def digital_precoder(self,
                         H_eff,
                         noise_power_linear,
                         method="MMSE"):

        cond_number = np.linalg.cond(H_eff)

        # If matrix badly conditioned, force MMSE
        if cond_number > 1e4:
            method = "MMSE"

        if method == "ZF":
            W_BB = np.linalg.pinv(H_eff)

        elif method == "MMSE":
            alpha = noise_power_linear
            Hh = H_eff.conj().T
            W_BB = Hh @ np.linalg.inv(H_eff @ Hh + alpha * np.eye(H_eff.shape[0]))

        else:
            raise ValueError("Unsupported precoding method")

        # Safe normalization
        norm = np.linalg.norm(W_BB, 'fro')
        if norm > 0:
            W_BB = W_BB / norm

        return W_BB, cond_number
