"""
Spectra - OFDM Engine (Core v2 - Fully Engineered)

Features:
- QAM symbol generation
- Pilot insertion
- IFFT / CP
- Multipath channel
- LS channel estimation
- ZF / MMSE equalization
- EVM computation
"""

import numpy as np


class OFDMSystem:

    def __init__(self,
                 num_subcarriers=256,
                 cp_length=32,
                 bandwidth_hz=20e6,
                 pilot_spacing=8,
                 rng=None):

        self.N = num_subcarriers
        self.cp = cp_length
        self.bandwidth = bandwidth_hz
        self.subcarrier_spacing = bandwidth_hz / num_subcarriers
        self.pilot_spacing = pilot_spacing
        self.rng = rng if rng is not None else np.random.default_rng()

    # ----------------------------------------------------
    # QAM Symbol Generation
    # ----------------------------------------------------
    def generate_symbols(self, modulation_order):

        symbols = self.rng.integers(0, modulation_order, self.N)
        constellation = np.exp(1j * 2*np.pi * symbols / modulation_order)
        return constellation

    # ----------------------------------------------------
    # Pilot Insertion
    # ----------------------------------------------------
    def insert_pilots(self, symbols):

        pilot_value = 1 + 0j
        pilot_indices = np.arange(0, self.N, self.pilot_spacing)

        symbols_with_pilots = symbols.copy()
        symbols_with_pilots[pilot_indices] = pilot_value

        return symbols_with_pilots, pilot_indices

    # ----------------------------------------------------
    # IFFT + CP
    # ----------------------------------------------------
    def apply_ifft(self, freq_symbols):

        time_signal = np.fft.ifft(freq_symbols)
        cp = time_signal[-self.cp:]
        return np.concatenate([cp, time_signal])

    # ----------------------------------------------------
    # Remove CP + FFT
    # ----------------------------------------------------
    def remove_cp_and_fft(self, rx_signal):

        rx_no_cp = rx_signal[self.cp:]
        return np.fft.fft(rx_no_cp)

    # ----------------------------------------------------
    # Multipath Channel
    # ----------------------------------------------------
    def frequency_selective_channel(self, num_taps=8):

        taps = (self.rng.standard_normal(num_taps) +
                1j*self.rng.standard_normal(num_taps)) / np.sqrt(2*num_taps)

        return taps

    def apply_channel(self, tx_signal, taps):

        return np.convolve(tx_signal, taps, mode='same')

    # ----------------------------------------------------
    # LS Channel Estimation
    # ----------------------------------------------------
    def ls_channel_estimate(self,
                            rx_freq,
                            tx_symbols,
                            pilot_indices):

        H_est = np.zeros(self.N, dtype=complex)

        for idx in pilot_indices:
            H_est[idx] = rx_freq[idx] / tx_symbols[idx]

        # Interpolate across subcarriers
        H_est = np.interp(
            np.arange(self.N),
            pilot_indices,
            H_est[pilot_indices]
        )

        return H_est

    # ----------------------------------------------------
    # Equalization
    # ----------------------------------------------------
    def equalize(self,
                 rx_freq,
                 H_est,
                 noise_power,
                 method="ZF"):

        epsilon = 1e-12

        if method == "ZF":
            return rx_freq / (H_est + epsilon)

        elif method == "MMSE":
            return (np.conj(H_est) /
                    (np.abs(H_est)**2 + noise_power + epsilon)) * rx_freq

        else:
            raise ValueError("Unsupported equalization method")

    # ----------------------------------------------------
    # EVM Calculation
    # ----------------------------------------------------
    def compute_evm(self, equalized_symbols, tx_symbols):

        error = equalized_symbols - tx_symbols
        evm = np.sqrt(np.mean(np.abs(error)**2) /
                      np.mean(np.abs(tx_symbols)**2))

        return evm
        # ----------------------------------------------------
    # Per-Subcarrier SNR
    # ----------------------------------------------------
    def per_subcarrier_snr(self, Hk, noise_power):

        signal_power = np.abs(Hk) ** 2
        epsilon = 1e-12

        return signal_power / (noise_power + epsilon)
