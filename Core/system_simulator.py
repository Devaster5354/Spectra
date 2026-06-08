"""
Spectra - Unified System Simulation Engine (Core v6 - Fully Physically Coherent)

Features:
- 3GPP-inspired spatially correlated channel with PDP
- Large-scale + small-scale fading unified
- 1 stream per user MU-MIMO
- Hybrid Beamforming (ZF/MMSE)
- Physical transmit power scaling
- Proper path-loss scaling before precoding
- Power allocation
- OFDM simulation
- Link budget consistency
- Diagnostics
"""

import numpy as np

from .channel_models import ChannelScenario, MIMOChannel, LargeScaleFading
from .hybrid_beamforming import HybridBeamformer
from .sinr_engine import MassiveMIMOSINR
from .ofdm_engine import OFDMSystem
from .link_budget import LinkBudget
from .power_allocation import PowerAllocator
from .constants import dbm_to_watts


class SpectraSystemSimulator:

    def __init__(self,
                 scenario_type="UMa",
                 Nt=64,
                 Nr=2,
                 K=4,
                 carrier_freq=28e9,
                 bandwidth=100e6,
                 seed=None):

        self.rng = np.random.default_rng(seed)

        if Nt < K:
            raise ValueError("Nt must be >= K for MU-MIMO.")

        self.scenario = ChannelScenario(scenario_type)
        self.Nt = Nt
        self.Nr = Nr
        self.K = K
        self.fc = carrier_freq
        self.bandwidth = bandwidth

        self.channel = MIMOChannel(Nt, Nr, self.scenario, rng=self.rng)
        self.sinr_engine = MassiveMIMOSINR(bandwidth)

        self.ofdm = OFDMSystem(
            num_subcarriers=512,
            cp_length=64,
            bandwidth_hz=bandwidth,
            rng=self.rng
        )

        nrf_chains = max(K, min(8, Nt // 4))

        self.hybrid = HybridBeamformer(
            Nt=Nt,
            NRF=nrf_chains,
            Ns=K,
            rng=self.rng
        )

    # ----------------------------------------------------
    # Channel Generation
    # ----------------------------------------------------
    def generate_channels(self, velocity=10, time_s=0.001):
        return [
            self.channel.generate(velocity, self.fc, time_s)
            for _ in range(self.K)
        ]

    # ----------------------------------------------------
    # Collapse Nr x Nt → 1 x Nt effective channel
    # ----------------------------------------------------
    def _effective_channel(self, Hk):
        U, S, Vh = np.linalg.svd(Hk)
        u_max = U[:, 0]
        return u_max.conj().T @ Hk

    # ----------------------------------------------------
    # Main Simulation
    # ----------------------------------------------------
    def run_simulation(self,
                       tx_power_dbm=30,
                       tx_gain_db=20,
                       rx_gain_db=5,
                       distance_m=200,
                       precoding_method="MMSE"):

        distance_m = max(distance_m, 1.0)

        # --------------------------------------------
        # 1️⃣ Generate Channels
        # --------------------------------------------
        H_list = self.generate_channels()

        H_eff_users = [
            self._effective_channel(Hk)
            for Hk in H_list
        ]

        H_mu = np.vstack(H_eff_users)  # (K x Nt)

        # --------------------------------------------
        # 2️⃣ Large-Scale Path Loss (APPLY BEFORE PRECODING)
        # --------------------------------------------
        lsf = LargeScaleFading(self.scenario, rng=self.rng)
        path_loss = lsf.path_loss(distance_m, self.fc)

        path_gain_linear = 10 ** (-path_loss / 10)
        H_mu = np.sqrt(path_gain_linear) * H_mu

        # --------------------------------------------
        # 3️⃣ Hybrid Precoding
        # --------------------------------------------
        W_RF = self.hybrid.random_analog_precoder()

        H_eff = H_mu @ W_RF

        noise_linear = self.sinr_engine.thermal_noise_power_linear()

        W_BB, cond_number = self.hybrid.digital_precoder(
            H_eff,
            noise_linear,
            method=precoding_method
        )

        W_full = W_RF @ W_BB
        W = W_full[:, :self.K]

        # --------------------------------------------
        # 4️⃣ Physical Transmit Power Allocation
        # --------------------------------------------
        total_power_linear = dbm_to_watts(tx_power_dbm)
        allocator = PowerAllocator(total_power_linear)

        channel_gains = [
            np.abs(H_mu[k, :] @ W[:, k])**2 + 1e-12
            for k in range(self.K)
        ]

        power_alloc = allocator.equal_power(self.K)

        for k in range(self.K):
            W[:, k] *= np.sqrt(power_alloc[k])

        # --------------------------------------------
        # 5️⃣ SINR
        # --------------------------------------------
        sinr_linear = self.sinr_engine.compute_sinr(
            [H_mu[k:k+1, :] for k in range(self.K)],
            W
        )

        sinr_db = self.sinr_engine.sinr_db(sinr_linear)
        sum_rate = self.sinr_engine.sum_rate(sinr_linear)

        # --------------------------------------------
        # 6️⃣ OFDM (small-scale only)
        # --------------------------------------------
                # 6️⃣ OFDM Simulation (Realistic)

        tx_symbols = self.ofdm.generate_symbols(modulation_order=64)

        tx_symbols, pilot_indices = self.ofdm.insert_pilots(tx_symbols)

        tx_time_signal = self.ofdm.apply_ifft(tx_symbols)

        taps = self.ofdm.frequency_selective_channel()

        rx_signal = self.ofdm.apply_channel(tx_time_signal, taps)

        rx_freq = self.ofdm.remove_cp_and_fft(rx_signal)

        H_est = self.ofdm.ls_channel_estimate(
            rx_freq,
            tx_symbols,
            pilot_indices
        )

        equalized = self.ofdm.equalize(
            rx_freq,
            H_est,
            noise_linear,
            method="MMSE"
        )

        evm = self.ofdm.compute_evm(equalized, tx_symbols)

        Hk = np.fft.fft(taps, self.ofdm.N)

        per_subcarrier_snr = self.ofdm.per_subcarrier_snr(
            Hk,
            noise_linear
        )


        # --------------------------------------------
        # 7️⃣ Link Budget
        # --------------------------------------------
        link = LinkBudget(
            tx_power_dbm,
            tx_gain_db,
            rx_gain_db,
            self.fc,
            self.bandwidth
        )

        pr_dbm = link.received_power(path_loss)
        snr_link_db = link.snr_db(pr_dbm)
        modulation = link.select_modulation(snr_link_db)
        achievable_rate = link.achievable_rate(snr_link_db)

        noise_dbm = 10 * np.log10(noise_linear) + 30

        return {
            "SINR_dB_per_user": sinr_db,
            "Sum_Rate_bps": sum_rate * self.bandwidth,
            "Link_SNR_dB": snr_link_db,
            "Selected_Modulation": modulation,
            "Achievable_Rate_bps": achievable_rate,
            "Path_Loss_dB": path_loss,
            "Precoder_Condition_Number": cond_number,
            "Noise_Power_dBm": noise_dbm,
            "Per_Subcarrier_SNR": per_subcarrier_snr,
            "EVM": evm,
        }
