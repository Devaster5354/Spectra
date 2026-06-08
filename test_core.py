"""
Spectra Core Validation Script
Run this before building GUI.
"""

import numpy as np
from Core.system_simulator import SpectraSystemSimulator


def validate_system():

    print("\n===== INITIALIZING SYSTEM =====")
    simulator = SpectraSystemSimulator(
        scenario_type="UMa",
        Nt=32,
        Nr=2,
        K=4,
        carrier_freq=28e9,
        bandwidth=100e6
    )

    print("System initialized.")

    print("\n===== RUNNING SIMULATION =====")
    results = simulator.run_simulation(
        tx_power_dbm=30,
        tx_gain_db=20,
        rx_gain_db=5,
        distance_m=200
    )

    print("\n===== VALIDATING OUTPUTS =====")

    # SINR check
    sinr = results["SINR_dB_per_user"]
    assert len(sinr) == 4, "SINR output size mismatch"
    assert np.all(np.isfinite(sinr)), "SINR contains invalid values"
    print("✔ SINR valid:", sinr)

    # Sum rate
    sum_rate = results["Sum_Rate_bps"]
    assert sum_rate >= 0, "Negative sum rate detected"
    print("✔ Sum Rate:", sum_rate)

    # Link budget SNR
    snr_link = results["Link_SNR_dB"]
    assert np.isfinite(snr_link), "Invalid link SNR"
    print("✔ Link SNR:", snr_link)

    # Modulation
    modulation = results["Selected_Modulation"]
    print("✔ Selected Modulation:", modulation)

    # Path loss sanity
    path_loss = results["Path_Loss_dB"]
    assert path_loss > 0, "Invalid path loss"
    print("✔ Path Loss:", path_loss)

    # OFDM SNR shape
    sub_snr = results["Per_Subcarrier_SNR"]
    assert len(sub_snr) == simulator.ofdm.N, "OFDM subcarrier count mismatch"
    print("✔ OFDM Subcarrier SNR count:", len(sub_snr))

    print("\n===== ALL CORE TESTS PASSED =====")


if __name__ == "__main__":
    validate_system()
