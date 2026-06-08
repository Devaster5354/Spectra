import numpy as np

from Core.config_utils import load_config, save_config
from Core.system_simulator import SpectraSystemSimulator


def test_load_config_reads_project_defaults(tmp_path):
    config_path = tmp_path / "spectra_config.json"
    config_path.write_text(
        '{"distance": 250, "tx_power": 30, "Nx": 8, "Ny": 8, "steering": 10}',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["distance"] == 250
    assert config["tx_power"] == 30
    assert config["Nx"] == 8
    assert config["Ny"] == 8
    assert config["steering"] == 10


def test_simulation_is_reproducible_with_seed():
    sim_a = SpectraSystemSimulator(scenario_type="UMa", Nt=8, Nr=2, K=4, seed=7)
    sim_b = SpectraSystemSimulator(scenario_type="UMa", Nt=8, Nr=2, K=4, seed=7)

    result_a = sim_a.run_simulation(tx_power_dbm=30, tx_gain_db=20, rx_gain_db=5, distance_m=200)
    result_b = sim_b.run_simulation(tx_power_dbm=30, tx_gain_db=20, rx_gain_db=5, distance_m=200)

    assert np.allclose(result_a["Path_Loss_dB"], result_b["Path_Loss_dB"])
    assert np.allclose(result_a["SINR_dB_per_user"], result_b["SINR_dB_per_user"])
    assert result_a["Selected_Modulation"] == result_b["Selected_Modulation"]


def test_save_config_writes_expected_shape(tmp_path):
    config_path = tmp_path / "saved_config.json"

    save_config(config_path, {"distance": 300, "tx_power": 35})

    loaded = load_config(config_path)
    assert loaded["distance"] == 300
    assert loaded["tx_power"] == 35
