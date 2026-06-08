# Spectra

Spectra is an internal 5G beamforming and RF analysis prototype for on-the-fly engineering workflows. It combines a PyQt6 front end with a Python simulation core to estimate path loss, link budget, OFDM behavior, and beamforming metrics for rapid engineering evaluation.

## Highlights
- 5G-style MU-MIMO / hybrid beamforming simulation path
- Path-loss, link-budget, SINR, modulation, and EVM reporting
- PyQt6 GUI for field and advanced lab views
- Reproducible simulation runs via seed support
- PDF export for engineering reports

## Project layout
- `Core/` — simulation engines for channel models, beamforming, OFDM and link budget
- `GUI/` — PyQt6 interface and controller logic
- `tests/` — regression tests for configuration and simulation reproducibility

## Quick start
1. Create a virtual environment:
   `python -m venv .venv`
2. Activate it and install dependencies:
   `.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. Launch the GUI:
   `.venv\Scripts\python.exe main.py`
4. Run the validation suite:
   `.venv\Scripts\python.exe -m pytest -q`

## Recommended next improvements
- Add real Monte Carlo dashboards and beam pattern plots
- Build a CLI mode for scripted batch runs
- Add CI/CD for automated tests on GitHub
- Add packaging and release notes for easy internal deployment
