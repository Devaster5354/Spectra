import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from Core.system_simulator import SpectraSystemSimulator


# =========================================================
# BACKGROUND SIMULATION WORKER
# =========================================================

class SimulationWorker(QObject):

    finished = pyqtSignal(dict)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):

        scenario = self.params["scenario"]
        Nt = self.params["Nt"]
        tx_power = self.params["tx_power"]
        distance = self.params["distance"]
        monte = self.params["monte"]

        sinr_samples = []
        last_result = None

        simulator = SpectraSystemSimulator(
            scenario_type=scenario,
            Nt=Nt,
            Nr=2,
            K=4,
            carrier_freq=28e9,
            bandwidth=100e6
        )

        for _ in range(monte):
            last_result = simulator.run_simulation(
                tx_power_dbm=tx_power,
                distance_m=distance
            )
            sinr_samples.extend(last_result["SINR_dB_per_user"])

        self.finished.emit({
            "core": last_result,
            "sinr_samples": np.array(sinr_samples)
        })


# =========================================================
# CONTROLLER
# =========================================================

class SpectraController:

    def __init__(self, main_window):

        self.main = main_window

        self.thread = None
        self.worker = None

        self.mesh_cached = False
        self.cached_Nt = None
        self.cached_scenario = None
        self.last_bundle = None

        self._connect()
        self.trigger_simulation()

    # =====================================================
    # CONNECT
    # =====================================================

    def _connect(self):

        controls = [
            self.main.distance_slider,
            self.main.tx_power_slider,
            self.main.nx_slider,
            self.main.ny_slider,
            self.main.steering_slider,
            self.main.scenario_selector,
            self.main.mode_selector,
            self.main.monte_slider
        ]

        for c in controls:
            if hasattr(c, "valueChanged"):
                c.valueChanged.connect(self.trigger_simulation)
            else:
                c.currentIndexChanged.connect(self.trigger_simulation)

        self.main.export_button.clicked.connect(self.export_pdf)

    # =====================================================
    # TRIGGER THREAD
    # =====================================================

    def trigger_simulation(self):

        scenario = self.main.scenario_selector.currentText()
        distance = self.main.distance_slider.value()
        tx_power = self.main.tx_power_slider.value()
        Nx = self.main.nx_slider.value()
        Ny = self.main.ny_slider.value()
        monte = self.main.monte_slider.value()

        Nt = max(Nx * Ny, 4)

        params = {
            "scenario": scenario,
            "Nt": Nt,
            "tx_power": tx_power,
            "distance": distance,
            "monte": monte
        }

        self.thread = QThread()
        self.worker = SimulationWorker(params)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_simulation_complete)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    # =====================================================
    # THREAD COMPLETE
    # =====================================================

    def on_simulation_complete(self, bundle):

        self.last_bundle = bundle
        res = bundle["core"]
        sinr = bundle["sinr_samples"]

        self._update_metrics(res)

        if self.main.mode_selector.currentIndex() == 0:
            self._render_field_mode(res)
        else:
            self._render_advanced_mode(res, sinr)

    # =====================================================
    # METRICS
    # =====================================================

    def _update_metrics(self, res):

        self.main.path_loss_label.setText(
            f"Path Loss: {res['Path_Loss_dB']:.2f} dB"
        )
        self.main.link_snr_label.setText(
            f"Link SNR: {res['Link_SNR_dB']:.2f} dB"
        )
        self.main.mod_label.setText(
            f"Modulation: {res['Selected_Modulation']}"
        )
        self.main.evm_label.setText(
            f"EVM: {res['EVM']:.4f}"
        )

    # =====================================================
    # FIELD MODE
    # =====================================================

    def _render_field_mode(self, res):

        view = self.main.basic_3d
        view.clear()

        bs = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]]),
            size=20,
            color=(1, 1, 0, 1)
        )
        view.addItem(bs)

        angle = np.deg2rad(self.main.steering_slider.value())
        beam = np.array([[0, 0, 0],
                         [40*np.cos(angle), 40*np.sin(angle), 0]])

        beam_line = gl.GLLinePlotItem(pos=beam, width=3, color=(0,1,1,1))
        view.addItem(beam_line)

        self.main.pathloss_curve.clear()
        d_vals = np.linspace(10, 1000, 100)
        pl = 32.4 + 20*np.log10(28) + 20*np.log10(d_vals/1000)
        self.main.pathloss_curve.plot(d_vals, pl)

        summary = f"""
Field Deployment Mode

Distance: {self.main.distance_slider.value()} m
Path Loss: {res['Path_Loss_dB']:.1f} dB
Link SNR: {res['Link_SNR_dB']:.1f} dB
Modulation: {res['Selected_Modulation']}
Throughput: {res['Sum_Rate_bps']/1e6:.2f} Mbps
"""
        self.main.summary_box.setText(summary)

    # =====================================================
    # ADVANCED MODE
    # =====================================================

    def _render_advanced_mode(self, res, sinr):

        self._render_histogram(sinr)
        self._render_cdf(sinr)
        self._render_waterfall(res)
        self._render_beam_weights()
        self._render_amc(res)
        self._render_radiation_mesh()

    # =====================================================
    # HISTOGRAM
    # =====================================================

    def _render_histogram(self, sinr):

        self.main.hist_plot.clear()
        y, x = np.histogram(sinr, bins=20)
        bg = pg.BarGraphItem(x=x[:-1], height=y, width=1)
        self.main.hist_plot.addItem(bg)

    # =====================================================
    # CDF
    # =====================================================

    def _render_cdf(self, sinr):

        self.main.cdf_plot.clear()
        sorted_sinr = np.sort(sinr)
        cdf = np.arange(len(sorted_sinr)) / len(sorted_sinr)
        self.main.cdf_plot.plot(sorted_sinr, cdf)

    # =====================================================
    # WATERFALL
    # =====================================================

    def _render_waterfall(self, res):

        snr = 10*np.log10(res["Per_Subcarrier_SNR"] + 1e-12)
        waterfall = np.tile(snr, (50, 1))
        self.main.waterfall_plot.setImage(waterfall)

    # =====================================================
    # BEAM WEIGHTS
    # =====================================================

    def _render_beam_weights(self):

        Nx = self.main.nx_slider.value()
        Ny = self.main.ny_slider.value()
        steering = self.main.steering_slider.value()

        self.main.weights_plot.clear()

        Nt = Nx * Ny
        angle = np.deg2rad(steering)

        weights = np.exp(
            1j * 2*np.pi*0.5*np.arange(Nt)*np.sin(angle)
        )

        self.main.weights_plot.plot(np.abs(weights), pen='g')
        self.main.weights_plot.plot(np.angle(weights), pen='r')

    # =====================================================
    # AMC
    # =====================================================

    def _render_amc(self, res):

        snr = 10*np.log10(res["Per_Subcarrier_SNR"] + 1e-12)

        amc = np.zeros_like(snr)
        amc[snr < 0] = 0
        amc[(snr >= 0) & (snr < 5)] = 1
        amc[(snr >= 5) & (snr < 10)] = 2
        amc[(snr >= 10) & (snr < 15)] = 4
        amc[snr >= 15] = 6

        self.main.amc_plot.clear()
        self.main.amc_plot.plot(amc)

    # =====================================================
    # RADIATION MESH (SAFE)
    # =====================================================

    def _render_radiation_mesh(self):

        view = self.main.adv_3d
        view.clear()

        theta = np.linspace(0, np.pi, 40)
        phi = np.linspace(0, 2*np.pi, 40)

        TH, PH = np.meshgrid(theta, phi)

        AF = np.abs(np.sin(TH))
        R = AF * 10

        X = R * np.sin(TH) * np.cos(PH)
        Y = R * np.sin(TH) * np.sin(PH)
        Z = R * np.cos(TH)

        verts = np.dstack((X, Y, Z)).reshape(-1, 3)

        rows, cols = X.shape
        faces = []

        for i in range(rows - 1):
            for j in range(cols - 1):
                idx = i * cols + j
                faces.append([idx, idx + 1, idx + cols])
                faces.append([idx + 1, idx + cols + 1, idx + cols])

        mesh = gl.GLMeshItem(
            vertexes=verts,
            faces=np.array(faces),
            smooth=False,
            computeNormals=False,
            shader='balloon'
        )

        mesh.setColor((0.2, 0.8, 1.0, 0.6))
        view.addItem(mesh)

    # =====================================================
    # PDF EXPORT
    # =====================================================

    def export_pdf(self):
        if self.last_bundle is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self.main,
            "Export PDF Report",
            "spectra_report.pdf",
            "PDF Files (*.pdf)"
        )

        if not file_name:
            return

        report = canvas.Canvas(file_name, pagesize=letter)
        report.setTitle("Spectra Beamforming Report")
        report.setFont("Helvetica-Bold", 16)
        report.drawString(50, 770, "Spectra Beamforming Engineer Report")

        report.setFont("Helvetica", 11)
        res = self.last_bundle["core"]

        lines = [
            f"Scenario: {self.main.scenario_selector.currentText()}",
            f"Distance: {self.main.distance_slider.value()} m",
            f"Tx Power: {self.main.tx_power_slider.value()} dBm",
            f"Path Loss: {res['Path_Loss_dB']:.2f} dB",
            f"Link SNR: {res['Link_SNR_dB']:.2f} dB",
            f"Selected Modulation: {res['Selected_Modulation']}",
            f"EVM: {res['EVM']:.4f}",
            f"Sum Rate: {res['Sum_Rate_bps'] / 1e6:.2f} Mbps",
            f"Precoder Condition Number: {res['Precoder_Condition_Number']:.2f}",
        ]

        y = 730
        for line in lines:
            y -= 18
            report.drawString(60, y, line)

        report.drawString(50, 120, "Generated by Spectra internal beamforming analysis tool.")
        report.save()

