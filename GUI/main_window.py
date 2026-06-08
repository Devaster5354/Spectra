from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QComboBox, QSlider, QFrame,
    QTabWidget, QTextEdit, QStackedWidget,
    QPushButton
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectra RF Engineering Workstation")
        self.resize(1400, 820)

        self._build_ui()

        from GUI.controller import SpectraController
        self.controller = SpectraController(self)

    # ===================================================
    # UI
    # ===================================================

    def _build_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        self.view_stack = QStackedWidget()

        self.param_panel = self._build_param_panel()
        main_layout.addWidget(self.param_panel, 1)
        main_layout.addWidget(self.view_stack, 2)

        self.metric_panel = self._build_metric_panel()
        main_layout.addWidget(self.metric_panel, 1)

        self._build_basic_view()
        self._build_advanced_view()

        self._toggle_mode()

    # ===================================================
    # PARAM PANEL
    # ===================================================

    def _add_slider(self, layout, label_text, slider, unit=""):

        row = QHBoxLayout()
        label = QLabel(label_text)
        value = QLabel(f"{slider.value()}{unit}")

        row.addWidget(label)
        row.addStretch()
        row.addWidget(value)

        layout.addLayout(row)
        layout.addWidget(slider)

        slider.valueChanged.connect(
            lambda val: value.setText(f"{val}{unit}")
        )

    def _build_param_panel(self):

        panel = QFrame()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        layout.addWidget(QLabel("Mode"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Basic (Field)", "Advanced (Lab)"])
        layout.addWidget(self.mode_selector)

        layout.addWidget(QLabel("Scenario"))
        self.scenario_selector = QComboBox()
        self.scenario_selector.addItems(["UMa", "UMi", "InH"])
        layout.addWidget(self.scenario_selector)

        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(10, 1000)
        self.distance_slider.setValue(200)
        self._add_slider(layout, "Distance", self.distance_slider, " m")

        self.tx_power_slider = QSlider(Qt.Orientation.Horizontal)
        self.tx_power_slider.setRange(0, 50)
        self.tx_power_slider.setValue(30)
        self._add_slider(layout, "Tx Power", self.tx_power_slider, " dBm")

        self.nx_slider = QSlider(Qt.Orientation.Horizontal)
        self.nx_slider.setRange(2, 16)
        self.nx_slider.setValue(8)
        self._add_slider(layout, "URA Nx", self.nx_slider)

        self.ny_slider = QSlider(Qt.Orientation.Horizontal)
        self.ny_slider.setRange(2, 16)
        self.ny_slider.setValue(8)
        self._add_slider(layout, "URA Ny", self.ny_slider)

        self.steering_slider = QSlider(Qt.Orientation.Horizontal)
        self.steering_slider.setRange(-60, 60)
        self.steering_slider.setValue(0)
        self._add_slider(layout, "Beam Steering", self.steering_slider, "°")

        # Advanced only
        self.advanced_section = QFrame()
        adv_layout = QVBoxLayout()
        self.advanced_section.setLayout(adv_layout)

        self.monte_slider = QSlider(Qt.Orientation.Horizontal)
        self.monte_slider.setRange(1, 100)
        self.monte_slider.setValue(10)
        self._add_slider(adv_layout, "Monte Carlo Runs", self.monte_slider)

        layout.addWidget(self.advanced_section)

        self.export_button = QPushButton("Export PDF Report")
        layout.addWidget(self.export_button)

        layout.addStretch()

        self.mode_selector.currentIndexChanged.connect(self._toggle_mode)

        return panel

    def _toggle_mode(self):
        if self.mode_selector.currentIndex() == 0:
            self.view_stack.setCurrentIndex(0)
            self.advanced_section.hide()
        else:
            self.view_stack.setCurrentIndex(1)
            self.advanced_section.show()

    # ===================================================
    # BASIC VIEW (Field)
    # ===================================================

    def _build_basic_view(self):

        basic = QWidget()
        layout = QVBoxLayout()
        basic.setLayout(layout)

        self.basic_3d = gl.GLViewWidget()
        self.basic_3d.setCameraPosition(distance=60)
        layout.addWidget(self.basic_3d)

        self.pathloss_curve = pg.PlotWidget(title="Distance vs Path Loss")
        layout.addWidget(self.pathloss_curve)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        layout.addWidget(self.summary_box)

        self.view_stack.addWidget(basic)

    # ===================================================
    # ADVANCED VIEW (Lab)
    # ===================================================

    def _build_advanced_view(self):

        advanced = QTabWidget()

        self.adv_3d = gl.GLViewWidget()
        self.adv_3d.setCameraPosition(distance=60)
        advanced.addTab(self.adv_3d, "3D Radiation")

        self.hist_plot = pg.PlotWidget(title="SINR Histogram")
        advanced.addTab(self.hist_plot, "Histogram")

        self.cdf_plot = pg.PlotWidget(title="SINR CDF")
        advanced.addTab(self.cdf_plot, "CDF")

        self.waterfall_plot = pg.ImageView()
        advanced.addTab(self.waterfall_plot, "OFDM Waterfall")

        self.weights_plot = pg.PlotWidget(title="Beam Weights")
        advanced.addTab(self.weights_plot, "Beam Weights")

        self.amc_plot = pg.PlotWidget(title="AMC per Subcarrier")
        advanced.addTab(self.amc_plot, "AMC")

        self.view_stack.addWidget(advanced)

    # ===================================================
    # METRICS PANEL
    # ===================================================

    def _build_metric_panel(self):

        panel = QFrame()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        self.path_loss_label = QLabel("Path Loss: -")
        self.link_snr_label = QLabel("Link SNR: -")
        self.mod_label = QLabel("Modulation: -")
        self.evm_label = QLabel("EVM: -")

        layout.addWidget(self.path_loss_label)
        layout.addWidget(self.link_snr_label)
        layout.addWidget(self.mod_label)
        layout.addWidget(self.evm_label)

        layout.addStretch()
        return panel
