import sys
from PyQt6.QtWidgets import QApplication
from GUI.main_window import MainWindow
import pyqtgraph as pg


def main():
    app = QApplication(sys.argv)

    # Dark theme base
    app.setStyle("Fusion")

    # PyQtGraph dark config
    pg.setConfigOption('background', '#0D1117')
    pg.setConfigOption('foreground', '#E6EDF3')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
