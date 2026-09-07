import sys
import multiprocessing
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from ToFPipeline.ToFPipeline import GlobalConfig
from main_window import MainWindow


# Load global configuration from config.yaml
config_path = Path(__file__).parent / "ToFPipeline/config.yaml"
if config_path.exists():
    GlobalConfig.load(config_path)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    # Use 'spawn' on all platforms so forking never copies Qt's internal state
    # into the worker process (critical on Linux where 'fork' is the default).
    multiprocessing.set_start_method('spawn')
    main()
