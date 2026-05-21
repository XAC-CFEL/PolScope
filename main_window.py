import time
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from threading import Thread
from concurrent.futures import ProcessPoolExecutor
from typing import List
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QPushButton, QSpinBox,
                              QDoubleSpinBox, QGroupBox, QGridLayout, QFileDialog,
                              QTextEdit, QCheckBox, QScrollArea, QTabWidget,
                              QTableWidget, QTableWidgetItem, QHeaderView,
                              QComboBox, QRadioButton, QButtonGroup, QMessageBox,
                              QDialog, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont

from ToFPipeline.ToFPipeline import GlobalConfig, Calibrate

from models import PlotData
from data_ingestion import CircularBuffer, DataStreamSimulator, DoocspieStream
from processing import process_detector_chunk, PerformanceMonitor, PlotPreparationWorker
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from plotting import FastMplCanvas, PolarPlotCanvas, AngularHeatmapCanvas, SingleDetectorCanvas


class SnapshotManagerDialog(QDialog):
    """Floating dialog for managing reference snapshots across all plot canvases."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Snapshot Manager")
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(300, 240)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Snapshots — check to show, click to select:"))

        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        del_btn = QPushButton("Delete Selected")
        del_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(del_btn)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

    def refresh(self, snapshot_labels):
        """Rebuild the list from [(label, visible), ...] pairs."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i, (label, visible) in enumerate(snapshot_labels):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def _on_item_changed(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        visible = item.checkState() == Qt.CheckState.Checked
        if self.parent() is not None:
            self.parent().set_snapshot_visible(idx, visible)

    def _delete_selected(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        idx = items[0].data(Qt.ItemDataRole.UserRole)
        if self.parent() is not None:
            self.parent().remove_snapshot(idx)

    def _clear_all(self):
        if self.parent() is not None:
            self.parent().clear_all_snapshots()


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-time ToF Data Processing")
        self.setGeometry(100, 100, 1400, 900)

        # State variables
        self.running = False
        self.data_simulator = None
        self.circular_buffer = None
        self.performance_monitor = PerformanceMonitor()
        self.n_detectors = 16
        self.enabled_detectors = set(range(16))
        self.detector_checkboxes = []

        # Process pool for parallel processing (bypasses GIL)
        self.process_pool = None
        self.processing_futures = []

        # Plot worker (thread is fine for I/O-bound plot prep)
        self.plot_worker = None
        self.plot_thread = None

        # Processing config
        self.processing_config = {}

        # Pipeline stages
        self.stage_load = None
        self.stage_process = None
        self.stage_plot = None

        # Intensity calibration coefficients (int det_id -> float coeff)
        self.calib_coefficients = {}

        # Snapshot manager dialog (created on demand)
        self.snapshot_manager = None

        # Setup UI
        self.setup_ui()

        # Timers
        self.main_timer = QTimer()
        self.main_timer.timeout.connect(self.main_loop_iteration)

        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance_display)
        self.perf_timer.start(500)

    def setup_ui(self):
        """Setup the user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel - Controls (wrapped in a scroll area so it doesn't get cut off)
        left_panel = self.create_control_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll, stretch=1)

        # Right panel - Tabbed view (Plots + Results)
        self.tab_widget = QTabWidget()

        # Tab 1: Plots
        self.canvas = FastMplCanvas(self, width=10, height=8, dpi=100, n_detectors=16)
        self.tab_widget.addTab(self.canvas, "Plots")

        # Tab 2: Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            "detector", "trainId", "pulseId", "peakNo",
            "pos", "height", "width left", "width right", "fwhm area"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.tab_widget.addTab(self.results_table, "Results")

        # Tab 3: Polarization Plot
        self.polar_canvas = PolarPlotCanvas(self, width=6, height=6, dpi=100)
        self.tab_widget.addTab(self.polar_canvas, "Polarization")

        # Tab 3: Angular Heatmap
        self.heatmap_canvas = AngularHeatmapCanvas(self, width=7, height=7, dpi=100)
        self.tab_widget.addTab(self.heatmap_canvas, "Angular Heatmap")

        # Tab 4: Single Detector (interactive zoom/pan)
        self.single_det_widget = QWidget()
        single_det_layout = QVBoxLayout(self.single_det_widget)
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Detector:"))
        self.single_det_combo = QComboBox()
        for i in range(16):
            self.single_det_combo.addItem(f"Det {i}")
        self.single_det_combo.currentIndexChanged.connect(self.on_single_det_changed)
        selector_layout.addWidget(self.single_det_combo)
        selector_layout.addStretch()
        single_det_layout.addLayout(selector_layout)
        self.single_det_canvas = SingleDetectorCanvas(self, width=8, height=4, dpi=100)
        self.single_det_toolbar = NavigationToolbar2QT(self.single_det_canvas, self.single_det_widget)
        # Patch toolbar so we know when the user has zoomed/panned vs. pressed Home
        _canvas = self.single_det_canvas
        _orig_push = self.single_det_toolbar.push_current
        _orig_home = self.single_det_toolbar.home
        def _on_push_current():
            _canvas._user_navigated = True
            _orig_push()
        def _on_home(*args, **kwargs):
            _canvas._user_navigated = False
            _orig_home(*args, **kwargs)
        self.single_det_toolbar.push_current = _on_push_current
        self.single_det_toolbar.home = _on_home
        single_det_layout.addWidget(self.single_det_toolbar)
        single_det_layout.addWidget(self.single_det_canvas)
        self.tab_widget.addTab(self.single_det_widget, "Single Detector")

        # Connect tab change signal to update results when Results tab is selected
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Ensure Plots tab is selected by default
        self.tab_widget.setCurrentIndex(0)

        main_layout.addWidget(self.tab_widget, stretch=3)

        # Store last results for inspection
        self.last_results_df = None
        self.last_plot_data = None        # Store last plot data for deferred updates
        self.results_need_update = False  # Flag to track if results need updating
        self.plots_need_update = False    # Flag for detector plots
        self.polar_needs_update = False   # Flag for polar plot updates
        self.heatmap_needs_update = False  # Flag for angular heatmap updates
        self.single_det_needs_update = False  # Flag for single detector plot

    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Data Source (mode-switching group)
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout()

        # Mode radio buttons
        mode_layout = QHBoxLayout()
        self.file_mode_radio = QRadioButton("File (NXS)")
        self.doocs_mode_radio = QRadioButton("Live (DOOCS)")
        self.file_mode_radio.setChecked(True)
        mode_layout.addWidget(self.file_mode_radio)
        mode_layout.addWidget(self.doocs_mode_radio)
        source_layout.addLayout(mode_layout)

        # File source widget
        self.file_source_widget = QWidget()
        file_src_layout = QVBoxLayout(self.file_source_widget)
        file_src_layout.setContentsMargins(0, 0, 0, 0)
        self.file_label = QLabel("No folder selected")
        self.file_label.setWordWrap(True)
        file_src_layout.addWidget(self.file_label)
        file_btn = QPushButton("Select .nxs Folder")
        file_btn.clicked.connect(self.select_file)
        file_src_layout.addWidget(file_btn)
        source_layout.addWidget(self.file_source_widget)

        # DOOCS source widget
        self.doocs_source_widget = QWidget()
        doocs_src_layout = QVBoxLayout(self.doocs_source_widget)
        doocs_src_layout.setContentsMargins(0, 0, 0, 0)
        doocs_src_layout.addWidget(QLabel("DOOCS addresses (one per detector):"))
        self.doocs_addresses_edit = QTextEdit()
        self.doocs_addresses_edit.setPlaceholderText(
            "FACILITY/DEVICE/LOCATION/PROPERTY.TD\n..."
        )
        self.doocs_addresses_edit.setMaximumHeight(100)
        
        # Load addresses from config if available
        doocs_config = GlobalConfig.get_for_class('DoocspieStream')
        if doocs_config and 'addresses' in doocs_config:
            addresses_cfg = doocs_config['addresses']
            if isinstance(addresses_cfg, dict):
                addresses_list = [addresses_cfg[k] for k in sorted(addresses_cfg)]
            else:
                addresses_list = list(addresses_cfg)
            self.doocs_addresses_edit.setPlainText('\n'.join(addresses_list))
        
        doocs_src_layout.addWidget(self.doocs_addresses_edit)
        self.doocs_source_widget.setVisible(False)
        source_layout.addWidget(self.doocs_source_widget)

        self.file_mode_radio.toggled.connect(self._on_source_mode_changed)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Processing parameters
        param_group = QGroupBox("Processing Parameters")
        param_layout = QGridLayout()

        row = 0
        param_layout.addWidget(QLabel("Buffer Size:"), row, 0)
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(1, 100)
        self.buffer_size_spin.setValue(1)
        self.buffer_size_spin.setEnabled(False)
        self.buffer_size_spin.valueChanged.connect(self.on_buffer_size_changed)
        param_layout.addWidget(self.buffer_size_spin, row, 1)

        row += 1
        self.buffer_enable_check = QCheckBox("Enable Buffer")
        self.buffer_enable_check.setChecked(False)
        self.buffer_enable_check.stateChanged.connect(self.on_buffer_enable_changed)
        param_layout.addWidget(self.buffer_enable_check, row, 0, 1, 2)

        row += 1
        param_layout.addWidget(QLabel("Detectors/Thread:"), row, 0)
        self.det_per_thread_spin = QSpinBox()
        self.det_per_thread_spin.setRange(1, 16)
        self.det_per_thread_spin.setValue(4)
        param_layout.addWidget(self.det_per_thread_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Pipeline Depth:"), row, 0)
        self.pipeline_depth_spin = QSpinBox()
        self.pipeline_depth_spin.setRange(1, 10)
        self.pipeline_depth_spin.setValue(3)
        self.pipeline_depth_spin.setToolTip("Number of processing batches that can run in parallel")
        param_layout.addWidget(self.pipeline_depth_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Update Rate (Hz):"), row, 0)
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(1, 50)
        self.update_rate_spin.setValue(10)
        self.update_rate_spin.valueChanged.connect(self.on_update_rate_changed)
        param_layout.addWidget(self.update_rate_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Plot Downsample:"), row, 0)
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 20)
        self.downsample_spin.setValue(1)
        self.downsample_spin.valueChanged.connect(self.on_downsample_changed)
        param_layout.addWidget(self.downsample_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Peak Threshold:"), row, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.1)
        self.threshold_spin.valueChanged.connect(lambda _: self.update_processing_config())
        param_layout.addWidget(self.threshold_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Number of Peaks:"), row, 0)
        self.peak_no_spin = QSpinBox()
        self.peak_no_spin.setRange(1, 20)
        self.peak_no_spin.setValue(1)
        self.peak_no_spin.valueChanged.connect(lambda _: self.update_processing_config())
        param_layout.addWidget(self.peak_no_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("ROI Start:"), row, 0)
        self.roi_start_spin = QSpinBox()
        self.roi_start_spin.setRange(0, 10000)
        self.roi_start_spin.setValue(0)
        self.roi_start_spin.editingFinished.connect(self.update_processing_config)
        param_layout.addWidget(self.roi_start_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("ROI End:"), row, 0)
        self.roi_end_spin = QSpinBox()
        self.roi_end_spin.setRange(0, 10000)
        self.roi_end_spin.setValue(1000)
        self.roi_end_spin.editingFinished.connect(self.update_processing_config)
        param_layout.addWidget(self.roi_end_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Peak ROI Start:"), row, 0)
        self.peak_roi_start_spin = QSpinBox()
        self.peak_roi_start_spin.setRange(-1, 10000)
        self.peak_roi_start_spin.setValue(-1)
        self.peak_roi_start_spin.setToolTip("-1 = no limit (use full loaded ROI)")
        self.peak_roi_start_spin.editingFinished.connect(self.update_processing_config)
        param_layout.addWidget(self.peak_roi_start_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Peak ROI End:"), row, 0)
        self.peak_roi_end_spin = QSpinBox()
        self.peak_roi_end_spin.setRange(-1, 10000)
        self.peak_roi_end_spin.setValue(-1)
        self.peak_roi_end_spin.setToolTip("-1 = no limit (use full loaded ROI)")
        self.peak_roi_end_spin.editingFinished.connect(self.update_processing_config)
        param_layout.addWidget(self.peak_roi_end_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Smooth Window:"), row, 0)
        self.smooth_window_spin = QSpinBox()
        self.smooth_window_spin.setRange(1, 50)
        self.smooth_window_spin.setValue(1)
        self.smooth_window_spin.setToolTip("Window size for rolling average smoothing (1 = no smoothing)")
        self.smooth_window_spin.valueChanged.connect(lambda _: self.update_processing_config())
        param_layout.addWidget(self.smooth_window_spin, row, 1)

        row += 1
        self.normalize_check = QCheckBox("Normalize Data")
        self.normalize_check.setChecked(True)
        self.normalize_check.setToolTip("Divide data by global maximum before processing")
        self.normalize_check.stateChanged.connect(lambda _: self.update_processing_config())
        param_layout.addWidget(self.normalize_check, row, 0, 1, 2)

        row += 1
        self.shared_y_check = QCheckBox("Shared Y Axis")
        self.shared_y_check.setChecked(False)
        self.shared_y_check.setToolTip("Use the same y-axis range across all detector subplots")
        self.shared_y_check.stateChanged.connect(self.on_shared_y_changed)
        param_layout.addWidget(self.shared_y_check, row, 0, 1, 2)

        row += 1
        self.show_baseline_check = QCheckBox("Show Baseline Adjusted")
        self.show_baseline_check.setChecked(True)
        self.show_baseline_check.stateChanged.connect(self.on_show_baseline_changed)
        param_layout.addWidget(self.show_baseline_check, row, 0, 1, 2)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Pulse Stacking
        stack_group = QGroupBox("Pulse Stacking")
        stack_layout = QGridLayout()

        # Pre-populate from DoocspieStream stacking config or PeakFinder config
        doocs_cfg = GlobalConfig.get_for_class('DoocspieStream')
        stacking_cfg = doocs_cfg.get('stacking', {}) if doocs_cfg else {}
        pf_cfg = GlobalConfig.get_for_class('PeakFinder')
        default_stack_pulses = stacking_cfg.get('stackPulses', pf_cfg.get('stackPulses', True))
        _raw_start = stacking_cfg.get('pulseStackStart', pf_cfg.get('pulseStackStart', None))
        _raw_stop  = stacking_cfg.get('pulseStackStop',  pf_cfg.get('pulseStackStop',  None))
        _raw_step  = stacking_cfg.get('pulseStackStep',  pf_cfg.get('pulseStackStep',  pf_cfg.get('pulseStackSize', None)))
        default_pulse_start = -1 if _raw_start is None else int(_raw_start)
        default_pulse_stop  = -1 if _raw_stop  is None else int(_raw_stop)
        default_pulse_step  = -1 if _raw_step  is None else int(_raw_step)

        row = 0
        self.stack_pulses_check = QCheckBox("Stack Pulses")
        self.stack_pulses_check.setChecked(bool(default_stack_pulses))
        self.stack_pulses_check.setToolTip("Average traces over pulses within each train")
        self.stack_pulses_check.stateChanged.connect(self._on_stack_pulses_toggled)
        stack_layout.addWidget(self.stack_pulses_check, row, 0, 1, 2)

        row += 1
        stack_layout.addWidget(QLabel("Pulse Start:"), row, 0)
        self.pulse_stack_start_spin = QSpinBox()
        self.pulse_stack_start_spin.setRange(-1, 9999)
        self.pulse_stack_start_spin.setValue(default_pulse_start)
        self.pulse_stack_start_spin.setToolTip("-1 = start from first pulse")
        self.pulse_stack_start_spin.valueChanged.connect(lambda _: self.update_processing_config())
        stack_layout.addWidget(self.pulse_stack_start_spin, row, 1)

        row += 1
        stack_layout.addWidget(QLabel("Pulse Stop:"), row, 0)
        self.pulse_stack_stop_spin = QSpinBox()
        self.pulse_stack_stop_spin.setRange(-1, 9999)
        self.pulse_stack_stop_spin.setValue(default_pulse_stop)
        self.pulse_stack_stop_spin.setToolTip("-1 = include all pulses")
        self.pulse_stack_stop_spin.valueChanged.connect(lambda _: self.update_processing_config())
        stack_layout.addWidget(self.pulse_stack_stop_spin, row, 1)

        row += 1
        stack_layout.addWidget(QLabel("Pulse Step:"), row, 0)
        self.pulse_stack_step_spin = QSpinBox()
        self.pulse_stack_step_spin.setRange(-1, 9999)
        self.pulse_stack_step_spin.setValue(default_pulse_step)
        self.pulse_stack_step_spin.setToolTip("-1 = default stride (process every pulse)")
        self.pulse_stack_step_spin.valueChanged.connect(lambda _: self.update_processing_config())
        stack_layout.addWidget(self.pulse_stack_step_spin, row, 1)

        stack_group.setLayout(stack_layout)
        layout.addWidget(stack_group)

        # Set initial enabled state for stacking spinboxes
        _stack_enabled = bool(default_stack_pulses)
        self.pulse_stack_start_spin.setEnabled(_stack_enabled)
        self.pulse_stack_stop_spin.setEnabled(_stack_enabled)
        self.pulse_stack_step_spin.setEnabled(_stack_enabled)

        # Polarization Plot Parameters
        polar_group = QGroupBox("Polarization Plot")
        polar_layout = QGridLayout()

        row = 0
        polar_layout.addWidget(QLabel("Peak Number:"), row, 0)
        self.polar_peak_spin = QSpinBox()
        self.polar_peak_spin.setRange(0, 19)
        self.polar_peak_spin.setValue(0)
        self.polar_peak_spin.valueChanged.connect(self.on_polar_param_changed)
        polar_layout.addWidget(self.polar_peak_spin, row, 1)

        row += 1
        polar_layout.addWidget(QLabel("Value Type:"), row, 0)
        self.polar_value_combo = QComboBox()
        self.polar_value_combo.addItems(["height", "fwhm area"])
        self.polar_value_combo.currentTextChanged.connect(self.on_polar_param_changed)
        polar_layout.addWidget(self.polar_value_combo, row, 1)

        row += 1
        polar_layout.addWidget(QLabel("Beta (β):"), row, 0)
        self.polar_beta_spin = QDoubleSpinBox()
        self.polar_beta_spin.setRange(-2.0, 2.0)
        self.polar_beta_spin.setSingleStep(0.1)
        self.polar_beta_spin.setDecimals(4)
        # Load default beta from config
        calibrate_config = GlobalConfig.get_for_class('Calibrate')
        default_beta = calibrate_config.get('beta', 2.0) if calibrate_config else 2.0
        self.polar_beta_spin.setValue(default_beta)
        self.polar_beta_spin.valueChanged.connect(self.on_polar_param_changed)
        polar_layout.addWidget(self.polar_beta_spin, row, 1)

        # Fit-mode radio buttons
        row += 1
        self.polar_fit_plin_radio = QRadioButton("Fit Plin")
        self.polar_fit_beta_radio = QRadioButton("Fit Beta")
        self.polar_fit_plin_radio.setChecked(True)
        self._polar_fit_mode_group = QButtonGroup(self)
        self._polar_fit_mode_group.addButton(self.polar_fit_plin_radio, 0)
        self._polar_fit_mode_group.addButton(self.polar_fit_beta_radio, 1)
        self._polar_fit_mode_group.idToggled.connect(self._on_fit_mode_changed)
        radio_row = QHBoxLayout()
        radio_row.addWidget(self.polar_fit_plin_radio)
        radio_row.addWidget(self.polar_fit_beta_radio)
        polar_layout.addLayout(radio_row, row, 0, 1, 2)

        row += 1
        polar_layout.addWidget(QLabel("Plin (fixed):"), row, 0)
        self.polar_plin_spin = QDoubleSpinBox()
        self.polar_plin_spin.setRange(0.0, 1.0)
        self.polar_plin_spin.setSingleStep(0.01)
        self.polar_plin_spin.setDecimals(4)
        self.polar_plin_spin.setValue(1.0)
        self.polar_plin_spin.setEnabled(False)  # disabled in Fit Plin mode
        self.polar_plin_spin.valueChanged.connect(self.on_polar_param_changed)
        polar_layout.addWidget(self.polar_plin_spin, row, 1)

        row += 1
        self.polar_fix_phi_check = QCheckBox("Fix φ")
        self.polar_fix_phi_check.setChecked(False)
        self.polar_fix_phi_check.stateChanged.connect(self._on_fix_phi_changed)
        polar_layout.addWidget(self.polar_fix_phi_check, row, 0)

        self.polar_phi_spin = QDoubleSpinBox()
        self.polar_phi_spin.setRange(-180.0, 180.0)
        self.polar_phi_spin.setSingleStep(1.0)
        self.polar_phi_spin.setDecimals(1)
        self.polar_phi_spin.setSuffix(" °")
        self.polar_phi_spin.setValue(0.0)
        self.polar_phi_spin.setEnabled(False)  # enabled only when Fix φ is checked
        self.polar_phi_spin.valueChanged.connect(self.on_polar_param_changed)
        polar_layout.addWidget(self.polar_phi_spin, row, 1)

        polar_group.setLayout(polar_layout)
        layout.addWidget(polar_group)

        # Angular Heatmap Parameters
        heatmap_group = QGroupBox("Angular Heatmap")
        heatmap_layout = QGridLayout()

        row = 0
        heatmap_layout.addWidget(QLabel("Sample Min:"), row, 0)
        self.heatmap_smin_spin = QSpinBox()
        self.heatmap_smin_spin.setRange(0, 10000)
        self.heatmap_smin_spin.setValue(0)
        self.heatmap_smin_spin.valueChanged.connect(self.on_heatmap_param_changed)
        heatmap_layout.addWidget(self.heatmap_smin_spin, row, 1)

        row += 1
        heatmap_layout.addWidget(QLabel("Sample Max:"), row, 0)
        self.heatmap_smax_spin = QSpinBox()
        self.heatmap_smax_spin.setRange(0, 10000)
        self.heatmap_smax_spin.setValue(1000)
        self.heatmap_smax_spin.valueChanged.connect(self.on_heatmap_param_changed)
        heatmap_layout.addWidget(self.heatmap_smax_spin, row, 1)

        row += 1
        self.heatmap_interpolate_check = QCheckBox("Interpolate")
        self.heatmap_interpolate_check.setChecked(True)
        self.heatmap_interpolate_check.stateChanged.connect(self.on_heatmap_param_changed)
        heatmap_layout.addWidget(self.heatmap_interpolate_check, row, 0, 1, 2)

        row += 1
        self.heatmap_showpeaks_check = QCheckBox("Show Peaks")
        self.heatmap_showpeaks_check.setChecked(True)
        self.heatmap_showpeaks_check.stateChanged.connect(self.on_heatmap_param_changed)
        heatmap_layout.addWidget(self.heatmap_showpeaks_check, row, 0, 1, 2)

        heatmap_group.setLayout(heatmap_layout)
        layout.addWidget(heatmap_group)

        # Calibration
        calib_group = QGroupBox("Intensity Calibration")
        calib_layout = QVBoxLayout()

        calib_load_btn = QPushButton("Load calib.yaml")
        calib_load_btn.clicked.connect(self.load_calibration_file)
        calib_layout.addWidget(calib_load_btn)

        self.calib_status_label = QLabel("No calibration loaded")
        self.calib_status_label.setWordWrap(True)
        calib_layout.addWidget(self.calib_status_label)

        calib_clear_btn = QPushButton("Clear Calibration")
        calib_clear_btn.clicked.connect(self.clear_calibration)
        calib_layout.addWidget(calib_clear_btn)

        # --- Calculate from buffer ---
        calc_group = QGroupBox("Calculate from Buffer")
        calc_layout = QGridLayout()

        calc_layout.addWidget(QLabel("Peak No:"), 0, 0)
        self.calib_peakno_spin = QSpinBox()
        self.calib_peakno_spin.setRange(0, 20)
        self.calib_peakno_spin.setValue(0)
        calc_layout.addWidget(self.calib_peakno_spin, 0, 1)

        calc_layout.addWidget(QLabel("Plin:"), 1, 0)
        self.calib_plin_spin = QDoubleSpinBox()
        self.calib_plin_spin.setRange(0.0, 1.0)
        self.calib_plin_spin.setSingleStep(0.01)
        self.calib_plin_spin.setDecimals(4)
        self.calib_plin_spin.setValue(1.0)
        calc_layout.addWidget(self.calib_plin_spin, 1, 1)

        calc_layout.addWidget(QLabel("Beta (β₂):"), 2, 0)
        self.calib_beta_spin = QDoubleSpinBox()
        self.calib_beta_spin.setRange(-2.0, 4.0)
        self.calib_beta_spin.setSingleStep(0.1)
        self.calib_beta_spin.setDecimals(4)
        self.calib_beta_spin.setValue(2.0)
        calc_layout.addWidget(self.calib_beta_spin, 2, 1)

        calc_layout.addWidget(QLabel("phi (°):"), 3, 0)
        self.calib_phi_spin = QDoubleSpinBox()
        self.calib_phi_spin.setRange(-360.0, 360.0)
        self.calib_phi_spin.setSingleStep(1.0)
        self.calib_phi_spin.setDecimals(2)
        self.calib_phi_spin.setValue(0.0)
        calc_layout.addWidget(self.calib_phi_spin, 3, 1)

        calc_layout.addWidget(QLabel("Int. Method:"), 4, 0)
        self.calib_intmethod_combo = QComboBox()
        self.calib_intmethod_combo.addItems(["height", "fwhm area"])
        calc_layout.addWidget(self.calib_intmethod_combo, 4, 1)

        calc_btn = QPushButton("Calculate & Load")
        calc_btn.clicked.connect(self.calibrate_from_buffer)
        calc_layout.addWidget(calc_btn, 5, 0, 1, 2)

        calc_group.setLayout(calc_layout)
        calib_layout.addWidget(calc_group)

        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        # Snapshots
        snap_group = QGroupBox("Reference Snapshots")
        snap_layout = QGridLayout()

        snap_layout.addWidget(QLabel("Alpha:"), 0, 0)
        self.snapshot_alpha_spin = QDoubleSpinBox()
        self.snapshot_alpha_spin.setRange(0.05, 1.0)
        self.snapshot_alpha_spin.setSingleStep(0.05)
        self.snapshot_alpha_spin.setDecimals(2)
        self.snapshot_alpha_spin.setValue(0.3)
        self.snapshot_alpha_spin.setToolTip("Opacity of the snapshot reference lines")
        snap_layout.addWidget(self.snapshot_alpha_spin, 0, 1)

        take_snap_btn = QPushButton("Take Snapshot")
        take_snap_btn.clicked.connect(self.take_snapshot)
        take_snap_btn.setToolTip("Freeze current spectra as a static reference overlay")
        snap_layout.addWidget(take_snap_btn, 1, 0, 1, 2)

        manage_snap_btn = QPushButton("Manage Snapshots")
        manage_snap_btn.clicked.connect(self.show_snapshot_manager)
        manage_snap_btn.setToolTip("Toggle visibility or delete individual snapshots")
        snap_layout.addWidget(manage_snap_btn, 2, 0, 1, 2)

        snap_group.setLayout(snap_layout)
        layout.addWidget(snap_group)

        # Control buttons
        btn_group = QGroupBox("Control")
        btn_layout = QVBoxLayout()

        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        self.clear_buffer_btn = QPushButton("Clear Buffer")
        self.clear_buffer_btn.clicked.connect(self.clear_buffer)
        btn_layout.addWidget(self.clear_buffer_btn)

        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)

        # Detector selection
        det_group = QGroupBox("Detector Selection")
        det_outer_layout = QVBoxLayout()

        det_btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("All")
        self.select_all_btn.clicked.connect(self.select_all_detectors)
        self.deselect_all_btn = QPushButton("None")
        self.deselect_all_btn.clicked.connect(self.deselect_all_detectors)
        det_btn_layout.addWidget(self.select_all_btn)
        det_btn_layout.addWidget(self.deselect_all_btn)
        det_outer_layout.addLayout(det_btn_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(120)

        self.det_checkbox_widget = QWidget()
        self.det_checkbox_layout = QGridLayout(self.det_checkbox_widget)
        self.det_checkbox_layout.setSpacing(2)

        self.create_detector_checkboxes(16)

        scroll_area.setWidget(self.det_checkbox_widget)
        det_outer_layout.addWidget(scroll_area)

        det_group.setLayout(det_outer_layout)
        layout.addWidget(det_group)

        # Performance display
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout()

        self.perf_text = QTextEdit()
        self.perf_text.setReadOnly(True)
        self.perf_text.setMaximumHeight(180)
        font = QFont("Courier")
        font.setPointSize(8)
        self.perf_text.setFont(font)
        perf_layout.addWidget(self.perf_text)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        layout.addStretch()

        return panel

    def create_detector_checkboxes(self, n_detectors):
        """Create detector selection checkboxes"""
        for cb in self.detector_checkboxes:
            cb.deleteLater()
        self.detector_checkboxes = []

        n_cols = 4
        for i in range(n_detectors):
            cb = QCheckBox(f"D{i}")
            cb.setChecked(i in self.enabled_detectors)
            cb.stateChanged.connect(lambda state, det=i: self.on_detector_toggled(det, state))
            self.det_checkbox_layout.addWidget(cb, i // n_cols, i % n_cols)
            self.detector_checkboxes.append(cb)

    def on_detector_toggled(self, detector_id, state):
        if state:
            self.enabled_detectors.add(detector_id)
        else:
            self.enabled_detectors.discard(detector_id)

    def select_all_detectors(self):
        for cb in self.detector_checkboxes:
            cb.setChecked(True)

    def deselect_all_detectors(self):
        for cb in self.detector_checkboxes:
            cb.setChecked(False)

    def on_downsample_changed(self, value):
        """Update downsample in plot worker"""
        if self.plot_worker:
            self.plot_worker.set_downsample(value)

    def on_update_rate_changed(self, value):
        """Restart the main timer with the new interval when the rate changes live."""
        if self.running and self.main_timer.isActive():
            self.main_timer.setInterval(int(1000 / value))

    def on_buffer_enable_changed(self, state):
        """Enable/disable buffer; unchecked forces buffer size to 1."""
        enabled = bool(state)
        self.buffer_size_spin.setEnabled(enabled)
        effective_size = self.buffer_size_spin.value() if enabled else 1
        if self.circular_buffer is not None:
            self.circular_buffer.resize(effective_size)
        print(f"Buffer {'enabled' if enabled else 'disabled'}: effective size={effective_size}")

    def on_buffer_size_changed(self, value):
        """Resize the circular buffer live when the spinbox changes."""
        if not self.buffer_enable_check.isChecked():
            return
        if self.circular_buffer is not None:
            self.circular_buffer.resize(value)
        print(f"Buffer resized to {value}")

    def update_processing_config(self):
        """Update processing config when spinbox values change (on Enter/focus loss)"""
        if hasattr(self, 'processing_config'):
            self.processing_config['threshold'] = self.threshold_spin.value()
            self.processing_config['peakNo'] = self.peak_no_spin.value()-1  # zero-index internally
            self.processing_config['roi'] = [self.roi_start_spin.value(), self.roi_end_spin.value()]
            peak_roi_start = self.peak_roi_start_spin.value()
            peak_roi_end = self.peak_roi_end_spin.value()
            self.processing_config['peakfinder_roi'] = [
                None if peak_roi_start == -1 else peak_roi_start,
                None if peak_roi_end == -1 else peak_roi_end,
            ]
            self.processing_config['smoothWindow'] = self.smooth_window_spin.value()

            # Pulse stacking
            if hasattr(self, 'stack_pulses_check'):
                self.processing_config['stackPulses'] = self.stack_pulses_check.isChecked()
                start = self.pulse_stack_start_spin.value()
                stop  = self.pulse_stack_stop_spin.value()
                step  = self.pulse_stack_step_spin.value()
                self.processing_config['pulseStackStart'] = None if start == -1 else start
                self.processing_config['pulseStackStop']  = None if stop  == -1 else stop
                self.processing_config['pulseStackStep']  = None if step  == -1 else step

            # Update plot worker ROI so plots show the limited range
            if hasattr(self, 'plot_worker') and self.plot_worker:
                self.plot_worker.set_roi(self.roi_start_spin.value(), self.roi_end_spin.value())

            # Reset canvas background to force axis limits update on next plot
            if hasattr(self, 'plot_canvas') and self.plot_canvas:
                self.plot_canvas.background = None
                # Force a redraw to update axis limits
                for ax in self.plot_canvas.axes:
                    ax.relim()
                    ax.autoscale_view(scalex=True, scaley=False)
                self.plot_canvas.draw_idle()

            print(f"Config updated: threshold={self.processing_config['threshold']}, "
                  f"peakNo={self.processing_config['peakNo']}, "
                  f"roi={self.processing_config['roi']}, "
                  f"peakfinder_roi={self.processing_config['peakfinder_roi']}, "
                  f"smoothWindow={self.processing_config['smoothWindow']}")

    def _on_stack_pulses_toggled(self, state):
        enabled = bool(state)
        self.pulse_stack_start_spin.setEnabled(enabled)
        self.pulse_stack_stop_spin.setEnabled(enabled)
        self.pulse_stack_step_spin.setEnabled(enabled)
        self.update_processing_config()

    def _on_source_mode_changed(self, checked):
        self.file_source_widget.setVisible(self.file_mode_radio.isChecked())
        self.doocs_source_widget.setVisible(self.doocs_mode_radio.isChecked())

    def select_file(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with .nxs Files")
        if folder:
            self.file_label.setText(folder)

    def start_processing(self):
        if self.file_mode_radio.isChecked():
            if not self.file_label.text() or self.file_label.text() == "No folder selected":
                self.file_label.setText("Please select a folder first!")
                return
            try:
                self.data_simulator = DataStreamSimulator(Path(self.file_label.text()))
                self.n_detectors = len(self.data_simulator.data.coords['detector'])
            except Exception as e:
                self.file_label.setText(f"Error: {e}")
                return
        else:
            raw = self.doocs_addresses_edit.toPlainText().strip()
            addresses = [a.strip() for a in raw.splitlines() if a.strip()]
            if not addresses:
                self.doocs_addresses_edit.setPlaceholderText("Enter at least one DOOCS address!")
                return
            try:
                self.data_simulator = DoocspieStream(addresses)
                self.n_detectors = len(self.data_simulator.data.coords['detector'])
            except Exception as e:
                print(f"DoocspieStream init error: {e}")
                return

            # Apply angles from DoocspieStream config to the polar / heatmap canvases
            doocs_cfg = GlobalConfig.get_for_class('DoocspieStream')
            if doocs_cfg and 'angles' in doocs_cfg:
                angles_cfg = doocs_cfg['angles']
                if isinstance(angles_cfg, dict):
                    angles_list = [angles_cfg[k] for k in sorted(angles_cfg)]
                else:
                    angles_list = list(angles_cfg)
                self.polar_canvas.set_angles(angles_list)
                self.heatmap_canvas.set_angles(angles_list)

        # Recreate canvas
        self.recreate_canvas()

        # Update detectors
        self.enabled_detectors = set(range(self.n_detectors))
        self.create_detector_checkboxes(self.n_detectors)

        # Initialize buffer (size=1 when buffer is disabled)
        effective_buf_size = self.buffer_size_spin.value() if self.buffer_enable_check.isChecked() else 1
        self.circular_buffer = CircularBuffer(effective_buf_size)

        # Setup workers
        self.setup_workers()

        # Initialize canvas background for blitting
        self.canvas.init_blit()

        # Reset pipeline
        self.stage_load = None
        self.stage_process = None
        self.stage_plot = None

        # Start
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        interval = int(1000 / self.update_rate_spin.value())
        self.main_timer.start(interval)

    def recreate_canvas(self):
        """Recreate canvas with correct detector count"""
        old_canvas = self.canvas

        # Remove old canvas from tab widget
        tab_index = self.tab_widget.indexOf(old_canvas)
        if tab_index >= 0:
            self.tab_widget.removeTab(tab_index)
        old_canvas.deleteLater()

        # Create new canvas and add back to tab widget at the same position
        self.canvas = FastMplCanvas(self, width=10, height=8, dpi=100, n_detectors=self.n_detectors)
        self.tab_widget.insertTab(0, self.canvas, "Plots")
        self.tab_widget.setCurrentIndex(0)  # Make sure Plots tab is active

        # Update single detector dropdown to match new detector count
        self.single_det_combo.blockSignals(True)
        current_det = self.single_det_combo.currentIndex()
        self.single_det_combo.clear()
        for i in range(self.n_detectors):
            self.single_det_combo.addItem(f"Det {i}")
        self.single_det_combo.setCurrentIndex(min(current_det, self.n_detectors - 1))
        self.single_det_combo.blockSignals(False)

        # Clear snapshots from the old canvas and reset manager
        self.single_det_canvas.clear_all_snapshots()
        if self.snapshot_manager is not None and self.snapshot_manager.isVisible():
            self.snapshot_manager.refresh([])

    def setup_workers(self):
        """Setup process pool and plotting workers"""
        # Shutdown existing process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=False)

        if self.plot_worker:
            self.plot_worker.stop()
        if self.plot_thread:
            self.plot_thread.join(timeout=1.0)

        # Create process pool for parallel processing (bypasses GIL)
        # Workers = (detectors / detectors_per_worker) * pipeline_depth
        # This allows multiple processing batches to run simultaneously
        det_per_worker = self.det_per_thread_spin.value()
        pipeline_depth = self.pipeline_depth_spin.value()
        n_workers_per_batch = int(np.ceil(self.n_detectors / det_per_worker))
        n_workers_total = n_workers_per_batch * pipeline_depth

        self.process_pool = ProcessPoolExecutor(max_workers=n_workers_total)

        print(f"Process pool: {n_workers_per_batch} workers/batch × {pipeline_depth} batches = {n_workers_total} total workers")

        # Store detector groupings for work distribution
        self.detector_groups = []
        for i in range(n_workers_per_batch):
            start_det = i * det_per_worker
            end_det = min((i + 1) * det_per_worker, self.n_detectors)
            self.detector_groups.append(list(range(start_det, end_det)))

        # Store config for workers - only override GUI-controlled parameters
        # Other parameters (like stackTrains, etc.) come from config.yaml
        _ps_start = self.pulse_stack_start_spin.value()
        _ps_stop  = self.pulse_stack_stop_spin.value()
        _ps_step  = self.pulse_stack_step_spin.value()
        self.processing_config = {
            'threshold': self.threshold_spin.value(),
            'peakNo': self.peak_no_spin.value(),
            'roi': [self.roi_start_spin.value(), self.roi_end_spin.value()],
            'smoothWindow': self.smooth_window_spin.value(),
            'stackPulses': self.stack_pulses_check.isChecked(),
            'pulseStackStart': None if _ps_start == -1 else _ps_start,
            'pulseStackStop':  None if _ps_stop  == -1 else _ps_stop,
            'pulseStackStep':  None if _ps_step  == -1 else _ps_step,
        }

        # Create plot preparation worker (thread is fine for I/O-bound work)
        self.plot_worker = PlotPreparationWorker(self.n_detectors)
        self.plot_worker.set_downsample(self.downsample_spin.value())
        self.plot_worker.set_roi(self.roi_start_spin.value(), self.roi_end_spin.value())
        self.plot_worker.plot_ready.connect(self.on_plot_ready)

        self.plot_thread = Thread(target=self.plot_worker.run, daemon=True)
        self.plot_thread.start()

    def main_loop_iteration(self):
        """Main loop - staggered pipeline"""
        iter_start = time.time()

        # Stage 3: Trigger plot preparation (stage_plot has processed data)
        if self.stage_plot is not None:
            data = self.stage_plot.get('data')
            normalized_data = self.stage_plot.get('normalized_data', {})
            results = self.stage_plot.get('results')

            if data is not None:
                train_ids = data.indexes['pulse'].get_level_values('trainId')
                pulse_ids = data.indexes['pulse'].get_level_values('pulseId')
                last_train = train_ids[-1]
                last_pulse = pulse_ids[-1]

                # Store results for inspection in the Results tab
                if results is not None and hasattr(results, 'empty'):
                    self.last_results_df = results
                    # Only update table if Results tab is currently visible
                    if self.tab_widget.currentIndex() == 1:  # Results tab
                        self.update_results_table()
                    else:
                        self.results_need_update = True  # Flag for later update

                # Send to plot worker (non-blocking)
                prep_start = time.time()
                self.plot_worker.prepare(
                    data, normalized_data, results,
                    self.enabled_detectors.copy(),
                    last_train, last_pulse
                )
                prep_time = time.time() - prep_start

                if hasattr(self, '_last_plot_time'):
                    plot_elapsed = time.time() - self._last_plot_time
                    if plot_elapsed > 50:  # Only print if > 50ms
                        print(f"Plot stage: {plot_elapsed*1000:.1f}ms (prep:{prep_time*1000:.1f}ms)")
                self._last_plot_time = time.time()

        # Check for completed processing futures from previous iteration
        self.check_processing_futures()

        # Only shift pipeline when processing is complete (results is a DataFrame, not a list)
        # This ensures we have synchronized data + results for plotting
        if self.stage_load is not None:
            results = self.stage_load.get('results')
            # Results starts as a list, becomes DataFrame when all futures complete
            is_complete = not isinstance(results, list)
            if is_complete:
                self.stage_plot = self.stage_process
                self.stage_process = self.stage_load
                # Don't clear stage_load yet - let it be replaced by new load
        else:
            # No pending load, shift normally
            self.stage_plot = self.stage_process
            self.stage_process = None

        # Stage 1: Load and submit to process pool
        # Always advance/drain the stream on every tick so we stay on the live
        # edge (stale chunks are discarded).  Only feed the data into the
        # pipeline when the previous batch has finished; otherwise we just drop
        # the chunk and wait for the next tick.
        load_start = time.time()
        load_allowed = (self.stage_load is None) or (
            not isinstance(self.stage_load.get('results'), list)
        )
        new_train = self.data_simulator.get_next_train()  # always drain/advance
        if not load_allowed:
            new_train = None  # pipeline still busy – skip this chunk
        if new_train is not None:
            self.circular_buffer.push(new_train)

            stacked_data = self.stack_buffer_data()
            if stacked_data is not None:
                # Apply per-detector intensity calibration BEFORE normalization
                # so relative detector weights influence the global scale
                if self.calib_coefficients:
                    det_coords = stacked_data.coords['detector'].values
                    coeffs = np.array([self.calib_coefficients.get(int(d), 1.0)
                                       for d in det_coords])
                    calib_da = xr.DataArray(coeffs, coords=[stacked_data.coords['detector']],
                                            dims=['detector'])
                    stacked_data = stacked_data * calib_da

                # Normalize the data BEFORE chunking to ensure consistent scaling
                # This way all detectors are normalized to the global maximum
                norm_start = time.time()
                if self.normalize_check.isChecked():
                    data_max = stacked_data.max().values
                    if data_max > 0:
                        stacked_data = stacked_data / data_max
                norm_time = time.time() - norm_start

                # Submit work to process pool
                enabled_set = self.enabled_detectors.copy()
                self.processing_futures = []

                submit_start = time.time()
                for i, detector_indices in enumerate(self.detector_groups):
                    # Filter detector indices for this worker
                    active_indices = [d for d in detector_indices if d in enabled_set]

                    if active_indices:
                        # Select this worker's detectors (xarray DataArrays are directly picklable)
                        detector_data = stacked_data.sel(detector=active_indices)

                        # Submit to process pool
                        future = self.process_pool.submit(
                            process_detector_chunk,
                            (detector_data, self.processing_config, i)
                        )
                        self.processing_futures.append(future)

                submit_time = time.time() - submit_start

                self.stage_load = {
                    'data': stacked_data,
                    'results': [],
                    'normalized_data': {},
                    'worker_count': len(self.processing_futures),
                    'finished_count': 0,
                    'process_start_time': time.time(),
                    'norm_time': norm_time,
                    'submit_time': submit_time
                }

                # Record load cycle time
                if hasattr(self, '_last_load_time'):
                    load_cycle_time = time.time() - self._last_load_time
                    self.performance_monitor.record_stage('load_cycle', load_cycle_time)
                self._last_load_time = time.time()

        self.performance_monitor.record_stage('load', time.time() - load_start)

        self.performance_monitor.record_iteration(time.time() - iter_start)

    def check_processing_futures(self):
        """Check for completed processing futures and collect results"""
        if not self.processing_futures or self.stage_load is None:
            return

        # Check each future
        completed = []
        for future in self.processing_futures:
            if future.done():
                completed.append(future)
                try:
                    worker_id, results, normalized_data = future.result()

                    # Record timing
                    self.performance_monitor.record_thread(worker_id, 0)

                    # Store normalized data (already an xarray DataArray)
                    if normalized_data is not None:
                        self.stage_load['normalized_data'][worker_id] = normalized_data

                    # Correct peak positions for ROI offset: PeakFinder returns 0-based array. Fixed in latest ToFPipechange
                    # indices, but the trace is displayed using the actual sample coordinates
                    # (which start at roi_start). Adding roi_start aligns markers with trace.
                    if results is not None and isinstance(self.stage_load['results'], list):
                        """
                        roi_start = self.processing_config.get('roi', [0, 10000])[0]
                        if roi_start and roi_start > 0 and not results.empty:
                            results = results.copy()
                            results['pos'] = results['pos'] + roi_start
                            for col in ('baseline left', 'baseline right'):
                                if col in results.columns:
                                    results[col] = results[col] + roi_start
                        """
                        self.stage_load['results'].append(results)

                    self.stage_load['finished_count'] += 1

                except Exception as e:
                    print(f"Error collecting process result: {e}")
                    self.stage_load['finished_count'] += 1

        # Remove completed futures
        for future in completed:
            self.processing_futures.remove(future)

        # Check if all workers finished
        if self.stage_load['finished_count'] >= self.stage_load['worker_count']:
            if 'process_start_time' in self.stage_load:
                process_duration = time.time() - self.stage_load['process_start_time']
                self.performance_monitor.record_stage('process', process_duration)

            # Combine results
            if isinstance(self.stage_load['results'], list) and self.stage_load['results']:
                non_empty = [r for r in self.stage_load['results'] if hasattr(r, 'empty') and not r.empty]
                self.stage_load['results'] = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
            elif isinstance(self.stage_load['results'], list):
                self.stage_load['results'] = pd.DataFrame()

    def stack_buffer_data(self):
        """Stack buffer data by concatenating trains along the pulse dimension.

        Each train has dimensions (pulse, detector, sample). This method concatenates
        all trains in the buffer along the pulse dimension, creating a larger dataset
        that PeakFinder's .stack() method will then average.

        The averaging is handled by PeakFinder.stack() based on stackTrains/stackPulses config.

        Note: Returns data even if buffer isn't full yet (allows immediate processing).
        ROI is applied here EARLY to reduce data size for all subsequent operations.
        """
        trains = self.circular_buffer.get_all()
        if not trains:
            return None

        # Get ROI from processing config - apply it early for performance
        roi = getattr(self, 'processing_config', {}).get('roi', [None, None])
        roi_start = roi[0] if roi[0] is not None else None
        roi_end = roi[1] if roi[1] is not None else None

        # Apply ROI BEFORE computing to reduce data loaded from dask arrays
        computed_trains = []
        for train in trains:
            # Apply ROI first (on dask array) - this makes compute() only load the ROI region
            if roi_start is not None or roi_end is not None:
                train = train.sel(sample=slice(roi_start, roi_end))

            # Now compute (loads only the ROI region into memory)
            if hasattr(train.data, 'compute'):
                train_computed = train.compute()
            else:
                train_computed = train
            computed_trains.append(train_computed)

        if len(computed_trains) == 1:
            return computed_trains[0]

        # Concatenate all trains along the pulse dimension
        # PeakFinder.stack() will handle the averaging
        stacked = xr.concat(computed_trains, dim='pulse')
        return stacked

    @pyqtSlot(list)
    def on_plot_ready(self, plot_data_list: List[PlotData]):
        """Handle prepared plot data - update canvas (runs in GUI thread)"""
        # Store plot data for later use
        self.last_plot_data = plot_data_list

        # Only update detector plots if Plots tab is visible
        if self.tab_widget.currentIndex() == 0:  # Plots tab
            render_start = time.time()
            self.canvas.fast_update(plot_data_list,
                                    show_baseline=self.show_baseline_check.isChecked(),
                                    normalize=self.normalize_check.isChecked(),
                                    shared_y=self.shared_y_check.isChecked())
            self.performance_monitor.record_stage('plot_render', time.time() - render_start)
        else:
            self.plots_need_update = True

        # Update polar plot if it's visible or flag for update
        if self.tab_widget.currentIndex() == 2:  # Polarization tab
            self.update_polar_plot()
        else:
            self.polar_needs_update = True

        # Update heatmap if it's visible or flag for update
        if self.tab_widget.currentIndex() == 3:  # Angular Heatmap tab
            self.update_heatmap_plot()
        else:
            self.heatmap_needs_update = True

        # Update single detector plot if visible or flag for update
        if self.tab_widget.currentIndex() == 4:  # Single Detector tab
            self.update_single_detector_plot(plot_data_list)
        else:
            self.single_det_needs_update = True

    def on_tab_changed(self, index):
        """Handle tab changes - update views when tabs are selected"""
        if index == 0 and self.plots_need_update:  # Plots tab
            if self.last_plot_data is not None:
                self.canvas.fast_update(self.last_plot_data,
                                        show_baseline=self.show_baseline_check.isChecked(),
                                        normalize=self.normalize_check.isChecked(),
                                        shared_y=self.shared_y_check.isChecked())
            self.plots_need_update = False
        elif index == 1 and self.results_need_update:  # Results tab
            self.update_results_table()
            self.results_need_update = False
        elif index == 2 and self.polar_needs_update:  # Polarization tab
            self.update_polar_plot()
            self.polar_needs_update = False
        elif index == 3 and self.heatmap_needs_update:  # Angular Heatmap tab
            self.update_heatmap_plot()
            self.heatmap_needs_update = False
        elif index == 4 and self.single_det_needs_update:  # Single Detector tab
            if self.last_plot_data is not None:
                self.update_single_detector_plot(self.last_plot_data)
            self.single_det_needs_update = False

    def _on_fit_mode_changed(self, btn_id, checked):
        """Switch which spinbox is active based on the fit-mode radio buttons"""
        if not checked:
            return
        fit_beta = (btn_id == 1)
        self.polar_beta_spin.setEnabled(not fit_beta)
        self.polar_plin_spin.setEnabled(fit_beta)
        self.on_polar_param_changed()

    def _on_fix_phi_changed(self, state):
        """Enable/disable the phi spinbox based on the Fix φ checkbox"""
        self.polar_phi_spin.setEnabled(bool(state))
        self.on_polar_param_changed()

    def on_polar_param_changed(self):
        """Handle changes to polar plot parameters"""
        if self.tab_widget.currentIndex() == 2:
            self.update_polar_plot()

    def on_shared_y_changed(self, state):
        """Re-render Plots tab immediately when shared y-axis toggle changes"""
        if self.last_plot_data is not None and self.tab_widget.currentIndex() == 0:
            self.canvas.background = None  # Force full redraw
            self.canvas.fast_update(self.last_plot_data,
                                    show_baseline=self.show_baseline_check.isChecked(),
                                    normalize=self.normalize_check.isChecked(),
                                    shared_y=bool(state))

    def on_show_baseline_changed(self, state):
        """Re-render Plots tab immediately when baseline toggle changes"""
        if self.last_plot_data is not None and self.tab_widget.currentIndex() == 0:
            self.canvas.background = None  # Force full redraw so artists are registered
            self.canvas.fast_update(self.last_plot_data, show_baseline=bool(state),
                                    normalize=self.normalize_check.isChecked(),
                                    shared_y=self.shared_y_check.isChecked())
        elif self.last_plot_data is not None and self.tab_widget.currentIndex() == 4:
            self.update_single_detector_plot(self.last_plot_data)

    def on_heatmap_param_changed(self):
        """Update heatmap when controls change"""
        if self.tab_widget.currentIndex() == 3:
            self.update_heatmap_plot()

    def update_heatmap_plot(self):
        """Rebuild the angular heatmap with current plot data and results"""
        if self.last_plot_data is None:
            return
        self.heatmap_canvas.update_heatmap(
            self.last_plot_data,
            self.last_results_df,
            sample_min=self.heatmap_smin_spin.value(),
            sample_max=self.heatmap_smax_spin.value(),
            interpolate=self.heatmap_interpolate_check.isChecked(),
            show_peaks=self.heatmap_showpeaks_check.isChecked(),
        )

    def update_single_detector_plot(self, plot_data_list):
        """Update the single detector canvas with the currently selected detector"""
        if plot_data_list is None:
            return
        det_idx = self.single_det_combo.currentIndex()
        if det_idx < 0 or det_idx >= len(plot_data_list):
            return
        plot_data = plot_data_list[det_idx]
        self.single_det_canvas.ax.set_title(f'Detector {det_idx}', fontsize=10)
        self.single_det_canvas.update_plot(
            plot_data,
            show_baseline=self.show_baseline_check.isChecked(),
            normalize=self.normalize_check.isChecked(),
            det_idx=det_idx
        )

    def on_single_det_changed(self, index):
        """Redraw single detector plot when dropdown selection changes"""
        # Reset navigation so the new detector auto-scales on next update
        self.single_det_canvas._user_navigated = False
        if self.last_plot_data is not None and self.tab_widget.currentIndex() == 4:
            self.update_single_detector_plot(self.last_plot_data)

    def clear_buffer(self):
        """Clear the circular buffer"""
        if self.circular_buffer is not None:
            self.circular_buffer.clear()

    # ------------------------------------------------------------------ #
    # Snapshot helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_snapshot_labels(self):
        """Return [(label, visible), ...] from the main canvas snapshot list."""
        return [(s['label'], s['visible']) for s in self.canvas.snapshots]

    def take_snapshot(self):
        """Freeze the current plot data as a static reference overlay on all canvases."""
        if self.last_plot_data is None:
            return
        alpha = self.snapshot_alpha_spin.value()
        det_idx = self.single_det_combo.currentIndex()
        self.canvas.take_snapshot(self.last_plot_data, alpha=alpha)
        self.single_det_canvas.take_snapshot(
            self.last_plot_data, alpha=alpha,
            current_det_idx=max(det_idx, 0)
        )
        if self.snapshot_manager is not None and self.snapshot_manager.isVisible():
            self.snapshot_manager.refresh(self._get_snapshot_labels())

    def remove_snapshot(self, idx):
        """Remove snapshot at *idx* from all canvases and refresh the manager dialog."""
        self.canvas.remove_snapshot(idx)
        self.single_det_canvas.remove_snapshot(idx)
        if self.snapshot_manager is not None and self.snapshot_manager.isVisible():
            self.snapshot_manager.refresh(self._get_snapshot_labels())

    def set_snapshot_visible(self, idx, visible):
        """Toggle snapshot visibility on all canvases."""
        self.canvas.set_snapshot_visible(idx, visible)
        self.single_det_canvas.set_snapshot_visible(idx, visible)

    def clear_all_snapshots(self):
        """Remove every snapshot from all canvases and reset the manager dialog."""
        self.canvas.clear_all_snapshots()
        self.single_det_canvas.clear_all_snapshots()
        if self.snapshot_manager is not None and self.snapshot_manager.isVisible():
            self.snapshot_manager.refresh([])

    def show_snapshot_manager(self):
        """Open (or bring to front) the snapshot manager dialog."""
        if self.snapshot_manager is None:
            self.snapshot_manager = SnapshotManagerDialog(self)
        self.snapshot_manager.refresh(self._get_snapshot_labels())
        self.snapshot_manager.show()
        self.snapshot_manager.raise_()
        self.snapshot_manager.activateWindow()

    def load_calibration_file(self):
        """Open a calib.yaml and load per-detector transmission coefficients"""
        if not _YAML_AVAILABLE:
            QMessageBox.critical(self, "Missing dependency",
                                 "PyYAML is not installed. Run: pip install pyyaml")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select calibration file", "",
            "YAML files (*.yaml *.yml);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, 'r') as fh:
                data = _yaml.safe_load(fh)
            if not isinstance(data, dict):
                raise ValueError("calib.yaml must be a YAML mapping")
            raw = data.get('detectors') or data.get('calibration')
            if not isinstance(raw, dict):
                raise ValueError("calib.yaml must contain a 'detectors' or 'calibration' mapping")
            self.calib_coefficients = {int(k): float(v) for k, v in raw.items()}
            n = len(self.calib_coefficients)
            self.calib_status_label.setText(f"Loaded {n} detector(s)\n{Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Calibration load error", str(e))
            self.calib_status_label.setText("Load failed — see console")

    def clear_calibration(self):
        """Remove loaded calibration coefficients"""
        self.calib_coefficients = {}
        self.calib_status_label.setText("No calibration loaded")

    def calibrate_from_buffer(self):
        """Calculate transmission calibration coefficients from current buffer results and save to temp_calibration.yaml"""
        if not _YAML_AVAILABLE:
            QMessageBox.critical(self, "Missing dependency",
                                 "PyYAML is not installed. Run: pip install pyyaml")
            return

        if self.last_results_df is None or self.last_results_df.empty:
            QMessageBox.warning(self, "No data", "No results in buffer. Run processing first.")
            return

        peak_no = self.calib_peakno_spin.value()
        set_plin = self.calib_plin_spin.value()
        set_beta = self.calib_beta_spin.value()
        set_phi = np.deg2rad(self.calib_phi_spin.value())
        int_method = self.calib_intmethod_combo.currentText()

        try:
            df = self.last_results_df.copy()

            # Add Angles column by mapping detector id -> angle (degrees) from polar canvas
            angles_deg = self.polar_canvas.angles_deg
            df["Angles"] = df["detector"].apply(
                lambda d: float(angles_deg[int(d)]) if int(d) < len(angles_deg) else 0.0
            )

            # Add dummy Photon Energy column (single energy in buffer)
            df["Photon Energy"] = 0

            calib = Calibrate(df)
            calib.transmission(
                peakNo=peak_no,
                setBeta=set_beta,
                setPhi=set_phi,
                setPlin=set_plin,
                intMethod=int_method,
            )

            # Build per-detector coefficient dict (mean over any duplicate entries)
            trans_df = calib.transmissionParam
            coeff_dict = (
                trans_df.groupby("detector")["Transmission Coefficient"]
                .mean()
                .to_dict()
            )
            coeff_dict = {int(k): float(v) for k, v in coeff_dict.items()}

            # Save to temp_calibration.yaml in the same format as calibration.yaml
            save_path = Path(__file__).parent / "temp_calibration.yaml"
            yaml_data = {
                "device": "temp_buffer_calibration",
                "calibration": {int(k): round(float(v), 6) for k, v in coeff_dict.items()},
            }
            with open(save_path, "w") as fh:
                _yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=True)

            # Load into active coefficients
            self.calib_coefficients = coeff_dict
            n = len(coeff_dict)
            self.calib_status_label.setText(
                f"Buffer calib: {n} detector(s)\nSaved → temp_calibration.yaml"
            )

        except Exception as e:
            QMessageBox.warning(self, "Calibration error", str(e))
            self.calib_status_label.setText("Calibration failed — see console")
            raise

    def update_polar_plot(self):
        """Update the polarization plot with current results"""
        if self.last_results_df is None or self.last_results_df.empty:
            return

        peak_no = self.polar_peak_spin.value()
        value_type = self.polar_value_combo.currentText()
        fit_beta = self.polar_fit_beta_radio.isChecked()
        beta = self.polar_beta_spin.value()
        set_plin = self.polar_plin_spin.value() if fit_beta else None
        fix_phi = self.polar_fix_phi_check.isChecked()
        set_phi = np.deg2rad(self.polar_phi_spin.value()) if fix_phi else None

        self.polar_canvas.update_polar_plot(
            self.last_results_df,
            peak_no=peak_no,
            value_type=value_type,
            beta=beta,
            setPlin=set_plin,
            fitBeta=fit_beta,
            setPhi=set_phi,
            fitPhi=not fix_phi
        )

    def update_results_table(self):
        """Update the results table with current results DataFrame"""
        if self.last_results_df is None or self.last_results_df.empty:
            self.results_table.setRowCount(0)
            return

        df = self.last_results_df
        self.results_table.setRowCount(len(df))

        # Column order matches the header
        columns = ["detector", "trainId", "pulseId", "peakNo",
                   "pos", "height", "width left", "width right", "fwhm area"]

        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col_name in enumerate(columns):
                if col_name in df.columns:
                    value = row[col_name]
                    # Format floats nicely
                    if isinstance(value, float):
                        text = f"{value:.4f}"
                    else:
                        text = str(value)
                    item = QTableWidgetItem(text)
                    self.results_table.setItem(row_idx, col_idx, item)

    def update_performance_display(self):
        """Update performance display"""
        stats = self.performance_monitor.get_stats()

        text = "Performance Metrics:\n"
        text += "=" * 40 + "\n"
        text += f"Iteration (avg):   {stats['iteration_avg']*1000:.1f} ms\n"
        text += f"Iteration (max):   {stats['iteration_max']*1000:.1f} ms\n"
        text += f"Load cycle:        {stats['load_cycle_avg']*1000:.1f} ms\n"
        text += f"Target:            {1000/self.update_rate_spin.value():.1f} ms\n"
        text += "\n"
        text += f"Load:              {stats['load_avg']*1000:.1f} ms\n"
        text += f"Process:           {stats['process_avg']*1000:.1f} ms\n"
        text += f"Plot prep:         {stats['plot_prep_avg']*1000:.1f} ms\n"
        text += f"Plot render (GUI): {stats['plot_render_avg']*1000:.1f} ms\n"

        self.perf_text.setPlainText(text)
        # Scroll to top to keep display stable
        self.perf_text.verticalScrollBar().setValue(0)

    def stop_processing(self):
        """Stop processing"""
        self.running = False
        self.main_timer.stop()

        # Stop data stream (DoocspieStream has a stop(); DataStreamSimulator does not)
        if self.data_simulator is not None and hasattr(self.data_simulator, 'stop'):
            self.data_simulator.stop()

        # Shutdown process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
            self.process_pool = None

        if self.plot_worker:
            self.plot_worker.stop()
        if self.plot_thread:
            self.plot_thread.join(timeout=1.0)

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()
