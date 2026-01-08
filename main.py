import sys
import numpy as np
import xarray as xr
from pathlib import Path
from collections import deque
from threading import Thread, Lock
import time
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional, Dict, List

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                              QDoubleSpinBox, QGroupBox, QGridLayout, QFileDialog,
                              QTextEdit, QCheckBox, QScrollArea)
from PyQt6.QtCore import QTimer, pyqtSignal, QObject, QThread, pyqtSlot
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ToFPipeline import NXSLoader, PeakFinder


@dataclass
class PlotData:
    """Pre-computed plot data for fast rendering"""
    detector_id: int
    samples: np.ndarray
    values: np.ndarray
    peak_positions: Optional[np.ndarray]
    is_enabled: bool
    has_data: bool


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    def __init__(self):
        self.iteration_times = deque(maxlen=100)
        self.load_times = deque(maxlen=100)
        self.process_times = deque(maxlen=100)
        self.plot_prep_times = deque(maxlen=100)
        self.plot_render_times = deque(maxlen=100)
        self.thread_times = {}
        self.lock = Lock()
        
    def record_iteration(self, duration):
        with self.lock:
            self.iteration_times.append(duration)
    
    def record_stage(self, stage, duration):
        with self.lock:
            if stage == 'load':
                self.load_times.append(duration)
            elif stage == 'process':
                self.process_times.append(duration)
            elif stage == 'plot_prep':
                self.plot_prep_times.append(duration)
            elif stage == 'plot_render':
                self.plot_render_times.append(duration)
    
    def record_thread(self, thread_id, duration):
        with self.lock:
            if thread_id not in self.thread_times:
                self.thread_times[thread_id] = deque(maxlen=100)
            self.thread_times[thread_id].append(duration)
    
    def get_stats(self):
        with self.lock:
            stats = {
                'iteration_avg': np.mean(self.iteration_times) if self.iteration_times else 0,
                'iteration_max': np.max(self.iteration_times) if self.iteration_times else 0,
                'load_avg': np.mean(self.load_times) if self.load_times else 0,
                'process_avg': np.mean(self.process_times) if self.process_times else 0,
                'plot_prep_avg': np.mean(self.plot_prep_times) if self.plot_prep_times else 0,
                'plot_render_avg': np.mean(self.plot_render_times) if self.plot_render_times else 0,
                'thread_stats': {}
            }
            
            for tid, times in self.thread_times.items():
                if times:
                    stats['thread_stats'][tid] = {
                        'avg': np.mean(times),
                        'max': np.max(times)
                    }
            
            return stats


class CircularBuffer:
    """Circular buffer for storing train data"""
    def __init__(self, size=10):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = Lock()
        
    def push(self, data):
        with self.lock:
            self.buffer.append(data)
    
    def get_all(self):
        """Get all items in buffer"""
        with self.lock:
            return list(self.buffer)
    
    def size_current(self):
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        with self.lock:
            self.buffer.clear()


class DataStreamSimulator:
    """Simulate data stream from .nxs file"""
    def __init__(self, nxs_path, run_numbers=None):
        self.loader = NXSLoader(nxs_path, run_numbers)
        self.loader.load()
        self.data = self.loader.data
        
        # Group by train
        self.trains = []
        train_ids = self.data.indexes['pulse'].get_level_values('trainId').unique()
        
        for train_id in train_ids:
            train_mask = self.data.indexes['pulse'].get_level_values('trainId') == train_id
            train_data = self.data.isel(pulse=train_mask)
            self.trains.append(train_data)
        
        self.current_index = 0
        
    def get_next_train(self):
        """Get next train (cycles through available trains)"""
        if len(self.trains) == 0:
            return None
            
        train = self.trains[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.trains)
        return train


class ProcessingWorker(QObject):
    """Worker for processing detector data"""
    finished = pyqtSignal(int, object)  # thread_id, (results, normalized_data, duration)
    
    def __init__(self, thread_id, detector_indices, config):
        super().__init__()
        self.thread_id = thread_id
        self.detector_indices = detector_indices
        self.config = config
        self.data_queue = Queue(maxsize=1)
        self.running = True
        
    def process(self, data, enabled_detectors):
        """Queue data for processing"""
        try:
            self.data_queue.put((data, enabled_detectors), block=False)
        except:
            pass  # Skip if queue is full
    
    def run(self):
        """Process data in loop"""
        while self.running:
            try:
                data, enabled_detectors = self.data_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Filter detector indices
                active_indices = [d for d in self.detector_indices if d in enabled_detectors]
                
                if not active_indices:
                    self.finished.emit(self.thread_id, (None, None, 0))
                    continue
                
                # Select detectors
                detector_data = data.sel(detector=active_indices)
                
                # Run peak finding
                pf = PeakFinder(detector_data, config=self.config)
                pf.normalize().process()
                
                normalized_data = pf.data
                results = pf.dataframe().results
                
                duration = time.time() - start_time
                self.finished.emit(self.thread_id, (results, normalized_data, duration))
                
            except Empty:
                continue
            except Exception as e:
                print(f"Processing thread {self.thread_id} error: {e}")
                
    def stop(self):
        self.running = False


class PlotPreparationWorker(QObject):
    """Worker for preparing plot data (runs in separate thread)"""
    plot_ready = pyqtSignal(list)  # List of PlotData objects
    
    def __init__(self, n_detectors):
        super().__init__()
        self.n_detectors = n_detectors
        self.data_queue = Queue(maxsize=1)
        self.running = True
        self.downsample = 2
        
    def set_downsample(self, value):
        self.downsample = value
    
    def prepare(self, data, normalized_data, results, enabled_detectors, last_train, last_pulse):
        """Queue data for plot preparation"""
        try:
            self.data_queue.put((data, normalized_data, results, enabled_detectors, last_train, last_pulse), block=False)
        except:
            pass  # Skip if queue full
    
    def run(self):
        """Prepare plot data in loop"""
        while self.running:
            try:
                packet = self.data_queue.get(timeout=0.1)
                data, normalized_data, results, enabled_detectors, last_train, last_pulse = packet
                
                plot_data_list = []
                available_detectors = data.coords['detector'].values
                
                for i in range(self.n_detectors):
                    # Check if detector exists in data
                    if i >= len(available_detectors):
                        plot_data_list.append(PlotData(
                            detector_id=i,
                            samples=np.array([]),
                            values=np.array([]),
                            peak_positions=None,
                            is_enabled=True,
                            has_data=False
                        ))
                        continue
                    
                    det_id = available_detectors[i]
                    
                    # Check if detector is enabled
                    if det_id not in enabled_detectors:
                        plot_data_list.append(PlotData(
                            detector_id=i,
                            samples=np.array([]),
                            values=np.array([]),
                            peak_positions=None,
                            is_enabled=False,
                            has_data=True
                        ))
                        continue
                    
                    try:
                        # Get trace (prefer normalized, fallback to raw)
                        trace = None
                        if normalized_data:
                            for worker_data in normalized_data.values():
                                try:
                                    trace = worker_data.sel(
                                        detector=det_id, 
                                        pulse={'trainId': last_train, 'pulseId': last_pulse}
                                    )
                                    break
                                except (KeyError, ValueError):
                                    continue
                        
                        # Fallback to raw data
                        if trace is None:
                            trace = data.sel(
                                detector=det_id, 
                                pulse={'trainId': last_train, 'pulseId': last_pulse}
                            )
                            trace_max = trace.max().values
                            if trace_max > 0:
                                trace = trace / trace_max
                        
                        # Downsample
                        samples = trace['sample'].values[::self.downsample]
                        values = trace.values[::self.downsample]
                        
                        # Get peaks
                        peak_positions = None
                        if results is not None and hasattr(results, 'empty') and not results.empty:
                            peaks = results[
                                (results['detector'] == det_id) &
                                (results['trainId'] == last_train) &
                                (results['pulseId'] == last_pulse)
                            ]
                            if not peaks.empty:
                                peak_positions = peaks[['pos', 'height']].values
                        
                        plot_data_list.append(PlotData(
                            detector_id=i,
                            samples=samples,
                            values=values,
                            peak_positions=peak_positions,
                            is_enabled=True,
                            has_data=True
                        ))
                        
                    except Exception as e:
                        print(f"Error preparing plot data for detector {i}: {e}")
                        plot_data_list.append(PlotData(
                            detector_id=i,
                            samples=np.array([]),
                            values=np.array([]),
                            peak_positions=None,
                            is_enabled=True,
                            has_data=False
                        ))
                
                # Emit prepared data
                self.plot_ready.emit(plot_data_list)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Plot preparation error: {e}")
    
    def stop(self):
        self.running = False


class FastMplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas optimized for fast updates using blitting"""
    def __init__(self, parent=None, width=5, height=4, dpi=100, n_detectors=16):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = []
        self.n_detectors = n_detectors
        
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(n_detectors)))
        n_rows = int(np.ceil(n_detectors / n_cols))
        
        # Create subplots
        for i in range(n_detectors):
            ax = self.fig.add_subplot(n_rows, n_cols, i+1)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Det {i}', fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_ylim([-0.1, 1.1])
            self.axes.append(ax)
        
        self.fig.tight_layout()
        super().__init__(self.fig)
        
        # Objects for fast updates
        self.lines = []
        self.scatters = []
        self.text_objects = []
        
        # Initialize plot objects
        for ax in self.axes:
            line, = ax.plot([], [], 'teal', linewidth=0.5, alpha=0.8)
            self.lines.append(line)
            
            scatter = ax.scatter([], [], color='red', s=20, zorder=5)
            self.scatters.append(scatter)
            
            text = ax.text(0.5, 0.5, '', ha='center', va='center', 
                          transform=ax.transAxes, fontsize=10, color='gray')
            text.set_visible(False)
            self.text_objects.append(text)
        
        # Background for blitting
        self.background = None
        
    def init_blit(self):
        """Initialize background for blitting"""
        self.draw()
        self.background = self.copy_from_bbox(self.fig.bbox)
    
    def fast_update(self, plot_data_list: List[PlotData]):
        """Fast update using blitting"""
        if self.background is None:
            self.init_blit()
        
        # Restore background
        self.restore_region(self.background)
        
        # Update each detector
        for i, plot_data in enumerate(plot_data_list):
            if i >= len(self.axes):
                continue
            
            ax = self.axes[i]
            line = self.lines[i]
            scatter = self.scatters[i]
            text = self.text_objects[i]
            
            # Handle different states
            if not plot_data.has_data:
                # No data available
                line.set_data([], [])
                scatter.set_offsets(np.empty((0, 2)))
                text.set_text('N/A')
                text.set_color('gray')
                text.set_visible(True)
                ax.set_facecolor('white')
                
            elif not plot_data.is_enabled:
                # Detector disabled
                line.set_data([], [])
                scatter.set_offsets(np.empty((0, 2)))
                text.set_text('OFF')
                text.set_color('red')
                text.set_visible(True)
                ax.set_facecolor('#ffeeee')
                
            else:
                # Update with data
                text.set_visible(False)
                ax.set_facecolor('white')
                
                if len(plot_data.samples) > 0:
                    line.set_data(plot_data.samples, plot_data.values)
                    ax.set_xlim([plot_data.samples[0], plot_data.samples[-1]])
                else:
                    line.set_data([], [])
                
                # Update peaks
                if plot_data.peak_positions is not None:
                    scatter.set_offsets(plot_data.peak_positions)
                else:
                    scatter.set_offsets(np.empty((0, 2)))
            
            # Redraw this axes
            ax.draw_artist(line)
            ax.draw_artist(scatter)
            ax.draw_artist(text)
        
        # Blit the updated region
        self.blit(self.fig.bbox)


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
        
        # Workers and threads
        self.processing_workers = []
        self.processing_threads = []
        self.plot_worker = None
        self.plot_thread = None
        
        # Pipeline stages
        self.stage_load = None
        self.stage_process = None
        self.stage_plot = None
        
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
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Right panel - Plots
        self.canvas = FastMplCanvas(self, width=10, height=8, dpi=100, n_detectors=16)
        main_layout.addWidget(self.canvas, stretch=3)
        
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File selection
        file_group = QGroupBox("Data Source")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        file_btn = QPushButton("Select .nxs File/Folder")
        file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(file_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Processing parameters
        param_group = QGroupBox("Processing Parameters")
        param_layout = QGridLayout()
        
        row = 0
        param_layout.addWidget(QLabel("Buffer Size:"), row, 0)
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(3, 100)
        self.buffer_size_spin.setValue(10)
        param_layout.addWidget(self.buffer_size_spin, row, 1)
        
        row += 1
        param_layout.addWidget(QLabel("Detectors/Thread:"), row, 0)
        self.det_per_thread_spin = QSpinBox()
        self.det_per_thread_spin.setRange(1, 16)
        self.det_per_thread_spin.setValue(4)
        param_layout.addWidget(self.det_per_thread_spin, row, 1)
        
        row += 1
        param_layout.addWidget(QLabel("Update Rate (Hz):"), row, 0)
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(1, 50)
        self.update_rate_spin.setValue(10)
        param_layout.addWidget(self.update_rate_spin, row, 1)
        
        row += 1
        param_layout.addWidget(QLabel("Plot Downsample:"), row, 0)
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 20)
        self.downsample_spin.setValue(2)
        self.downsample_spin.valueChanged.connect(self.on_downsample_changed)
        param_layout.addWidget(self.downsample_spin, row, 1)
        
        row += 1
        param_layout.addWidget(QLabel("Peak Threshold:"), row, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.1)
        param_layout.addWidget(self.threshold_spin, row, 1)
        
        row += 1
        param_layout.addWidget(QLabel("Number of Peaks:"), row, 0)
        self.peak_no_spin = QSpinBox()
        self.peak_no_spin.setRange(1, 20)
        self.peak_no_spin.setValue(8)
        param_layout.addWidget(self.peak_no_spin, row, 1)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
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
        for i, cb in enumerate(self.detector_checkboxes):
            cb.setChecked(True)
    
    def deselect_all_detectors(self):
        for i, cb in enumerate(self.detector_checkboxes):
            cb.setChecked(False)
    
    def on_downsample_changed(self, value):
        """Update downsample in plot worker"""
        if self.plot_worker:
            self.plot_worker.set_downsample(value)
    
    def select_file(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with .nxs Files")
        if folder:
            self.file_label.setText(folder)
            
    def start_processing(self):
        if not self.file_label.text() or self.file_label.text() == "No file selected":
            self.file_label.setText("Please select a file/folder first!")
            return
        
        # Load data
        try:
            self.data_simulator = DataStreamSimulator(Path(self.file_label.text()))
            self.n_detectors = len(self.data_simulator.data.coords['detector'])
        except Exception as e:
            self.file_label.setText(f"Error: {e}")
            return
        
        # Recreate canvas
        self.recreate_canvas()
        
        # Update detectors
        self.enabled_detectors = set(range(self.n_detectors))
        self.create_detector_checkboxes(self.n_detectors)
        
        # Initialize buffer
        self.circular_buffer = CircularBuffer(self.buffer_size_spin.value())
        
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
        layout = self.centralWidget().layout()
        layout.removeWidget(old_canvas)
        old_canvas.deleteLater()
        
        self.canvas = FastMplCanvas(self, width=10, height=8, dpi=100, n_detectors=self.n_detectors)
        layout.addWidget(self.canvas, stretch=3)
    
    def setup_workers(self):
        """Setup processing and plotting workers"""
        # Stop existing workers
        for worker in self.processing_workers:
            worker.stop()
        for thread in self.processing_threads:
            thread.join(timeout=1.0)
        
        if self.plot_worker:
            self.plot_worker.stop()
        if self.plot_thread:
            self.plot_thread.join(timeout=1.0)
        
        self.processing_workers = []
        self.processing_threads = []
        
        # Create processing workers
        det_per_thread = self.det_per_thread_spin.value()
        n_threads = int(np.ceil(self.n_detectors / det_per_thread))
        
        config = {
            'threshold': self.threshold_spin.value(),
            'peakNo': self.peak_no_spin.value(),
            'stackTrains': True,
            'stackPulses': False,
        }
        
        for i in range(n_threads):
            start_det = i * det_per_thread
            end_det = min((i + 1) * det_per_thread, self.n_detectors)
            detector_indices = list(range(start_det, end_det))
            
            worker = ProcessingWorker(i, detector_indices, config)
            worker.finished.connect(self.on_worker_finished)
            
            thread = Thread(target=worker.run, daemon=True)
            thread.start()
            
            self.processing_workers.append(worker)
            self.processing_threads.append(thread)
        
        # Create plot preparation worker
        self.plot_worker = PlotPreparationWorker(self.n_detectors)
        self.plot_worker.set_downsample(self.downsample_spin.value())
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
                
                # Send to plot worker (non-blocking)
                self.plot_worker.prepare(
                    data, normalized_data, results, 
                    self.enabled_detectors.copy(), 
                    last_train, last_pulse
                )
        
        # Shift pipeline
        self.stage_plot = self.stage_process
        self.stage_process = self.stage_load
        
        # Stage 1: Load
        load_start = time.time()
        new_train = self.data_simulator.get_next_train()
        if new_train is not None:
            self.circular_buffer.push(new_train)
            
            stacked_data = self.stack_buffer_data()
            if stacked_data is not None:
                for worker in self.processing_workers:
                    worker.process(stacked_data, self.enabled_detectors.copy())
                
                self.stage_load = {
                    'data': stacked_data,
                    'results': None,
                    'worker_count': len(self.processing_workers),
                    'process_start_time': time.time()
                }
        
        self.performance_monitor.record_stage('load', time.time() - load_start)
        self.performance_monitor.record_iteration(time.time() - iter_start)
    
    def stack_buffer_data(self):
        """Stack buffer data"""
        trains = self.circular_buffer.get_all()
        if not trains:
            return None
        return xr.concat(trains, dim='pulse')
    
    def on_worker_finished(self, thread_id, result):
        """Handle processing worker completion"""
        results, normalized_data, duration = result
        self.performance_monitor.record_thread(thread_id, duration)
        
        if normalized_data is not None and self.stage_load is not None:
            if 'normalized_data' not in self.stage_load:
                self.stage_load['normalized_data'] = {}
            self.stage_load['normalized_data'][thread_id] = normalized_data
        
        if self.stage_load is not None:
            if 'results' not in self.stage_load or self.stage_load['results'] is None:
                self.stage_load['results'] = []
                self.stage_load['finished_count'] = 0
            
            self.stage_load['finished_count'] = self.stage_load.get('finished_count', 0) + 1
            
            if results is not None:
                self.stage_load['results'].append(results)
            
            # All workers finished
            if self.stage_load['finished_count'] >= self.stage_load['worker_count']:
                if 'process_start_time' in self.stage_load:
                    process_duration = time.time() - self.stage_load['process_start_time']
                    self.performance_monitor.record_stage('process', process_duration)
                
                # Combine results
                import pandas as pd
                if self.stage_load['results']:
                    non_empty = [r for r in self.stage_load['results'] if not r.empty]
                    self.stage_load['results'] = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
                else:
                    self.stage_load['results'] = pd.DataFrame()
    
    @pyqtSlot(list)
    def on_plot_ready(self, plot_data_list: List[PlotData]):
        """Handle prepared plot data - update canvas (runs in GUI thread)"""
        render_start = time.time()
        
        # Fast update using blitting
        self.canvas.fast_update(plot_data_list)
        
        self.performance_monitor.record_stage('plot_render', time.time() - render_start)
    
    def update_performance_display(self):
        """Update performance display"""
        stats = self.performance_monitor.get_stats()
        
        text = "Performance Metrics:\n"
        text += "="*40 + "\n"
        text += f"Iteration (avg):   {stats['iteration_avg']*1000:.1f} ms\n"
        text += f"Iteration (max):   {stats['iteration_max']*1000:.1f} ms\n"
        text += f"Target:            {1000/self.update_rate_spin.value():.1f} ms\n"
        text += "\n"
        text += f"Load:              {stats['load_avg']*1000:.1f} ms\n"
        text += f"Process:           {stats['process_avg']*1000:.1f} ms\n"
        text += f"Plot prep:         {stats['plot_prep_avg']*1000:.1f} ms\n"
        text += f"Plot render (GUI): {stats['plot_render_avg']*1000:.1f} ms\n"
        text += "\n"
        
        if stats['thread_stats']:
            text += "Processing Threads:\n"
            for tid, tstats in stats['thread_stats'].items():
                text += f"  Thread {tid}: {tstats['avg']*1000:.1f} ms (max: {tstats['max']*1000:.1f} ms)\n"
        
        self.perf_text.setText(text)
    
    def stop_processing(self):
        """Stop processing"""
        self.running = False
        self.main_timer.stop()
        
        for worker in self.processing_workers:
            worker.stop()
        for thread in self.processing_threads:
            thread.join(timeout=1.0)
        
        if self.plot_worker:
            self.plot_worker.stop()
        if self.plot_thread:
            self.plot_thread.join(timeout=1.0)
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def closeEvent(self, event):
        self.stop_processing()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()