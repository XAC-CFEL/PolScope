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
from concurrent.futures import ProcessPoolExecutor, as_completed

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                              QDoubleSpinBox, QGroupBox, QGridLayout, QFileDialog,
                              QTextEdit, QCheckBox, QScrollArea, QTabWidget,
                              QTableWidget, QTableWidgetItem, QHeaderView, QComboBox)
from PyQt6.QtCore import QTimer, pyqtSignal, QObject, QThread, pyqtSlot
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ToFPipeline import NXSLoader, PeakFinder, GlobalConfig, polarization_model
from scipy.optimize import curve_fit


# Wong color palette (colorblind-friendly)
WONG_COLORS = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7'
}

# Assign colors for specific purposes
COLOR_TRACE = WONG_COLORS['blue']           # Main trace line
COLOR_PEAK = WONG_COLORS['vermillion']      # Peak markers
COLOR_FWHM = WONG_COLORS['orange']          # FWHM lines
COLOR_DATA = WONG_COLORS['bluish_green']    # Data points in polar plot
COLOR_FIT = WONG_COLORS['blue']             # Fit line in polar plot
COLOR_PHI = WONG_COLORS['orange']           # Phi angle lines
COLOR_DISABLED = WONG_COLORS['vermillion']  # Disabled detector indicator
COLOR_GRAY = '#808080'                      # Gray for N/A text


# Load global configuration from config.yaml
config_path = Path(__file__).parent / "config.yaml"
if config_path.exists():
    GlobalConfig.load(config_path)


# Top-level function for multiprocessing (must be picklable)
def process_detector_chunk(args):
    """Process detector data in a separate process - data is already normalized and ROI-applied"""
    detector_data, config, worker_id = args
    try:
        # detector_data is passed directly (xarray DataArrays are picklable)
        # Note: Data is already normalized AND ROI-applied before chunking
        pf = PeakFinder(detector_data, config=config)
        # stack() averages across trains/pulses based on config
        pf.stack()
        
        # ROI is already applied in stack_buffer_data() for early data reduction
        # No need to apply it again here
        
        # Apply smoothing if window size > 1
        smooth_window = config.get('smoothWindow', 1)
        if smooth_window > 1:
            pf.smooth(windowSize=smooth_window)
        
        # Find peaks (ROI already applied, so pass [None, None])
        pf.process(roi=[None, None])
        
        # Return stacked data and results directly (both are picklable)
        stacked_data = pf.data
        results = pf.dataframe().results
        
        return (worker_id, results, stacked_data)
    except Exception as e:
        print(f"Process {worker_id} error: {e}")
        import traceback
        traceback.print_exc()
        return (worker_id, None, None)


@dataclass
class PlotData:
    """Pre-computed plot data for fast rendering"""
    detector_id: int
    samples: np.ndarray
    values: np.ndarray
    peak_positions: Optional[np.ndarray]  # [[pos, height], ...]
    fwhm_lines: Optional[np.ndarray]       # [[pos, widthL, widthR, height/2], ...]
    is_enabled: bool
    has_data: bool


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    def __init__(self):
        self.iteration_times = deque(maxlen=10)
        self.load_times = deque(maxlen=10)
        self.load_cycle_times = deque(maxlen=10)
        self.process_times = deque(maxlen=10)
        self.plot_prep_times = deque(maxlen=10)
        self.plot_render_times = deque(maxlen=10)
        self.thread_times = {}
        self.lock = Lock()
        
    def record_iteration(self, duration):
        with self.lock:
            self.iteration_times.append(duration)
    
    def record_stage(self, stage, duration):
        with self.lock:
            if stage == 'load':
                self.load_times.append(duration)
            elif stage == 'load_cycle':
                self.load_cycle_times.append(duration)
            elif stage == 'process':
                self.process_times.append(duration)
            elif stage == 'plot_prep':
                self.plot_prep_times.append(duration)
            elif stage == 'plot_render':
                self.plot_render_times.append(duration)
    
    def record_thread(self, thread_id, duration):
        with self.lock:
            if thread_id not in self.thread_times:
                self.thread_times[thread_id] = deque(maxlen=10)
            self.thread_times[thread_id].append(duration)
    
    def get_stats(self):
        with self.lock:
            stats = {
                'iteration_avg': np.mean(self.iteration_times) if self.iteration_times else 0,
                'iteration_max': np.max(self.iteration_times) if self.iteration_times else 0,
                'load_avg': np.mean(self.load_times) if self.load_times else 0,
                'load_cycle_avg': np.mean(self.load_cycle_times) if self.load_cycle_times else 0,
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


class PlotPreparationWorker(QObject):
    """Worker for preparing plot data (runs in separate thread)"""
    plot_ready = pyqtSignal(list)  # List of PlotData objects
    
    def __init__(self, n_detectors):
        super().__init__()
        self.n_detectors = n_detectors
        self.data_queue = Queue(maxsize=1)
        self.running = True
        self.downsample = 2
        self.roi = [0, 10000]  # Default ROI
        
    def set_downsample(self, value):
        self.downsample = value
    
    def set_roi(self, roi_start, roi_end):
        self.roi = [roi_start, roi_end]
    
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
                            fwhm_lines=None,
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
                            fwhm_lines=None,
                            is_enabled=False,
                            has_data=True
                        ))
                        continue
                    
                    try:
                        # Get trace from normalized (smoothed/stacked) data only
                        # This ensures trace matches the peak finding results
                        trace = None
                        if normalized_data:
                            for worker_data in normalized_data.values():
                                try:
                                    # Check if this worker has this detector
                                    if det_id not in worker_data.coords['detector'].values:
                                        continue
                                    
                                    # For stacked data, just take the first pulse
                                    trace = worker_data.sel(detector=det_id).isel(pulse=0)
                                    break
                                except (KeyError, ValueError, IndexError) as e:
                                    continue
                        
                        # If trace not found in normalized data, skip this detector
                        # (Don't fallback to raw data as it would have different sample coordinates
                        # due to smoothing trim, causing peaks to not align with trace)
                        if trace is None:
                            plot_data_list.append(PlotData(
                                detector_id=i,
                                samples=np.array([]),
                                values=np.array([]),
                                peak_positions=None,
                                fwhm_lines=None,
                                is_enabled=True,
                                has_data=False
                            ))
                            continue
                        
                        # Downsample the smoothed trace
                        # Note: smoothing already trimmed sample coordinates, so they match peak positions
                        samples = trace['sample'].values[::self.downsample]
                        values = trace.values[::self.downsample]
                        
                        
                        # Get peaks - filter only by detector (independent per detector)
                        # For averaged/rolling buffer data, we don't filter by trainId/pulseId
                        peak_positions = None
                        fwhm_lines = None
                        if results is not None and hasattr(results, 'empty') and not results.empty:
                            peaks = results[results['detector'] == det_id]
                            if not peaks.empty:
                                peak_positions = peaks[['pos', 'height']].values
                                # Extract FWHM line data: [pos, width left, width right, height/2]
                                fwhm_lines = peaks[['pos', 'width left', 'width right', 'height']].values.copy()
                                fwhm_lines[:, 3] = fwhm_lines[:, 3] / 2  # Convert height to half height
                        
                        plot_data_list.append(PlotData(
                            detector_id=i,
                            samples=samples,
                            values=values,
                            peak_positions=peak_positions,
                            fwhm_lines=fwhm_lines,
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
                            fwhm_lines=None,
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
            ax.set_xlim([0, 1000])  # Initial range, will be updated with real data
            # Format x-axis to show sample coordinates as plain numbers
            ax.ticklabel_format(style='plain', axis='x', useOffset=False)
            ax.xaxis.get_major_formatter().set_scientific(False)
            self.axes.append(ax)
        
        self.fig.tight_layout()
        super().__init__(self.fig)
        
        # Objects for fast updates
        self.lines = []
        self.scatters = []
        self.text_objects = []
        self.fwhm_lines = []  # LineCollection for FWHM horizontal lines
        
        # Initialize plot objects
        for ax in self.axes:
            line, = ax.plot([], [], color=COLOR_TRACE, linewidth=0.5, alpha=0.8)
            self.lines.append(line)
            
            scatter = ax.scatter([], [], color=COLOR_PEAK, s=20, zorder=5)
            self.scatters.append(scatter)
            
            text = ax.text(0.5, 0.5, '', ha='center', va='center', 
                          transform=ax.transAxes, fontsize=10, color=COLOR_GRAY)
            text.set_visible(False)
            self.text_objects.append(text)
            
            # LineCollection for FWHM lines (horizontal lines at half height)
            from matplotlib.collections import LineCollection
            fwhm_lc = LineCollection([], colors=COLOR_FWHM, linewidths=1.5, zorder=4)
            ax.add_collection(fwhm_lc)
            self.fwhm_lines.append(fwhm_lc)
        
        # Background for blitting
        self.background = None
        
    def resizeEvent(self, event):
        """Handle resize events to redraw plots properly"""
        super().resizeEvent(event)
        # Reset background on resize so blitting works correctly
        self.background = None
        self.fig.tight_layout()
        self.draw_idle()
        
    def init_blit(self):
        """Initialize background for blitting"""
        self.draw()
        self.background = self.copy_from_bbox(self.fig.bbox)
    
    def fast_update(self, plot_data_list: List[PlotData]):
        """Fast update using blitting"""
        # First, update xlim for all axes that have data and check if any changed
        xlim_changed = False
        if self.background is not None:
            for i, plot_data in enumerate(plot_data_list):
                if i >= len(self.axes):
                    continue
                if plot_data.has_data and plot_data.is_enabled and len(plot_data.samples) > 0:
                    ax = self.axes[i]
                    current_xlim = ax.get_xlim()
                    new_xmin, new_xmax = float(plot_data.samples[0]), float(plot_data.samples[-1])
                    # Set the new xlim now
                    if new_xmax > new_xmin:
                        ax.set_xlim([new_xmin, new_xmax])
                    # Check if limits changed significantly (more than 1% difference)
                    if (abs(current_xlim[0] - new_xmin) > abs(new_xmin) * 0.01 or 
                        abs(current_xlim[1] - new_xmax) > abs(new_xmax) * 0.01):
                        xlim_changed = True
        
        # If xlim changed or no background, need full redraw to update axis labels
        if self.background is None or xlim_changed:
            # Clear old data before taking new background snapshot
            for line in self.lines:
                line.set_data([], [])
            for scatter in self.scatters:
                scatter.set_offsets(np.empty((0, 2)))
            for fwhm_lc in self.fwhm_lines:
                fwhm_lc.set_segments([])
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
            fwhm_lc = self.fwhm_lines[i]
            
            # Handle different states
            if not plot_data.has_data:
                # No data available
                line.set_data([], [])
                scatter.set_offsets(np.empty((0, 2)))
                fwhm_lc.set_segments([])
                text.set_text('N/A')
                text.set_color(COLOR_GRAY)
                text.set_visible(True)
                ax.set_facecolor('white')
                
            elif not plot_data.is_enabled:
                # Detector disabled
                line.set_data([], [])
                scatter.set_offsets(np.empty((0, 2)))
                fwhm_lc.set_segments([])
                text.set_text('OFF')
                text.set_color(COLOR_DISABLED)
                text.set_visible(True)
                ax.set_facecolor('#ffeeee')
                
            else:
                # Update with data
                text.set_visible(False)
                ax.set_facecolor('white')
                
                if len(plot_data.samples) > 0:
                    line.set_data(plot_data.samples, plot_data.values)
                    # xlim already set in the pre-check above
                    ax.relim()  # Recalculate data limits
                    ax.autoscale_view(scalex=False, scaley=False)  # Don't autoscale, use our limits
                else:
                    line.set_data([], [])
                
                # Update peaks
                if plot_data.peak_positions is not None:
                    scatter.set_offsets(plot_data.peak_positions)
                else:
                    scatter.set_offsets(np.empty((0, 2)))
                
                # Update FWHM lines
                if plot_data.fwhm_lines is not None and len(plot_data.fwhm_lines) > 0:
                    # fwhm_lines is [[pos, widthL, widthR, half_height], ...]
                    segments = []
                    for fwhm in plot_data.fwhm_lines:
                        pos, widthL, widthR, half_height = fwhm
                        # Draw horizontal line from pos+widthL to pos+widthR at half_height
                        x_start = pos + widthL
                        x_end = pos + widthR
                        segments.append([(x_start, half_height), (x_end, half_height)])
                    fwhm_lc.set_segments(segments)
                else:
                    fwhm_lc.set_segments([])
            
            # Redraw this axes
            ax.draw_artist(line)
            ax.draw_artist(scatter)
            ax.draw_artist(fwhm_lc)
            ax.draw_artist(text)
        
        # Blit the updated region
        self.blit(self.fig.bbox)


class PolarPlotCanvas(FigureCanvasQTAgg):
    """Polar plot canvas with blitting for fast polarization visualization"""
    
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='polar')
        
        # Configure polar plot appearance
        self.ax.set_theta_zero_location("E")  # 0° at East (right)
        self.ax.set_theta_direction(1)        # Counter-clockwise
        self.ax.set_yticks([])
        self.ax.set_title("Polarization", fontsize=12)
        
        super().__init__(self.fig)
        
        # Plot objects for fast updates
        self.data_line, = self.ax.plot([], [], 'o', markersize=8, color=COLOR_DATA, label='Data')
        self.fit_line, = self.ax.plot([], [], '-', linewidth=2, color=COLOR_FIT, label='Fit')
        self.phi_line1, = self.ax.plot([], [], '-', linewidth=2, color=COLOR_PHI, alpha=0.7)
        self.phi_line2, = self.ax.plot([], [], '-', linewidth=2, color=COLOR_PHI, alpha=0.7)
        
        # Text for fit parameters
        self.fit_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      fontsize=10, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax.legend(loc='lower right', fontsize=8)
        self.fig.tight_layout()
        
        # Background for blitting
        self.background = None
        
        # Angles from config (degrees, will be converted to radians)
        nxs_config = GlobalConfig.get_for_class('NXSLoader')
        self.angles_deg = np.array(nxs_config.get('angles', 
                                   np.linspace(0, 337.5, 16).tolist()) if nxs_config else
                                   np.linspace(0, 337.5, 16).tolist())
        self.angles_rad = np.deg2rad(self.angles_deg)
        
    def resizeEvent(self, event):
        """Handle resize events to redraw plots properly"""
        super().resizeEvent(event)
        # Reset background on resize so blitting works correctly
        self.background = None
        self.fig.tight_layout()
        self.draw_idle()
        
        # Store last fit results
        self.last_fit_params = None
        
    def init_blit(self):
        """Initialize background for blitting"""
        self.draw()
        self.background = self.copy_from_bbox(self.fig.bbox)
    
    def update_polar_plot(self, results_df, peak_no=0, value_type='height', beta=2.0):
        """
        Update the polar plot with new data.
        
        Parameters:
            results_df: DataFrame with peak results
            peak_no: Which peak number to plot (0-indexed)
            value_type: 'height' or 'fwhm area'
            beta: Beta parameter for polarization model
        """
        if self.background is None:
            self.init_blit()
        
        # Restore background
        self.restore_region(self.background)
        
        # Clear previous data
        self.data_line.set_data([], [])
        self.fit_line.set_data([], [])
        self.phi_line1.set_data([], [])
        self.phi_line2.set_data([], [])
        self.fit_text.set_text('')
        self.last_fit_params = None
        
        if results_df is None or results_df.empty:
            self._redraw_artists()
            return
        
        # Filter by peak number
        peak_data = results_df[results_df['peakNo'] == peak_no]
        
        if peak_data.empty:
            self._redraw_artists()
            return
        
        # Get values for each detector
        # Group by detector and take mean if multiple pulses/trains
        detector_values = peak_data.groupby('detector')[value_type].mean()
        
        if detector_values.empty:
            self._redraw_artists()
            return
        
        # Get angles for available detectors
        available_detectors = detector_values.index.values
        
        # Make sure we have valid detector indices
        valid_mask = available_detectors < len(self.angles_rad)
        available_detectors = available_detectors[valid_mask]
        
        if len(available_detectors) == 0:
            self._redraw_artists()
            return
        
        theta = self.angles_rad[available_detectors]
        r_values = detector_values.loc[available_detectors].values
        
        # Update data points
        self.data_line.set_data(theta, r_values)
        
        # Update radial limit based on data
        r_max = np.nanmax(r_values) * 1.2 if len(r_values) > 0 else 1.0
        self.ax.set_rlim(0, r_max)
        
        # Fit polarization model if we have enough data points
        if len(theta) >= 3:
            try:
                self._fit_and_plot(theta, r_values, beta, r_max)
            except Exception as e:
                print(f"Fit error: {e}")
                self.fit_text.set_text(f"Fit failed: {str(e)[:30]}")
        
        self._redraw_artists()
    
    def _fit_and_plot(self, theta, r_values, beta, r_max):
        """Fit the polarization model and update plot"""
        # Define model with fixed beta
        def model(theta, Plin, phi, scale):
            return polarization_model(theta, Plin=Plin, phi=phi, beta2=beta, scale=scale)
        
        # Initial guess
        scale_guess = np.mean(r_values)
        initial_guess = [0.2, 0.0, scale_guess]
        bounds = ([0.0, -np.pi, 0], [2.0, np.pi, np.inf])
        
        # Fit
        popt, pcov = curve_fit(
            model, theta, r_values,
            p0=initial_guess,
            bounds=bounds,
            method='trf',
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            maxfev=5000
        )
        
        Plin_fit, phi_fit, scale_fit = popt
        self.last_fit_params = {
            'Plin': Plin_fit,
            'phi': phi_fit,
            'scale': scale_fit,
            'beta': beta,
            'pcov': pcov
        }
        
        # Generate smooth fit curve
        theta_fit = np.linspace(0, 2 * np.pi, 360)
        r_fit = model(theta_fit, Plin_fit, phi_fit, scale_fit)
        
        # Update fit line
        self.fit_line.set_data(theta_fit, r_fit)
        
        # Update phi indicator lines if polarization is significant
        if Plin_fit > 0.015:
            self.phi_line1.set_data([phi_fit, phi_fit], [0, r_max])
            self.phi_line2.set_data([phi_fit + np.pi, phi_fit + np.pi], [0, r_max])
        else:
            self.phi_line1.set_data([], [])
            self.phi_line2.set_data([], [])
        
        # Update fit text
        phi_deg = np.rad2deg(phi_fit) % 360
        fit_info = (f"Plin: {Plin_fit:.4f}\n"
                   f"φ: {phi_deg:.1f}°\n"
                   f"β: {beta:.3f}\n"
                   f"Scale: {scale_fit:.4f}")
        self.fit_text.set_text(fit_info)
    
    def _redraw_artists(self):
        """Redraw all artists and blit"""
        self.ax.draw_artist(self.data_line)
        self.ax.draw_artist(self.fit_line)
        self.ax.draw_artist(self.phi_line1)
        self.ax.draw_artist(self.phi_line2)
        self.ax.draw_artist(self.fit_text)
        self.blit(self.fig.bbox)
    
    def force_full_redraw(self):
        """Force a full redraw (needed when r-limits change significantly)"""
        self.draw()
        self.background = self.copy_from_bbox(self.fig.bbox)


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
        self.buffer_size_spin.setRange(1, 100)
        self.buffer_size_spin.setValue(1)
        param_layout.addWidget(self.buffer_size_spin, row, 1)
        
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
        self.threshold_spin.editingFinished.connect(self.update_processing_config)
        param_layout.addWidget(self.threshold_spin, row, 1)
        
        row += 1
        param_layout.addWidget(QLabel("Number of Peaks:"), row, 0)
        self.peak_no_spin = QSpinBox()
        self.peak_no_spin.setRange(1, 20)
        self.peak_no_spin.setValue(1)
        self.peak_no_spin.editingFinished.connect(self.update_processing_config)
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
        param_layout.addWidget(QLabel("Smooth Window:"), row, 0)
        self.smooth_window_spin = QSpinBox()
        self.smooth_window_spin.setRange(1, 50)
        self.smooth_window_spin.setValue(5)
        self.smooth_window_spin.setToolTip("Window size for rolling average smoothing (1 = no smoothing)")
        self.smooth_window_spin.editingFinished.connect(self.update_processing_config)
        param_layout.addWidget(self.smooth_window_spin, row, 1)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
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
        
        polar_group.setLayout(polar_layout)
        layout.addWidget(polar_group)
        
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
    
    def update_processing_config(self):
        """Update processing config when spinbox values change (on Enter/focus loss)"""
        if hasattr(self, 'processing_config'):
            self.processing_config['threshold'] = self.threshold_spin.value()
            self.processing_config['peakNo'] = self.peak_no_spin.value()
            self.processing_config['roi'] = [self.roi_start_spin.value(), self.roi_end_spin.value()]
            self.processing_config['smoothWindow'] = self.smooth_window_spin.value()
            
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
                  f"smoothWindow={self.processing_config['smoothWindow']}")
    
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
        
        # Remove old canvas from tab widget
        tab_index = self.tab_widget.indexOf(old_canvas)
        if tab_index >= 0:
            self.tab_widget.removeTab(tab_index)
        old_canvas.deleteLater()
        
        # Create new canvas and add back to tab widget at the same position
        self.canvas = FastMplCanvas(self, width=10, height=8, dpi=100, n_detectors=self.n_detectors)
        self.tab_widget.insertTab(0, self.canvas, "Plots")
        self.tab_widget.setCurrentIndex(0)  # Make sure Plots tab is active
    
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
        # Other parameters (like stackTrains, stackPulses, etc.) come from config.yaml
        self.processing_config = {
            'threshold': self.threshold_spin.value(),
            'peakNo': self.peak_no_spin.value(),
            'roi': [self.roi_start_spin.value(), self.roi_end_spin.value()],
            'smoothWindow': self.smooth_window_spin.value(),
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
            plot_stage_start = time.time()
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
        # Always load new data (don't wait for previous processing to complete)
        # This enables true parallelism with multiple processing jobs in flight
        load_start = time.time()
        new_train = self.data_simulator.get_next_train()
        if new_train is not None:
                self.circular_buffer.push(new_train)
                
                stacked_data = self.stack_buffer_data()
                if stacked_data is not None:
                    # Normalize the data BEFORE chunking to ensure consistent scaling
                    # This way all detectors are normalized to the global maximum
                    norm_start = time.time()
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
        
        import pandas as pd
        
        # Check each future
        completed = []
        for future in self.processing_futures:
            if future.done():
                completed.append(future)
                try:
                    worker_id, results, normalized_data = future.result()
                    
                    # Record timing
                    self.performance_monitor.record_thread(worker_id, 0)  # Can't get exact timing from process
                    
                    # Store normalized data (already an xarray DataArray)
                    if normalized_data is not None:
                        self.stage_load['normalized_data'][worker_id] = normalized_data
                    
                    # Store results - peak positions are already in correct sample coordinates
                    # (ROI is applied early in stack_buffer_data, and PeakFinder converts
                    # array indices to actual sample coordinates)
                    if results is not None and isinstance(self.stage_load['results'], list):
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
                non_empty = [r for r in self.stage_load['results'] if not r.empty]
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
            self.canvas.fast_update(plot_data_list)
            self.performance_monitor.record_stage('plot_render', time.time() - render_start)
        else:
            self.plots_need_update = True
        
        # Update polar plot if it's visible or flag for update
        if self.tab_widget.currentIndex() == 2:  # Polarization tab
            self.update_polar_plot()
        else:
            self.polar_needs_update = True
    
    def on_tab_changed(self, index):
        """Handle tab changes - update views when tabs are selected"""
        if index == 0 and self.plots_need_update:  # Plots tab
            if self.last_plot_data is not None:
                self.canvas.fast_update(self.last_plot_data)
            self.plots_need_update = False
        elif index == 1 and self.results_need_update:  # Results tab
            self.update_results_table()
            self.results_need_update = False
        elif index == 2 and self.polar_needs_update:  # Polarization tab
            self.update_polar_plot()
            self.polar_needs_update = False
    
    def on_polar_param_changed(self):
        """Handle changes to polar plot parameters"""
        # Update polar plot immediately if visible
        if self.tab_widget.currentIndex() == 2:
            self.update_polar_plot()
    
    def update_polar_plot(self):
        """Update the polarization plot with current results"""
        if self.last_results_df is None or self.last_results_df.empty:
            return
        
        peak_no = self.polar_peak_spin.value()
        value_type = self.polar_value_combo.currentText()
        beta = self.polar_beta_spin.value()
        
        self.polar_canvas.update_polar_plot(
            self.last_results_df,
            peak_no=peak_no,
            value_type=value_type,
            beta=beta
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
        text += "="*40 + "\n"
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


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()