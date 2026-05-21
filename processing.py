import math
import numpy as np
import traceback
from collections import deque
from threading import Lock
from queue import Queue, Empty

from PyQt6.QtCore import pyqtSignal, QObject

from ToFPipeline.ToFPipeline import PeakFinder
from models import PlotData


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
            pf.smooth(windowSize=smooth_window).smooth(windowSize=smooth_window)

        # Find peaks within the peak-specific ROI (separate from the loader ROI).
        # Subtract loader roi_start so the indices map onto the PeakFinder's 0-based slice.
        roi_start = config.get('roi', [None, None])[0] or 0
        raw_peak_roi = config.get('peakfinder_roi', [None, None])
        peak_roi = [
            (raw_peak_roi[0] - roi_start) if raw_peak_roi[0] is not None else None,
            (raw_peak_roi[1] - roi_start) if raw_peak_roi[1] is not None else None,
        ]
        pf.process()

        # Offset peak positions back to original sample coordinate space.
        results = pf.results
        print(results.results)
        if results is not None and hasattr(results, 'empty') and not results.empty and roi_start != 0:
            print("resuklts not empty, applying ROI offset to positions")
            if "pos" in results.columns:
                print("found peaks, applying ROI offset to positions")
                results["pos"] = results["pos"] + roi_start

        stacked_data = pf.data

        return (worker_id, results, stacked_data)
    except Exception as e:
        print(f"Process {worker_id} error: {e}")
        traceback.print_exc()
        return (worker_id, None, None)


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
            self.data_queue.put(
                (data, normalized_data, results, enabled_detectors, last_train, last_pulse),
                block=False
            )
        except Exception:
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
                                except (KeyError, ValueError, IndexError):
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

                        # Full-resolution arrays (used for baseline computation)
                        samples_all = trace['sample'].values
                        values_all = trace.values

                        # Downsampled arrays (used for display)
                        samples = samples_all[::self.downsample]
                        values = values_all[::self.downsample]

                        # Get peaks - filter only by detector (independent per detector)
                        # For averaged/rolling buffer data, we don't filter by trainId/pulseId
                        peak_positions = None
                        fwhm_lines = None
                        baseline_data = None
                        if results is not None and hasattr(results, 'empty') and not results.empty:
                            peaks = results[results['detector'] == det_id]
                            if not peaks.empty:
                                peak_positions = peaks[['pos', 'height']].values
                                # Extract FWHM line data: [pos, width left, width right, height/2]
                                fwhm_lines = peaks[['pos', 'width left', 'width right', 'height']].values.copy()
                                fwhm_lines[:, 3] = fwhm_lines[:, 3] / 2  # Convert height to half height

                                # Build per-peak baseline data when available
                                if 'baseline left' in peaks.columns and 'baseline right' in peaks.columns:
                                    baseline_data = []
                                    for _, peak_row in peaks.iterrows():
                                        bl = peak_row['baseline left']
                                        br = peak_row['baseline right']
                                        # Skip invalid baseline values
                                        if bl is False or br is False:
                                            continue
                                        try:
                                            if math.isnan(float(bl)) or math.isnan(float(br)):
                                                continue
                                        except (TypeError, ValueError):
                                            continue
                                        bl_idx = int(np.searchsorted(samples_all, float(bl)))
                                        br_idx = int(np.searchsorted(samples_all, float(br), side='right')) - 1
                                        bl_idx = max(0, min(bl_idx, len(samples_all) - 1))
                                        br_idx = max(0, min(br_idx, len(samples_all) - 1))
                                        if br_idx <= bl_idx:
                                            continue
                                        yL = float(values_all[bl_idx])
                                        yR = float(values_all[br_idx])
                                        xL = float(samples_all[bl_idx])
                                        xR = float(samples_all[br_idx])
                                        slope = (yR - yL) / (xR - xL) if xR != xL else 0.0
                                        offset = yL - slope * xL
                                        adj_x = samples_all[bl_idx:br_idx + 1]
                                        adj_y = values_all[bl_idx:br_idx + 1] - (slope * adj_x + offset)
                                        baseline_data.append({
                                            'bl_x': [xL, xR],
                                            'bl_y': [yL, yR],
                                            'adj_x': adj_x.copy(),
                                            'adj_y': adj_y.copy(),
                                        })

                        plot_data_list.append(PlotData(
                            detector_id=i,
                            samples=samples,
                            values=values,
                            peak_positions=peak_positions,
                            fwhm_lines=fwhm_lines,
                            is_enabled=True,
                            has_data=True,
                            baseline_data=baseline_data,
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
