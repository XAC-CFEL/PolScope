import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from collections import deque
from threading import Lock, Thread
from queue import Queue, Empty

from ToFPipeline.ToFPipeline import NXSLoader


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


class _DataProxy:
    """Minimal proxy so MainWindow can read n_detectors before any train arrives."""

    def __init__(self, n_detectors: int):
        self.coords = {'detector': np.arange(n_detectors)}


class DoocspieStream:
    """Real-time data stream from DOOCS via doocspie TrainAbo.

    Runs a background daemon thread that blocks on the TrainAbo iterator and
    converts each TrainEvent into an xr.DataArray matching the format produced
    by NXSLoader / DataStreamSimulator:

        dims:  (detector, pulse, sample)
        pulse: pd.MultiIndex with levels (trainId, pulseId)

    Parameters
    ----------
    addresses : list of str
        DOOCS property addresses in detector order.
        Each property must deliver a 1-D (single-pulse) or 2-D
        (pulse-resolved, shape ``n_pulses × n_samples``) array per train.
    timeout_seconds : int
        Timeout for the TrainAbo synchronisation backend (default 10 s).
    """

    def __init__(self, addresses, timeout_seconds: int = 10):
        from doocspie.abo import TrainAbo

        # Accept either a list or a keyed dict (e.g. {1: addr1, 2: addr2, ...})
        if isinstance(addresses, dict):
            self._addresses = [addresses[k] for k in sorted(addresses)]
        else:
            self._addresses = list(addresses)
        self.n_detectors = len(self._addresses)
        # Expose .data so MainWindow can call len(stream.data.coords['detector'])
        self.data = _DataProxy(self.n_detectors)

        self._train_abo = TrainAbo(timeout_seconds=timeout_seconds)
        for i, addr in enumerate(self._addresses):
            self._train_abo.add(addr, label=f'det_{i}')

        self._queue: Queue = Queue(maxsize=4)
        self._running = True
        self._thread = Thread(target=self._fetch_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def _fetch_loop(self):
        """Background thread: blocks on next train, converts, enqueues."""
        for train_event in self._train_abo:
            if not self._running:
                break
            try:
                da = self._convert(train_event)
                # Drop oldest if consumer is slow (keep latency low)
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                self._queue.put_nowait(da)
            except Exception as e:
                print(f"DoocspieStream conversion error: {e}")

    def _convert(self, train_event) -> xr.DataArray:
        """Convert a doocspie TrainEvent to an xr.DataArray.

        Each DOOCS readout is expected to have ``.data`` that is either:
        - 1-D ``(n_samples,)``  → treated as a single-pulse train
        - 2-D ``(n_pulses, n_samples)``  → pulse-resolved train
        """
        train_id = train_event.id
        detector_arrays = []
        n_pulses = None
        n_samples = None

        for i in range(self.n_detectors):
            readout = train_event.get(f'det_{i}')
            arr = np.asarray(readout.data, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]   # (1, n_samples)
            if n_pulses is None:
                n_pulses, n_samples = arr.shape
            detector_arrays.append(arr)   # each (n_pulses, n_samples)

        # Stack → (n_detectors, n_pulses, n_samples)
        stacked = np.stack(detector_arrays, axis=0)

        pulse_index = pd.MultiIndex.from_arrays(
            [[train_id] * n_pulses, list(range(n_pulses))],
            names=['trainId', 'pulseId']
        )

        return xr.DataArray(
            stacked,
            dims=['detector', 'pulse', 'sample'],
            coords={
                'detector': np.arange(self.n_detectors),
                'pulse': pulse_index,
                'sample': np.arange(n_samples, dtype=np.int64),
            }
        )

    # ------------------------------------------------------------------
    def get_next_train(self):
        """Non-blocking: return the next available train DataArray, or None."""
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        """Signal the background thread to terminate after the current train."""
        self._running = False
