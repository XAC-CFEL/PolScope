import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from multiprocessing import Process, Queue, Lock, Event
from multiprocessing.queues import Empty

from ToFPipeline.ToFPipeline import NXSLoader


# ---------------------------------------------------------------------------
# Module-level helpers – must live at module scope so multiprocessing (spawn)
# can pickle them when starting the worker process on Windows.
# ---------------------------------------------------------------------------

def _convert_train_event(train_event, addresses: list) -> xr.DataArray:
    """Convert a doocspie TrainEvent to an xr.DataArray."""
    train_id = train_event.id
    n_detectors = len(addresses)
    detector_arrays = []
    n_pulses = None
    n_samples = None

    for i, addr in enumerate(addresses):
        readout = train_event.get(addr)
        arr = np.asarray(readout.data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]   # (1, n_samples)
        if n_pulses is None:
            n_pulses, n_samples = arr.shape
        detector_arrays.append(arr)

    stacked = np.stack(detector_arrays, axis=0)  # (n_detectors, n_pulses, n_samples)

    pulse_index = pd.MultiIndex.from_arrays(
        [[train_id] * n_pulses, list(range(n_pulses))],
        names=['trainId', 'pulseId'],
    )

    return xr.DataArray(
        stacked,
        dims=['detector', 'pulse', 'sample'],
        coords={
            'detector': np.arange(n_detectors),
            'pulse': pulse_index,
            'sample': np.arange(n_samples, dtype=np.int64),
        },
    )


def _doocspie_worker(addresses, timeout_seconds, queue, stop_event):
    """Worker process: blocks on successive trains and enqueues DataArrays."""
    from doocspie.abo import TrainAbo

    train_abo = TrainAbo(timeout_seconds=timeout_seconds)
    for addr in addresses:
        train_abo.add(addr)

    for train_event in train_abo:
        if stop_event.is_set():
            break
        try:
            da = _convert_train_event(train_event, addresses)
            if queue.full():
                try:
                    queue.get_nowait()
                except Exception:
                    pass
            queue.put_nowait(da)
        except Exception as e:
            print(f"DoocspieStream conversion error: {e}")


class CircularBuffer:
    """Circular buffer for storing train data"""

    def __init__(self, size=10):
        self.size = size
        self._buffer = []
        self.lock = Lock()

    def push(self, data):
        with self.lock:
            if len(self._buffer) >= self.size:
                self._buffer.pop(0)
            self._buffer.append(data)

    def get_all(self):
        """Get all items in buffer"""
        with self.lock:
            return list(self._buffer)

    def size_current(self):
        with self.lock:
            return len(self._buffer)

    def clear(self):
        with self.lock:
            self._buffer.clear()


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
        # Accept either a list or a keyed dict (e.g. {1: addr1, 2: addr2, ...})
        if isinstance(addresses, dict):
            self._addresses = [addresses[k] for k in sorted(addresses)]
        else:
            self._addresses = list(addresses)
        self.n_detectors = len(self._addresses)
        # Expose .data so MainWindow can call len(stream.data.coords['detector'])
        self.data = _DataProxy(self.n_detectors)

        self._queue: Queue = Queue(maxsize=4)
        self._stop_event = Event()
        self._process = Process(
            target=_doocspie_worker,
            args=(self._addresses, timeout_seconds, self._queue, self._stop_event),
            daemon=True,
        )
        self._process.start()

    # ------------------------------------------------------------------
    def get_next_train(self):
        """Non-blocking: drain the queue and return only the most recent train.

        Older buffered trains are discarded so the caller always works on the
        live edge of the stream rather than falling behind.
        """
        item = None
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
        return item

    def stop(self):
        """Signal the worker process to terminate and wait for it to exit."""
        self._stop_event.set()
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
