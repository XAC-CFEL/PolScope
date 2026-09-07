"""Shot-by-shot history buffer with periodic CSV dumps.

Records scalar per-shot values (Plin, φ, β, per-detector peak position and height),
keeps the last MEMORY_LIMIT records in RAM, and appends older records to a CSV file
so that long sessions are fully preserved on disk.

Usage
-----
buf = HistoryBuffer(dump_dir=Path("history_data"))
buf.append({'Plin': 0.8, 'phi': 45.0, 'beta': 2.0, 'pos_0': 312.4, ...})
df = buf.get_recent(100)          # last 100 shots from RAM
df = buf.get_range(0, 500)        # shots 0-499, loading CSV if needed
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List


class HistoryBuffer:
    """Accumulates per-shot scalar records in RAM and flushes oldest to CSV.

    * Keeps at most ``MEMORY_LIMIT`` records in RAM.
    * Flushes older records to a single session CSV (appended on every flush).
    * ``get_range`` reads back earlier shots from that CSV when needed.
    """

    MEMORY_LIMIT: int = 1000   # max records kept in RAM at one time

    def __init__(self, dump_dir: Optional[Path] = None):
        self._records: List[dict] = []
        self._shot_counter: int = 0
        self._dump_file: Optional[Path] = None

        dump_base = Path(dump_dir) if dump_dir is not None else Path('.')
        dump_base.mkdir(parents=True, exist_ok=True)
        self._dump_dir = dump_base
        self._session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(self, record: dict) -> int:
        """Append one shot record; returns the assigned shot index."""
        r = dict(record)
        r['shot'] = self._shot_counter
        self._shot_counter += 1
        self._records.append(r)

        if len(self._records) > self.MEMORY_LIMIT:
            self._flush_oldest(len(self._records) - self.MEMORY_LIMIT)

        return r['shot']

    def _flush_oldest(self, n: int) -> None:
        to_dump = self._records[:n]
        self._records = self._records[n:]
        df = pd.DataFrame(to_dump)
        if self._dump_file is None:
            self._dump_file = self._dump_dir / f'history_{self._session_id}.csv'
        write_header = not self._dump_file.exists()
        df.to_csv(self._dump_file, mode='a', header=write_header, index=False)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    @property
    def total_shots(self) -> int:
        """Total number of shots appended in this session."""
        return self._shot_counter

    @property
    def memory_start_shot(self) -> int:
        """First shot index still in RAM (0 when nothing has been flushed)."""
        return self._shot_counter - len(self._records)

    @property
    def dump_file(self) -> Optional[Path]:
        """Path to the current session CSV dump file, or None if none exists yet."""
        return self._dump_file

    def get_recent(self, n: int = 100) -> pd.DataFrame:
        """Return the last *n* records from RAM."""
        recent = self._records[-n:] if n < len(self._records) else list(self._records)
        return pd.DataFrame(recent) if recent else pd.DataFrame()

    def get_range(self, shot_start: int, shot_end: int) -> pd.DataFrame:
        """Return records with shot in [shot_start, shot_end), loading the CSV if needed."""
        parts: List[pd.DataFrame] = []
        mem_start = self.memory_start_shot

        if shot_start < mem_start and self._dump_file is not None:
            try:
                df_file = pd.read_csv(self._dump_file)
                mask = (df_file['shot'] >= shot_start) & (
                    df_file['shot'] < min(shot_end, mem_start)
                )
                chunk = df_file[mask]
                if not chunk.empty:
                    parts.append(chunk)
            except Exception as exc:
                print(f'HistoryBuffer: could not read {self._dump_file}: {exc}')

        if self._records and shot_end > mem_start:
            mem_df = pd.DataFrame(self._records)
            mask = (mem_df['shot'] >= max(shot_start, mem_start)) & (
                mem_df['shot'] < shot_end
            )
            chunk = mem_df[mask]
            if not chunk.empty:
                parts.append(chunk)

        if not parts:
            return pd.DataFrame()
        return (
            pd.concat(parts, ignore_index=True)
            .sort_values('shot')
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def flush_all(self) -> None:
        """Force-flush all remaining RAM records (e.g. on stop)."""
        if self._records:
            self._flush_oldest(len(self._records))

    def clear(self) -> None:
        """Discard in-RAM data and start a fresh session (does not delete dump files)."""
        self._records.clear()
        self._shot_counter = 0
        self._dump_file = None
        self._session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
