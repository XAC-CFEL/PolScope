import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


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
    # Per-peak baseline data (only present when baseline columns exist in results)
    # Each dict: {'bl_x': [xL, xR], 'bl_y': [yL, yR], 'adj_x': np.ndarray, 'adj_y': np.ndarray}
    baseline_data: Optional[List[Dict[str, Any]]] = field(default=None)
