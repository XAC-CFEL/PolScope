import numpy as np
from dataclasses import dataclass
from typing import Optional


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
