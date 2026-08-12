"""
Pyqtgraph-based plotting canvases.

Replaces the original matplotlib/blitting implementation.  Public API is kept
identical so main_window.py requires only small changes (removing the
matplotlib NavigationToolbar import/usage and a direct .ax.set_title call).

Performance notes
-----------------
* No manual blitting -- pyqtgraph uses an OpenGL (or optimised software)
  back-end that is intrinsically incremental.
* ``fast_update`` just calls ``setData`` on pre-existing items; the Qt event
  loop coalesces all pending paints into a single GPU flush.
* ``background`` and ``init_blit`` are kept as no-ops for API compatibility.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget
from scipy.optimize import curve_fit
from typing import List, Optional

from ToFPipeline.ToFPipeline import GlobalConfig, polarization_model, sepModel
from colors import (COLOR_TRACE, COLOR_PEAK, COLOR_FWHM, COLOR_DATA,
                    COLOR_FIT, COLOR_PHI, COLOR_DISABLED, COLOR_GRAY)
from models import PlotData

# Global pyqtgraph config — white background, black foreground, antialiased
# useOpenGL=False prevents STATUS_FATAL_USER_CALLBACK_EXCEPTION (0xC000041D)
# crashes on Windows systems without a suitable OpenGL driver.
pg.setConfigOptions(antialias=True, background='w', foreground='k',
                    useOpenGL=False)

_COLOR_BASELINE = '#228833'   # green dashed baseline
_COLOR_ADJUSTED = '#228833'   # green dotted adjusted trace
_MAX_BASELINE_PEAKS = 5       # pre-created adjusted-trace items per detector

_SNAPSHOT_COLORS = [
    '#CC79A7',  # reddish purple
    '#009E73',  # bluish green
    '#E69F00',  # orange
    '#F0E442',  # yellow
    '#000000',  # black
]

# 16 distinct colours (tab20 first 16) for per-detector history lines
_DET_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94',
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _pen(color: str, width: float = 1.0,
         style: Qt.PenStyle = Qt.PenStyle.SolidLine,
         alpha: int = 255) -> pg.QtGui.QPen:
    c = QColor(color)
    c.setAlpha(alpha)
    return pg.mkPen(c, width=width, style=style)


def _brush(color: str, alpha: int = 255) -> pg.QtGui.QBrush:
    c = QColor(color)
    c.setAlpha(alpha)
    return pg.mkBrush(c)


def _snap_pen(color: str, alpha: float) -> pg.QtGui.QPen:
    """Return a pen for snapshot lines respecting fractional alpha."""
    return _pen(color, width=0.8, alpha=int(alpha * 255))


# ---------------------------------------------------------------------------
# FastMplCanvas — 16-detector grid
# ---------------------------------------------------------------------------

class FastMplCanvas(pg.GraphicsLayoutWidget):
    """Pyqtgraph grid canvas replacing the original matplotlib blitting canvas."""

    def __init__(self, parent=None, width=5, height=4, dpi=100, n_detectors=16):
        super().__init__(parent=parent)
        self.setViewport(QWidget())  # force software rendering (no OpenGL)
        self.n_detectors = n_detectors
        # Compatibility stubs — these attributes are written from main_window.py
        self.background = None

        n_cols = int(np.ceil(np.sqrt(n_detectors)))

        self.plots: List[pg.PlotItem] = []
        self.curves: List[pg.PlotDataItem] = []
        self.scatter_items: List[pg.ScatterPlotItem] = []
        self.fwhm_curves: List[pg.PlotDataItem] = []
        self.baseline_curves: List[pg.PlotDataItem] = []
        self.adj_curves: List[List[pg.PlotDataItem]] = []
        self.text_items: List[pg.TextItem] = []

        for i in range(n_detectors):
            row = i // n_cols
            col = i % n_cols
            p = self.addPlot(row=row, col=col)
            p.setTitle(f'Det {i}')
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setYRange(-0.1, 1.1, padding=0)
            p.setXRange(0, 1000, padding=0)
            p.hideButtons()
            self.plots.append(p)

            curve = p.plot([], [], pen=_pen(COLOR_TRACE, width=0.5))
            self.curves.append(curve)

            scatter = pg.ScatterPlotItem(size=5, pen=None, brush=_brush(COLOR_PEAK))
            p.addItem(scatter)
            self.scatter_items.append(scatter)

            # FWHM — multiple horizontal segments encoded as connect='pairs'
            fwhm = pg.PlotDataItem([], [], pen=_pen(COLOR_FWHM, width=1.5),
                                   connect='pairs')
            p.addItem(fwhm)
            self.fwhm_curves.append(fwhm)

            # Baseline — dashed green segments
            baseline = pg.PlotDataItem(
                [], [],
                pen=_pen(_COLOR_BASELINE, width=1.2, style=Qt.PenStyle.DashLine),
                connect='pairs')
            p.addItem(baseline)
            self.baseline_curves.append(baseline)

            # Baseline-adjusted traces
            adj_list: List[pg.PlotDataItem] = []
            for _ in range(_MAX_BASELINE_PEAKS):
                adj = p.plot([], [],
                             pen=_pen(_COLOR_ADJUSTED, width=1.0,
                                      style=Qt.PenStyle.DotLine, alpha=178))
                adj_list.append(adj)
            self.adj_curves.append(adj_list)

            # Overlay text (N/A, OFF)
            text = pg.TextItem('', color=COLOR_GRAY, anchor=(0.5, 0.5))
            text.setVisible(False)
            p.addItem(text)
            self.text_items.append(text)

        self.snapshots: List[dict] = []

    # ------------------------------------------------------------------ #
    # Compatibility stubs                                                   #
    # ------------------------------------------------------------------ #

    def init_blit(self):
        """No-op — pyqtgraph does not need explicit blit initialisation."""

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _show_overlay(self, i: int, msg: str, color: str, bg: str = 'white'):
        p = self.plots[i]
        p.setXRange(0, 1000, padding=0)
        p.setYRange(-0.1, 1.1, padding=0)
        t = self.text_items[i]
        t.setPos(500.0, 0.5)
        t.setText(msg)
        t.setColor(QColor(color))
        t.setVisible(True)
        p.getViewBox().setBackgroundColor(QColor(bg))

    def _hide_overlay(self, i: int):
        self.text_items[i].setVisible(False)
        self.plots[i].getViewBox().setBackgroundColor(QColor('white'))

    # ------------------------------------------------------------------ #
    # Snapshot API                                                          #
    # ------------------------------------------------------------------ #

    def take_snapshot(self, plot_data_list, alpha: float = 0.3, label=None):
        color = _SNAPSHOT_COLORS[len(self.snapshots) % len(_SNAPSHOT_COLORS)]
        if label is None:
            label = f"Snap {len(self.snapshots) + 1}"
        snap_lines: List[Optional[pg.PlotDataItem]] = []
        for i, pd_item in enumerate(plot_data_list):
            if i >= len(self.plots):
                snap_lines.append(None)
                continue
            line = self.plots[i].plot([], [], pen=_snap_pen(color, alpha))
            if pd_item.has_data and pd_item.is_enabled and len(pd_item.samples) > 0:
                line.setData(pd_item.samples, pd_item.values)
            snap_lines.append(line)
        self.snapshots.append({
            'label': label, 'alpha': alpha, 'visible': True,
            'color': color, 'lines': snap_lines,
        })

    def set_snapshot_alpha(self, idx: int, alpha: float):
        if 0 <= idx < len(self.snapshots):
            snap = self.snapshots[idx]
            snap['alpha'] = alpha
            p = _snap_pen(snap['color'], alpha)
            for line in snap['lines']:
                if line is not None:
                    line.setPen(p)

    def set_snapshot_color(self, idx: int, color: str):
        if 0 <= idx < len(self.snapshots):
            snap = self.snapshots[idx]
            snap['color'] = color
            p = _snap_pen(color, snap['alpha'])
            for line in snap['lines']:
                if line is not None:
                    line.setPen(p)

    def rename_snapshot(self, idx: int, name: str):
        if 0 <= idx < len(self.snapshots):
            self.snapshots[idx]['label'] = name

    def remove_snapshot(self, idx: int):
        if 0 <= idx < len(self.snapshots):
            snap = self.snapshots.pop(idx)
            for i, line in enumerate(snap['lines']):
                if line is not None and i < len(self.plots):
                    self.plots[i].removeItem(line)

    def set_snapshot_visible(self, idx: int, visible: bool):
        if 0 <= idx < len(self.snapshots):
            self.snapshots[idx]['visible'] = visible
            for line in self.snapshots[idx]['lines']:
                if line is not None:
                    line.setVisible(visible)

    def clear_all_snapshots(self):
        for snap in self.snapshots:
            for i, line in enumerate(snap['lines']):
                if line is not None and i < len(self.plots):
                    self.plots[i].removeItem(line)
        self.snapshots.clear()

    # ------------------------------------------------------------------ #
    # Main update                                                           #
    # ------------------------------------------------------------------ #

    def fast_update(self, plot_data_list: List[PlotData],
                    show_baseline: bool = True,
                    normalize: bool = True,
                    shared_y: bool = False):
        # Pre-compute shared y-range when requested (non-normalised mode)
        if shared_y and not normalize:
            all_maxes = [
                float(np.max(pd_i.values))
                for pd_i in plot_data_list
                if pd_i.has_data and pd_i.is_enabled and len(pd_i.values) > 0
            ]
            global_max = max(all_maxes) if all_maxes else 1.0
            if global_max <= 0:
                global_max = 1.0
            shared_ylim = (-0.05 * global_max, 1.05 * global_max)
        else:
            shared_ylim = None

        for i, pd_item in enumerate(plot_data_list):
            if i >= len(self.plots):
                continue

            p        = self.plots[i]
            curve    = self.curves[i]
            scatter  = self.scatter_items[i]
            fwhm     = self.fwhm_curves[i]
            baseline = self.baseline_curves[i]
            adj_list = self.adj_curves[i]

            if not pd_item.has_data:
                curve.setData([], [])
                scatter.setData([], [])
                fwhm.setData([], [])
                baseline.setData([], [])
                for al in adj_list:
                    al.setData([], [])
                self._show_overlay(i, 'N/A', COLOR_GRAY)
                continue

            if not pd_item.is_enabled:
                curve.setData([], [])
                scatter.setData([], [])
                fwhm.setData([], [])
                baseline.setData([], [])
                for al in adj_list:
                    al.setData([], [])
                self._show_overlay(i, 'OFF', COLOR_DISABLED, '#ffeeee')
                continue

            self._hide_overlay(i)

            # Trace
            if len(pd_item.samples) > 0:
                curve.setData(pd_item.samples, pd_item.values)
                x_min = float(pd_item.samples[0])
                x_max = float(pd_item.samples[-1])
                if x_max > x_min:
                    p.setXRange(x_min, x_max, padding=0)
                if normalize:
                    p.setYRange(-0.1, 1.1, padding=0)
                elif shared_ylim is not None:
                    p.setYRange(shared_ylim[0], shared_ylim[1], padding=0)
                else:
                    d_max = float(np.max(pd_item.values)) if len(pd_item.values) > 0 else 1.0
                    if d_max <= 0:
                        d_max = 1.0
                    p.setYRange(-0.05 * d_max, 1.05 * d_max, padding=0)
            else:
                curve.setData([], [])

            # Peaks
            if pd_item.peak_positions is not None and len(pd_item.peak_positions) > 0:
                scatter.setData(x=pd_item.peak_positions[:, 0],
                                y=pd_item.peak_positions[:, 1])
            else:
                scatter.setData([], [])

            # FWHM lines (connect='pairs')
            if pd_item.fwhm_lines is not None and len(pd_item.fwhm_lines) > 0:
                xs, ys = [], []
                for pos, wl, wr, hh in pd_item.fwhm_lines:
                    xs += [pos + wl, pos + wr]
                    ys += [hh, hh]
                fwhm.setData(np.asarray(xs), np.asarray(ys))
            else:
                fwhm.setData([], [])

            # Baseline + adjusted traces
            if show_baseline and pd_item.baseline_data:
                xs_bl, ys_bl = [], []
                for k, bd in enumerate(pd_item.baseline_data):
                    xs_bl += [bd['bl_x'][0], bd['bl_x'][1]]
                    ys_bl += [bd['bl_y'][0], bd['bl_y'][1]]
                    if k < len(adj_list):
                        adj_list[k].setData(bd['adj_x'], bd['adj_y'])
                baseline.setData(np.asarray(xs_bl), np.asarray(ys_bl))
                for k in range(len(pd_item.baseline_data), len(adj_list)):
                    adj_list[k].setData([], [])
            else:
                baseline.setData([], [])
                for al in adj_list:
                    al.setData([], [])


# ---------------------------------------------------------------------------
# PolarPlotCanvas
# ---------------------------------------------------------------------------

class PolarPlotCanvas(pg.PlotWidget):
    """Polar plot rendered in Cartesian coordinates."""

    def __init__(self, parent=None, width=6, height=6, dpi=100):
        super().__init__(parent=parent)
        self.setViewport(QWidget())  # force software rendering (no OpenGL)
        # Compatibility stubs
        self.background = None
        self.last_fit_params: Optional[dict] = None

        vb = self.getPlotItem().getViewBox()
        vb.setAspectLocked(True)
        self.getPlotItem().hideAxis('bottom')
        self.getPlotItem().hideAxis('left')
        self.getPlotItem().setTitle('Polarization')
        self.getPlotItem().hideButtons()
        self.showGrid(x=False, y=False)

        # Polar grid (rebuilt when r_max changes significantly)
        self._grid_items: List[pg.PlotDataItem] = []
        self._label_items: List[pg.TextItem] = []
        self._r_max: float = 1.0
        self._build_grid(1.0)

        # Live data items
        self.data_line = pg.PlotDataItem(
            [], [],
            pen=None,
            symbol='o', symbolSize=8,
            symbolPen=None, symbolBrush=_brush(COLOR_DATA))
        self.addItem(self.data_line)

        self.fit_line = pg.PlotDataItem([], [], pen=_pen(COLOR_FIT, width=2))
        self.addItem(self.fit_line)

        self.phi_line1 = pg.PlotDataItem([], [], pen=_pen(COLOR_PHI, width=2, alpha=178))
        self.phi_line2 = pg.PlotDataItem([], [], pen=_pen(COLOR_PHI, width=2, alpha=178))
        self.addItem(self.phi_line1)
        self.addItem(self.phi_line2)

        self.fit_text = pg.TextItem('', anchor=(0.0, 1.0), color='k')
        self.fit_text.setPos(-self._r_max * 1.2, self._r_max * 1.2)
        self.addItem(self.fit_text)

        # Angles from config
        nxs_config = GlobalConfig.get_for_class('NXSLoader')
        self.angles_deg = np.array(
            nxs_config.get('angles', np.linspace(0, 337.5, 16).tolist())
            if nxs_config else np.linspace(0, 337.5, 16).tolist()
        )
        self.angles_rad = np.deg2rad(self.angles_deg)

        self.snapshots: List[dict] = []

    # ------------------------------------------------------------------ #
    # Polar grid                                                            #
    # ------------------------------------------------------------------ #

    def _build_grid(self, r_max: float):
        for item in self._grid_items:
            self.removeItem(item)
        for item in self._label_items:
            self.removeItem(item)
        self._grid_items.clear()
        self._label_items.clear()

        self._r_max = r_max
        gray = _pen('#aaaaaa', width=0.5)
        theta = np.linspace(0, 2 * np.pi, 300)

        for frac in (0.25, 0.5, 0.75, 1.0):
            r = frac * r_max
            item = pg.PlotDataItem(r * np.cos(theta), r * np.sin(theta), pen=gray)
            self.addItem(item)
            self._grid_items.append(item)

        for deg in range(0, 360, 45):
            rad = np.deg2rad(deg)
            ray = pg.PlotDataItem(
                [0, r_max * np.cos(rad)], [0, r_max * np.sin(rad)], pen=gray)
            self.addItem(ray)
            self._grid_items.append(ray)
            lbl = pg.TextItem(f'{deg}\u00b0', anchor=(0.5, 0.5), color='#555555')
            lbl.setPos(1.15 * r_max * np.cos(rad), 1.15 * r_max * np.sin(rad))
            self.addItem(lbl)
            self._label_items.append(lbl)

        margin = 1.35 * r_max
        self.setXRange(-margin, margin, padding=0)
        self.setYRange(-margin, margin, padding=0)

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def set_angles(self, angles_deg):
        self.angles_deg = np.array(angles_deg)
        self.angles_rad = np.deg2rad(self.angles_deg)

    def init_blit(self):
        pass

    def force_full_redraw(self):
        pass

    @staticmethod
    def _to_cart(theta, r):
        return r * np.cos(theta), r * np.sin(theta)

    def update_polar_plot(self, results_df,
                          peak_no=0, value_type='height', beta=2.0,
                          setPlin=None, fitBeta=False,
                          setPhi=None, fitPhi=True):
        self.data_line.setData([], [])
        self.fit_line.setData([], [])
        self.phi_line1.setData([], [])
        self.phi_line2.setData([], [])
        self.fit_text.setText('')
        self.last_fit_params = None

        if results_df is None or results_df.empty:
            return

        peak_data = results_df[results_df['peakNo'] == peak_no]
        if peak_data.empty:
            return

        det_vals = peak_data.groupby('detector')[value_type].mean()
        if det_vals.empty:
            return

        avail = det_vals.index.values
        avail = avail[avail < len(self.angles_rad)]
        if len(avail) == 0:
            return

        theta = self.angles_rad[avail]
        r_vals = det_vals.loc[avail].values

        r_max = float(np.nanmax(r_vals)) * 1.2 if len(r_vals) > 0 else 1.0
        if r_max <= 0:
            r_max = 1.0

        if abs(r_max - self._r_max) > self._r_max * 0.1:
            self._build_grid(r_max)
            self.fit_text.setPos(-r_max * 1.2, r_max * 1.2)

        xd, yd = self._to_cart(theta, r_vals)
        self.data_line.setData(xd, yd)

        if len(theta) >= 3:
            try:
                self._fit_and_plot(theta, r_vals, beta, r_max,
                                   setPlin=setPlin, fitBeta=fitBeta,
                                   setPhi=setPhi, fitPhi=fitPhi)
            except Exception as e:
                print(f"Fit error: {e}")
                self.fit_text.setText(f"Fit failed: {str(e)[:40]}")

    def _fit_and_plot(self, theta, r_values, beta, r_max,
                      setPlin=None, fitBeta=False, setPhi=None, fitPhi=True):
        fit_kws = dict(method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-10, maxfev=5000)
        scale_guess = float(np.mean(r_values))
        beta0 = beta if beta != 0 else 1.0
        phi0 = setPhi if setPhi is not None else 0.0

        Plin_err = phi_err = beta2_err = scale_err = None

        if fitBeta:
            plin_val = setPlin if setPlin is not None else 1.0
            Plin_fit = plin_val
            if fitPhi:
                def model(t, phi, beta2, scale):
                    return polarization_model(t, Plin=plin_val, phi=phi,
                                              beta2=beta2, scale=scale)
                popt, pcov = curve_fit(model, theta, r_values,
                                       p0=[phi0, beta0, scale_guess],
                                       bounds=([-np.pi, -4., 0], [np.pi, 4., np.inf]),
                                       **fit_kws)
                phi_fit, beta2_fit, scale_fit = popt
                phi_err, beta2_err, scale_err = np.sqrt(np.diag(pcov))
            else:
                phi_fit = setPhi if setPhi is not None else 0.0
                def model(t, beta2, scale):
                    return polarization_model(t, Plin=plin_val, phi=phi_fit,
                                              beta2=beta2, scale=scale)
                popt, pcov = curve_fit(model, theta, r_values,
                                       p0=[beta0, scale_guess],
                                       bounds=([-4., 0], [4., np.inf]),
                                       **fit_kws)
                beta2_fit, scale_fit = popt
                beta2_err, scale_err = np.sqrt(np.diag(pcov))
        else:
            beta2_fit = beta
            if fitPhi:
                def model(t, A, B, scale):
                    return sepModel(t, A, B, beta2=beta, scale=scale)
                popt, pcov = curve_fit(model, theta, r_values,
                                       p0=[0., 0., scale_guess],
                                       bounds=([-2., -2., 0], [2., 2., np.inf]),
                                       **fit_kws)
                A_fit, B_fit, scale_fit = popt
                scale_err = float(np.sqrt(pcov[2, 2]))
                cov_AB = pcov[:2, :2]
                Plin_fit = float(np.sqrt(A_fit**2 + B_fit**2))
                phi_fit  = float(0.5 * np.arctan2(B_fit, A_fit))
                sA2, sB2, sAB = cov_AB[0, 0], cov_AB[1, 1], cov_AB[0, 1]
                if Plin_fit > 1e-10:
                    Plin_err = float(np.sqrt(
                        (A_fit / Plin_fit)**2 * sA2 +
                        (B_fit / Plin_fit)**2 * sB2 +
                        2 * (A_fit * B_fit / Plin_fit**2) * sAB))
                    phi_err = float(0.5 * np.sqrt(
                        (B_fit**2 * sA2 + A_fit**2 * sB2 -
                         2 * A_fit * B_fit * sAB)
                        / (A_fit**2 + B_fit**2)**2))
                else:
                    Plin_err = phi_err = 0.0
            else:
                phi_fit = setPhi if setPhi is not None else 0.0
                def model(t, Plin, scale):
                    return polarization_model(t, Plin=Plin, phi=phi_fit,
                                              beta2=beta, scale=scale)
                popt, pcov = curve_fit(model, theta, r_values,
                                       p0=[0.2, scale_guess],
                                       bounds=([0., 0], [2., np.inf]),
                                       **fit_kws)
                Plin_fit, scale_fit = popt
                Plin_err, scale_err = np.sqrt(np.diag(pcov))

        self.last_fit_params = {
            'Plin': Plin_fit, 'phi': phi_fit,
            'scale': scale_fit, 'beta': beta2_fit, 'pcov': pcov,
        }

        tf = np.linspace(0, 2 * np.pi, 360)
        rf = polarization_model(tf, Plin=Plin_fit, phi=phi_fit,
                                beta2=beta2_fit, scale=scale_fit)
        xf, yf = self._to_cart(tf, rf)
        self.fit_line.setData(xf, yf)

        if Plin_fit > 0.015:
            for line_item, angle in ((self.phi_line1, phi_fit),
                                     (self.phi_line2, phi_fit + np.pi)):
                x, y = self._to_cart(np.array([angle, angle]),
                                     np.array([0.0, r_max]))
                line_item.setData(x, y)
        else:
            self.phi_line1.setData([], [])
            self.phi_line2.setData([], [])

        phi_deg = np.rad2deg(phi_fit) % 360
        plin_str = (f"Plin: {Plin_fit:.4f} \u00b1 {Plin_err:.4f}"
                    if Plin_err is not None
                    else f"Plin: {Plin_fit:.4f} (fixed)")
        phi_str = (f"\u03c6: {phi_deg:.1f}\u00b0 \u00b1 {np.rad2deg(phi_err):.1f}\u00b0"
                   if phi_err is not None
                   else f"\u03c6: {phi_deg:.1f}\u00b0 (fixed)")
        beta_str = (f"\u03b2: {beta2_fit:.4f} \u00b1 {beta2_err:.4f}"
                    if beta2_err is not None
                    else f"\u03b2: {beta2_fit:.3f} (fixed)")
        scale_str = (f"Scale: {scale_fit:.4f} \u00b1 {scale_err:.4f}"
                     if scale_err is not None
                     else f"Scale: {scale_fit:.4f}")
        self.fit_text.setText(f"{plin_str}\n{phi_str}\n{beta_str}\n{scale_str}")

    # ------------------------------------------------------------------ #
    # Snapshot API                                                          #
    # ------------------------------------------------------------------ #

    def take_snapshot(self, alpha: float = 0.3, label=None):
        color = _SNAPSHOT_COLORS[len(self.snapshots) % len(_SNAPSHOT_COLORS)]
        if label is None:
            label = f"Snap {len(self.snapshots) + 1}"
        a_int = int(alpha * 255)

        xd, yd = self.data_line.getData()
        snap_data = pg.PlotDataItem(
            [] if xd is None else list(xd),
            [] if yd is None else list(yd),
            pen=None,
            symbol='o', symbolSize=5,
            symbolPen=None,
            symbolBrush=_brush(color, alpha=a_int))
        self.addItem(snap_data)

        xf, yf = self.fit_line.getData()
        snap_fit = pg.PlotDataItem(
            [] if xf is None else list(xf),
            [] if yf is None else list(yf),
            pen=_pen(color, width=1.5, style=Qt.PenStyle.DashLine, alpha=a_int))
        self.addItem(snap_fit)

        self.snapshots.append({
            'label': label, 'alpha': alpha, 'visible': True, 'color': color,
            'data_item': snap_data, 'fit_item': snap_fit,
        })

    def _update_snap_style(self, snap: dict):
        a = int(snap['alpha'] * 255)
        c = snap['color']
        snap['data_item'].setSymbolBrush(_brush(c, alpha=a))
        snap['fit_item'].setPen(_pen(c, width=1.5, style=Qt.PenStyle.DashLine, alpha=a))

    def set_snapshot_alpha(self, idx: int, alpha: float):
        if 0 <= idx < len(self.snapshots):
            self.snapshots[idx]['alpha'] = alpha
            self._update_snap_style(self.snapshots[idx])

    def set_snapshot_color(self, idx: int, color: str):
        if 0 <= idx < len(self.snapshots):
            self.snapshots[idx]['color'] = color
            self._update_snap_style(self.snapshots[idx])

    def rename_snapshot(self, idx: int, name: str):
        if 0 <= idx < len(self.snapshots):
            self.snapshots[idx]['label'] = name

    def remove_snapshot(self, idx: int):
        if 0 <= idx < len(self.snapshots):
            snap = self.snapshots.pop(idx)
            self.removeItem(snap['data_item'])
            self.removeItem(snap['fit_item'])

    def set_snapshot_visible(self, idx: int, visible: bool):
        if 0 <= idx < len(self.snapshots):
            snap = self.snapshots[idx]
            snap['visible'] = visible
            snap['data_item'].setVisible(visible)
            snap['fit_item'].setVisible(visible)

    def clear_all_snapshots(self):
        for snap in self.snapshots:
            self.removeItem(snap['data_item'])
            self.removeItem(snap['fit_item'])
        self.snapshots.clear()


# ---------------------------------------------------------------------------
# AngularHeatmapCanvas
# ---------------------------------------------------------------------------

class AngularHeatmapCanvas(pg.GraphicsLayoutWidget):
    """Polar heatmap using PColorMeshItem (intensity vs angle vs sample pos)."""

    def __init__(self, parent=None, width=7, height=7, dpi=100):
        super().__init__(parent=parent)
        self.setViewport(QWidget())  # force software rendering (no OpenGL)

        self.plot = self.addPlot(row=0, col=0)
        self.plot.getViewBox().setAspectLocked(True)
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        self.plot.setTitle('Angular Heatmap')
        self.plot.hideButtons()

        nxs_config = GlobalConfig.get_for_class('NXSLoader')
        self.angles_deg = np.array(
            nxs_config.get('angles', np.linspace(0, 337.5, 16).tolist())
            if nxs_config else np.linspace(0, 337.5, 16).tolist()
        )
        self.angles_rad = np.deg2rad(self.angles_deg)

        self._mesh_item: Optional[pg.PColorMeshItem] = None
        self._peak_items: List[pg.GraphicsObject] = []
        self._grid_items: List[pg.PlotDataItem] = []
        self._grid_labels: List[pg.TextItem] = []

    def set_angles(self, angles_deg):
        self.angles_deg = np.array(angles_deg)
        self.angles_rad = np.deg2rad(self.angles_deg)

    @staticmethod
    def _build_intensity_grid(traces, angles_rad, n_theta=720):
        n_samples = traces.shape[1]
        theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        grid = np.zeros((n_theta, n_samples))
        sort_idx = np.argsort(angles_rad)
        sorted_angles = angles_rad[sort_idx]
        sorted_traces = traces[sort_idx]
        for si in range(n_samples):
            vals = sorted_traces[:, si]
            xp_ext = np.concatenate([sorted_angles - 2 * np.pi,
                                      sorted_angles,
                                      sorted_angles + 2 * np.pi])
            fp_ext = np.concatenate([vals, vals, vals])
            grid[:, si] = np.interp(theta_grid, xp_ext, fp_ext)
        return grid, theta_grid

    def _rebuild_polar_grid(self, r_min: float, r_max: float):
        for item in self._grid_items + self._grid_labels:
            self.plot.removeItem(item)
        self._grid_items.clear()
        self._grid_labels.clear()

        gray = _pen('#bbbbbb', width=0.5)
        theta = np.linspace(0, 2 * np.pi, 300)
        for r in (r_min, (r_min + r_max) / 2, r_max):
            item = pg.PlotDataItem(r * np.cos(theta), r * np.sin(theta), pen=gray)
            self.plot.addItem(item)
            self._grid_items.append(item)

        for deg in range(0, 360, 45):
            rad = np.deg2rad(deg)
            ray = pg.PlotDataItem(
                [r_min * np.cos(rad), r_max * np.cos(rad)],
                [r_min * np.sin(rad), r_max * np.sin(rad)],
                pen=gray)
            self.plot.addItem(ray)
            self._grid_items.append(ray)
            lbl = pg.TextItem(f'{deg}\u00b0', anchor=(0.5, 0.5), color='#555555')
            lbl.setPos(1.12 * r_max * np.cos(rad), 1.12 * r_max * np.sin(rad))
            self.plot.addItem(lbl)
            self._grid_labels.append(lbl)

        margin = 1.25 * r_max
        self.plot.setXRange(-margin, margin, padding=0)
        self.plot.setYRange(-margin, margin, padding=0)

    def update_heatmap(self, plot_data_list: List['PlotData'],
                       results_df,
                       sample_min: int, sample_max: int,
                       interpolate: bool = True,
                       show_peaks: bool = True):
        # Remove previous dynamic items
        if self._mesh_item is not None:
            self.plot.removeItem(self._mesh_item)
            self._mesh_item = None
        for item in self._peak_items:
            self.plot.removeItem(item)
        self._peak_items.clear()

        # Collect enabled detectors
        det_ids, traces_list, sample_coords = [], [], None
        for pd_obj in plot_data_list:
            if not pd_obj.has_data or not pd_obj.is_enabled:
                continue
            if len(pd_obj.samples) == 0:
                continue
            det_id = pd_obj.detector_id
            if det_id >= len(self.angles_deg):
                continue
            si = int(np.searchsorted(pd_obj.samples, sample_min))
            ei = int(np.searchsorted(pd_obj.samples, sample_max, side='right'))
            si, ei = max(0, si), min(len(pd_obj.samples), ei)
            if ei <= si:
                continue
            det_ids.append(det_id)
            traces_list.append(pd_obj.values[si:ei])
            s_slice = pd_obj.samples[si:ei]
            if sample_coords is None or len(s_slice) > len(sample_coords):
                sample_coords = s_slice

        if not det_ids or sample_coords is None or len(sample_coords) == 0:
            return

        n_samples = len(sample_coords)
        traces = np.zeros((len(det_ids), n_samples))
        for k, t in enumerate(traces_list):
            n = min(len(t), n_samples)
            traces[k, :n] = t[:n]

        angles_rad = self.angles_rad[det_ids]
        dr = float(sample_coords[1] - sample_coords[0]) if n_samples > 1 else 1.0
        r_edges = np.concatenate([[sample_coords[0] - dr / 2],
                                   sample_coords + dr / 2])

        vmin = float(traces.min())
        vmax = float(traces.max()) if traces.max() > vmin else vmin + 1e-9
        cmap = pg.colormap.get('viridis')

        if interpolate:
            grid, theta_grid = self._build_intensity_grid(traces, angles_rad, n_theta=720)
            d_theta = theta_grid[1] - theta_grid[0]
            theta_edges = np.append(theta_grid - d_theta / 2,
                                    theta_grid[-1] + d_theta / 2)   # (721,)
            th_m, r_m = np.meshgrid(theta_edges, r_edges, indexing='ij')  # (721, ns+1)
            x_mesh = r_m * np.cos(th_m)
            y_mesh = r_m * np.sin(th_m)
            mesh = pg.PColorMeshItem(x_mesh, y_mesh, grid,
                                     colorMap=cmap, levels=(vmin, vmax))
            self.plot.addItem(mesh)
            self._mesh_item = mesh
        else:
            n_dets = len(det_ids)
            if n_dets > 1:
                s_idx = np.argsort(angles_rad)
                s_ang = angles_rad[s_idx]
                gaps = np.diff(s_ang, append=s_ang[0] + 2 * np.pi)
                left_half  = np.roll(gaps, 1) / 2
                right_half = gaps / 2
                inv_idx = np.argsort(s_idx)
                wl = left_half[inv_idx]
                wr = right_half[inv_idx]
            else:
                wl = wr = np.array([np.pi])

            for k in range(n_dets):
                ang = angles_rad[k]
                th_e = np.array([ang - wl[k], ang + wr[k]])
                th_m, r_m = np.meshgrid(th_e, r_edges, indexing='ij')
                x_w = r_m * np.cos(th_m)
                y_w = r_m * np.sin(th_m)
                z_w = traces[k, :][np.newaxis, :]
                item = pg.PColorMeshItem(x_w, y_w, z_w,
                                         colorMap=cmap, levels=(vmin, vmax))
                self.plot.addItem(item)
                self._peak_items.append(item)

        # Rebuild polar grid
        self._rebuild_polar_grid(float(sample_coords[0]), float(sample_coords[-1]))

        # Peak overlay
        if show_peaks and results_df is not None and not results_df.empty:
            px_list, py_list = [], []
            for _, row in results_df.iterrows():
                det = int(row['detector'])
                if det >= len(self.angles_rad):
                    continue
                ang = self.angles_rad[det]
                pos = row['pos']
                wl_p = row['width left']
                wr_p = row['width right']
                px_list.append(pos * np.cos(ang))
                py_list.append(pos * np.sin(ang))
                wline = pg.PlotDataItem(
                    [(pos + wl_p) * np.cos(ang), (pos + wr_p) * np.cos(ang)],
                    [(pos + wl_p) * np.sin(ang), (pos + wr_p) * np.sin(ang)],
                    pen=_pen('#ff0000', width=0.8))
                self.plot.addItem(wline)
                self._peak_items.append(wline)
            if px_list:
                sc = pg.ScatterPlotItem(x=px_list, y=py_list,
                                        size=6, pen=None, brush=_brush('#ff0000'))
                self.plot.addItem(sc)
                self._peak_items.append(sc)


# ---------------------------------------------------------------------------
# SingleDetectorCanvas
# ---------------------------------------------------------------------------

class SingleDetectorCanvas(pg.PlotWidget):
    """Single-detector zoom/pan canvas replacing the original matplotlib one."""

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        super().__init__(parent=parent)
        self.setViewport(QWidget())  # force software rendering (no OpenGL)
        self.setMaximumHeight(520)

        p = self.getPlotItem()
        p.showGrid(x=True, y=True, alpha=0.3)
        p.setTitle('Detector 0')
        p.setYRange(-0.1, 1.1, padding=0)
        p.setXRange(0, 1000, padding=0)
        p.hideButtons()

        self.line = p.plot([], [], pen=_pen(COLOR_TRACE, width=0.8))
        self.scatter = pg.ScatterPlotItem(size=8, pen=None, brush=_brush(COLOR_PEAK))
        p.addItem(self.scatter)

        self.fwhm_lc = pg.PlotDataItem([], [], pen=_pen(COLOR_FWHM, width=1.5),
                                        connect='pairs')
        p.addItem(self.fwhm_lc)

        self.baseline_lc = pg.PlotDataItem(
            [], [],
            pen=_pen(_COLOR_BASELINE, width=1.2, style=Qt.PenStyle.DashLine),
            connect='pairs')
        p.addItem(self.baseline_lc)

        self.adj_lines: List[pg.PlotDataItem] = []
        for _ in range(_MAX_BASELINE_PEAKS):
            al = p.plot([], [],
                        pen=_pen(_COLOR_ADJUSTED, width=1.0,
                                 style=Qt.PenStyle.DotLine, alpha=178))
            self.adj_lines.append(al)

        self.text_obj = pg.TextItem('', color=COLOR_GRAY, anchor=(0.5, 0.5))
        self.text_obj.setVisible(False)
        p.addItem(self.text_obj)

        self._user_navigated = False
        self.getPlotItem().getViewBox().sigRangeChangedManually.connect(
            self._on_user_navigate)

        self.snapshots: List[dict] = []

    def _on_user_navigate(self, *args):
        self._user_navigated = True

    def _show_overlay(self, msg: str, color: str, bg: str = 'white'):
        self.setXRange(0, 1000, padding=0)
        self.setYRange(-0.1, 1.1, padding=0)
        self.text_obj.setPos(500.0, 0.5)
        self.text_obj.setText(msg)
        self.text_obj.setColor(QColor(color))
        self.text_obj.setVisible(True)
        self.getPlotItem().getViewBox().setBackgroundColor(QColor(bg))

    def _hide_overlay(self):
        self.text_obj.setVisible(False)
        self.getPlotItem().getViewBox().setBackgroundColor(QColor('white'))

    # ------------------------------------------------------------------ #
    # Snapshot API                                                          #
    # ------------------------------------------------------------------ #

    def take_snapshot(self, plot_data_list, alpha: float = 0.3,
                      label=None, current_det_idx: int = 0):
        color = _SNAPSHOT_COLORS[len(self.snapshots) % len(_SNAPSHOT_COLORS)]
        if label is None:
            label = f"Snap {len(self.snapshots) + 1}"
        line = self.getPlotItem().plot([], [], pen=_snap_pen(color, alpha))
        if current_det_idx < len(plot_data_list):
            pd_s = plot_data_list[current_det_idx]
            if pd_s.has_data and pd_s.is_enabled and len(pd_s.samples) > 0:
                line.setData(pd_s.samples, pd_s.values)
        self.snapshots.append({
            'label': label, 'alpha': alpha, 'visible': True, 'color': color,
            'line': line, 'plot_data_list': list(plot_data_list),
        })

    def set_snapshot_alpha(self, idx: int, alpha: float):
        if 0 <= idx < len(self.snapshots):
            snap = self.snapshots[idx]
            snap['alpha'] = alpha
            snap['line'].setPen(_snap_pen(snap['color'], alpha))

    def set_snapshot_color(self, idx: int, color: str):
        if 0 <= idx < len(self.snapshots):
            snap = self.snapshots[idx]
            snap['color'] = color
            snap['line'].setPen(_snap_pen(color, snap['alpha']))

    def rename_snapshot(self, idx: int, name: str):
        if 0 <= idx < len(self.snapshots):
            self.snapshots[idx]['label'] = name

    def remove_snapshot(self, idx: int):
        if 0 <= idx < len(self.snapshots):
            snap = self.snapshots.pop(idx)
            self.getPlotItem().removeItem(snap['line'])

    def set_snapshot_visible(self, idx: int, visible: bool):
        if 0 <= idx < len(self.snapshots):
            self.snapshots[idx]['visible'] = visible
            self.snapshots[idx]['line'].setVisible(visible)

    def clear_all_snapshots(self):
        for snap in self.snapshots:
            self.getPlotItem().removeItem(snap['line'])
        self.snapshots.clear()

    # ------------------------------------------------------------------ #
    # Main update                                                           #
    # ------------------------------------------------------------------ #

    def update_plot(self, plot_data: PlotData,
                    show_baseline: bool = True,
                    normalize: bool = True,
                    det_idx: int = 0):
        self.getPlotItem().setTitle(f'Detector {det_idx}')

        if plot_data is None or not plot_data.has_data:
            self.line.setData([], [])
            self.scatter.setData([], [])
            self.fwhm_lc.setData([], [])
            self.baseline_lc.setData([], [])
            for al in self.adj_lines:
                al.setData([], [])
            self._show_overlay('N/A', COLOR_GRAY)
            return

        if not plot_data.is_enabled:
            self.line.setData([], [])
            self.scatter.setData([], [])
            self.fwhm_lc.setData([], [])
            self.baseline_lc.setData([], [])
            for al in self.adj_lines:
                al.setData([], [])
            self._show_overlay('OFF', COLOR_DISABLED, '#ffeeee')
            return

        self._hide_overlay()

        if len(plot_data.samples) > 0:
            self.line.setData(plot_data.samples, plot_data.values)
            if not self._user_navigated:
                x_min = float(plot_data.samples[0])
                x_max = float(plot_data.samples[-1])
                if x_max > x_min:
                    self.setXRange(x_min, x_max, padding=0)
                if normalize:
                    self.setYRange(-0.1, 1.1, padding=0)
                else:
                    self.getPlotItem().enableAutoRange(axis='y')
        else:
            self.line.setData([], [])

        if plot_data.peak_positions is not None and len(plot_data.peak_positions) > 0:
            self.scatter.setData(x=plot_data.peak_positions[:, 0],
                                 y=plot_data.peak_positions[:, 1])
        else:
            self.scatter.setData([], [])

        if plot_data.fwhm_lines is not None and len(plot_data.fwhm_lines) > 0:
            xs, ys = [], []
            for pos, wl, wr, hh in plot_data.fwhm_lines:
                xs += [pos + wl, pos + wr]
                ys += [hh, hh]
            self.fwhm_lc.setData(np.asarray(xs), np.asarray(ys))
        else:
            self.fwhm_lc.setData([], [])

        if show_baseline and plot_data.baseline_data:
            xs_bl, ys_bl = [], []
            for k, bd in enumerate(plot_data.baseline_data):
                xs_bl += [bd['bl_x'][0], bd['bl_x'][1]]
                ys_bl += [bd['bl_y'][0], bd['bl_y'][1]]
                if k < len(self.adj_lines):
                    self.adj_lines[k].setData(bd['adj_x'], bd['adj_y'])
            self.baseline_lc.setData(np.asarray(xs_bl), np.asarray(ys_bl))
            for k in range(len(plot_data.baseline_data), len(self.adj_lines)):
                self.adj_lines[k].setData([], [])
        else:
            self.baseline_lc.setData([], [])
            for al in self.adj_lines:
                al.setData([], [])

        # Update snapshot lines for the active detector
        for snap in self.snapshots:
            pdl = snap['plot_data_list']
            if det_idx < len(pdl):
                pd_s = pdl[det_idx]
                if pd_s.has_data and pd_s.is_enabled and len(pd_s.samples) > 0:
                    snap['line'].setData(pd_s.samples, pd_s.values)
                else:
                    snap['line'].setData([], [])
            else:
                snap['line'].setData([], [])


# ---------------------------------------------------------------------------
# HistoryCanvas
# ---------------------------------------------------------------------------

class HistoryCanvas(pg.GraphicsLayoutWidget):
    """Shot-history canvas: five stacked plots sharing the x-axis."""

    def __init__(self, parent=None, width=10, height=8, dpi=100, n_detectors=16):
        super().__init__(parent=parent)
        self.setViewport(QWidget())  # force software rendering (no OpenGL)
        self.n_detectors = n_detectors

        self.ax_plin   = self.addPlot(row=0, col=0)
        self.ax_phi    = self.addPlot(row=1, col=0)
        self.ax_beta   = self.addPlot(row=2, col=0)
        self.ax_pos    = self.addPlot(row=3, col=0)
        self.ax_height = self.addPlot(row=4, col=0)

        self.ax_phi.setXLink(self.ax_plin)
        self.ax_beta.setXLink(self.ax_plin)
        self.ax_pos.setXLink(self.ax_plin)
        self.ax_height.setXLink(self.ax_plin)

        for ax, lbl in zip(
            [self.ax_plin, self.ax_phi, self.ax_beta, self.ax_pos, self.ax_height],
            ['Plin', '\u03c6 (\u00b0)', '\u03b2', 'Pos.', 'Height'],
        ):
            ax.setLabel('left', lbl)
            ax.showGrid(x=True, y=True, alpha=0.3)
            ax.hideButtons()

        self.ax_height.setLabel('bottom', 'Shot')

        self.line_plin = self.ax_plin.plot([], [], pen=_pen('#4477aa', width=1.0))
        self.line_phi  = self.ax_phi.plot([], [], pen=_pen('#228833', width=1.0))
        self.line_beta = self.ax_beta.plot([], [], pen=_pen('#bb5566', width=1.0))

        self.lines_pos:    dict = {}
        self.lines_height: dict = {}
        self._build_det_lines()

        self.on_range_request = None
        self._memory_start_shot: int = 0
        self._user_xlim: bool = False

        self.ax_plin.getViewBox().sigRangeChangedManually.connect(
            self._on_range_changed_manually)

    def _build_det_lines(self):
        for line in list(self.lines_pos.values()) + list(self.lines_height.values()):
            try:
                self.ax_pos.removeItem(line)
            except Exception:
                pass
            try:
                self.ax_height.removeItem(line)
            except Exception:
                pass
        self.lines_pos.clear()
        self.lines_height.clear()

        for det_id in range(self.n_detectors):
            color = _DET_PALETTE[det_id % len(_DET_PALETTE)]
            lp = self.ax_pos.plot([], [], pen=_pen(color, width=0.8, alpha=217))
            lh = self.ax_height.plot([], [], pen=_pen(color, width=0.8, alpha=217))
            self.lines_pos[det_id]    = lp
            self.lines_height[det_id] = lh

    def _on_range_changed_manually(self, vb, ranges):
        self._user_xlim = True
        xlim = self.ax_plin.getViewBox().viewRange()[0]
        shot_start = int(np.floor(xlim[0]))
        shot_end   = int(np.ceil(xlim[1]))
        if shot_start < self._memory_start_shot and self.on_range_request is not None:
            self.on_range_request(shot_start, shot_end)

    def reset_view(self):
        self._user_xlim = False

    def set_n_detectors(self, n: int):
        self.n_detectors = n
        self._build_det_lines()

    def update_history(self, df: pd.DataFrame,
                       memory_start_shot: int = 0,
                       enabled_detectors=None,
                       window: Optional[int] = 100):
        if df is None or df.empty:
            return

        self._memory_start_shot = memory_start_shot
        shots = df['shot'].values

        def _upd(line, ax, col):
            if col in df.columns:
                line.setData(shots, df[col].values.astype(float))
            else:
                line.setData([], [])
            ax.enableAutoRange(axis='y')

        _upd(self.line_plin,  self.ax_plin,  'Plin')
        _upd(self.line_phi,   self.ax_phi,   'phi')
        _upd(self.line_beta,  self.ax_beta,  'beta')

        any_pos = any_height = False
        for det_id in range(self.n_detectors):
            show = (enabled_detectors is None or det_id in enabled_detectors)
            lp = self.lines_pos.get(det_id)
            lh = self.lines_height.get(det_id)
            col_p = f'pos_{det_id}'
            col_h = f'height_{det_id}'

            if lp is not None:
                if show and col_p in df.columns:
                    lp.setData(shots, df[col_p].values.astype(float))
                    lp.setVisible(True)
                    any_pos = True
                else:
                    lp.setData([], [])
                    lp.setVisible(False)

            if lh is not None:
                if show and col_h in df.columns:
                    lh.setData(shots, df[col_h].values.astype(float))
                    lh.setVisible(True)
                    any_height = True
                else:
                    lh.setData([], [])
                    lh.setVisible(False)

        if any_pos:
            self.ax_pos.enableAutoRange(axis='y')
        if any_height:
            self.ax_height.enableAutoRange(axis='y')

        if not self._user_xlim and shots.size > 0 and window is not None and window > 0:
            xmax = float(shots[-1]) + 1.0
            xmin = max(float(shots[0]) - 0.5, xmax - window)
            self.ax_plin.getViewBox().setXRange(xmin, xmax, padding=0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
