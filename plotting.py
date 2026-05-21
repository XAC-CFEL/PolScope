import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit
from typing import List

from ToFPipeline.ToFPipeline import GlobalConfig, polarization_model
from colors import (COLOR_TRACE, COLOR_PEAK, COLOR_FWHM, COLOR_DATA,
                    COLOR_FIT, COLOR_PHI, COLOR_DISABLED, COLOR_GRAY)
from models import PlotData

_COLOR_BASELINE = '#228833'   # Tol green — dashed baseline endpoints
_COLOR_ADJUSTED = '#228833'   # Tol green dotted — baseline-adjusted trace
_MAX_BASELINE_PEAKS = 5       # max pre-created artists per detector

# Cycling palette for snapshot reference lines
_SNAPSHOT_COLORS = [
    '#CC79A7',  # reddish purple
    '#009E73',  # bluish green
    '#E69F00',  # orange
    '#F0E442',  # yellow
    '#000000',  # black
]


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
            ax = self.fig.add_subplot(n_rows, n_cols, i + 1)
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
        self.fwhm_lines = []      # LineCollection for FWHM horizontal lines
        self.baseline_lcs = []    # LineCollection for baseline dashed segments
        self.adjusted_lines = []  # List[List[Line2D]] for baseline-adjusted traces

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
            fwhm_lc = LineCollection([], colors=COLOR_FWHM, linewidths=1.5, zorder=4)
            ax.add_collection(fwhm_lc)
            self.fwhm_lines.append(fwhm_lc)

            # Baseline artists (green dashed segment between baseline endpoints)
            baseline_lc = LineCollection([], colors=_COLOR_BASELINE, linewidths=1.2,
                                          linestyles='dashed', zorder=3)
            ax.add_collection(baseline_lc)
            self.baseline_lcs.append(baseline_lc)

            # Baseline-adjusted trace lines (one per possible peak)
            adj_lines_for_det = []
            for _ in range(_MAX_BASELINE_PEAKS):
                adj_line, = ax.plot([], [], color=_COLOR_ADJUSTED, linestyle='dotted',
                                    linewidth=1.0, alpha=0.7, zorder=3)
                adj_lines_for_det.append(adj_line)
            self.adjusted_lines.append(adj_lines_for_det)

        # Background for blitting
        self.background = None

        # Snapshot reference lines — list of dicts:
        #   {'label': str, 'alpha': float, 'visible': bool, 'lines': [Line2D|None, ...]}
        self.snapshots = []

    def resizeEvent(self, event):
        """Handle resize events to redraw plots properly"""
        super().resizeEvent(event)
        # Reset background on resize so blitting works correctly
        self.background = None
        self.fig.tight_layout()
        self.draw_idle()

    def take_snapshot(self, plot_data_list, alpha=0.3, label=None):
        """Capture current plot data as a static reference background line per subplot."""
        color = _SNAPSHOT_COLORS[len(self.snapshots) % len(_SNAPSHOT_COLORS)]
        if label is None:
            label = f"Snap {len(self.snapshots) + 1}"
        snap_lines = []
        for i, plot_data in enumerate(plot_data_list):
            if i >= len(self.axes):
                snap_lines.append(None)
                continue
            ax = self.axes[i]
            if plot_data.has_data and plot_data.is_enabled and len(plot_data.samples) > 0:
                (line,) = ax.plot(plot_data.samples, plot_data.values,
                                  color=color, linewidth=0.8, alpha=alpha, zorder=1.5)
            else:
                (line,) = ax.plot([], [], color=color, linewidth=0.8, alpha=alpha, zorder=1.5)
            snap_lines.append(line)
        self.snapshots.append({'label': label, 'alpha': alpha, 'visible': True, 'lines': snap_lines})
        self.background = None
        self.draw_idle()

    def remove_snapshot(self, idx):
        """Remove a snapshot by index and force re-blit."""
        if 0 <= idx < len(self.snapshots):
            for line in self.snapshots[idx]['lines']:
                if line is not None:
                    try:
                        line.remove()
                    except ValueError:
                        pass
            self.snapshots.pop(idx)
            self.background = None
            self.draw_idle()

    def set_snapshot_visible(self, idx, visible):
        """Toggle a snapshot's visibility and force re-blit."""
        if 0 <= idx < len(self.snapshots):
            self.snapshots[idx]['visible'] = visible
            for line in self.snapshots[idx]['lines']:
                if line is not None:
                    line.set_visible(visible)
            self.background = None
            self.draw_idle()

    def clear_all_snapshots(self):
        """Remove all snapshots."""
        for snap in self.snapshots:
            for line in snap['lines']:
                if line is not None:
                    try:
                        line.remove()
                    except ValueError:
                        pass
        self.snapshots.clear()
        self.background = None
        self.draw_idle()

    def init_blit(self):
        """Initialize background for blitting"""
        self.draw()
        self.background = self.copy_from_bbox(self.fig.bbox)

    def fast_update(self, plot_data_list: List[PlotData], show_baseline: bool = True, normalize: bool = True, shared_y: bool = False):
        """Fast update using blitting"""
        # Pre-compute shared y-range when requested (and not normalizing)
        if shared_y and not normalize:
            all_maxes = [
                float(np.max(pd.values))
                for pd in plot_data_list
                if pd.has_data and pd.is_enabled and len(pd.values) > 0
            ]
            global_max = max(all_maxes) if all_maxes else 1.0
            if global_max <= 0:
                global_max = 1.0
            shared_ylim = (-0.05 * global_max, 1.05 * global_max)
        else:
            shared_ylim = None

        # First, update xlim/ylim for all axes that have data and check if any changed
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
                    # Update ylim based on normalize / shared_y flags
                    current_ylim = ax.get_ylim()
                    if normalize:
                        new_ylim = (-0.1, 1.1)
                    elif shared_ylim is not None:
                        new_ylim = shared_ylim
                    else:
                        data_max = float(np.max(plot_data.values)) if len(plot_data.values) > 0 else 1.0
                        if data_max <= 0:
                            data_max = 1.0
                        new_ylim = (-0.05 * data_max, 1.05 * data_max)
                    if (abs(current_ylim[0] - new_ylim[0]) > abs(new_ylim[1]) * 0.01 or
                            abs(current_ylim[1] - new_ylim[1]) > abs(new_ylim[1]) * 0.01):
                        ax.set_ylim(new_ylim)
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
            for baseline_lc in self.baseline_lcs:
                baseline_lc.set_segments([])
            for adj_lines in self.adjusted_lines:
                for al in adj_lines:
                    al.set_data([], [])
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
            baseline_lc = self.baseline_lcs[i]
            adj_lines = self.adjusted_lines[i]

            # Handle different states
            if not plot_data.has_data:
                # No data available
                line.set_data([], [])
                scatter.set_offsets(np.empty((0, 2)))
                fwhm_lc.set_segments([])
                baseline_lc.set_segments([])
                for al in adj_lines:
                    al.set_data([], [])
                text.set_text('N/A')
                text.set_color(COLOR_GRAY)
                text.set_visible(True)
                ax.set_facecolor('white')

            elif not plot_data.is_enabled:
                # Detector disabled
                line.set_data([], [])
                scatter.set_offsets(np.empty((0, 2)))
                fwhm_lc.set_segments([])
                baseline_lc.set_segments([])
                for al in adj_lines:
                    al.set_data([], [])
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

                # Update baseline / adjusted-trace artists
                if show_baseline and plot_data.baseline_data:
                    bl_segments = []
                    for peak_idx, bd in enumerate(plot_data.baseline_data):
                        # Dashed green segment connecting baseline endpoints
                        bl_segments.append([(bd['bl_x'][0], bd['bl_y'][0]),
                                             (bd['bl_x'][1], bd['bl_y'][1])])
                        # Dotted adjusted trace
                        if peak_idx < len(adj_lines):
                            adj_lines[peak_idx].set_data(bd['adj_x'], bd['adj_y'])
                    baseline_lc.set_segments(bl_segments)
                    # Clear unused adjusted-trace lines
                    n_peaks = len(plot_data.baseline_data)
                    for k in range(n_peaks, len(adj_lines)):
                        adj_lines[k].set_data([], [])
                else:
                    baseline_lc.set_segments([])
                    for al in adj_lines:
                        al.set_data([], [])

            # Redraw this axes
            ax.draw_artist(line)
            ax.draw_artist(scatter)
            ax.draw_artist(fwhm_lc)
            ax.draw_artist(baseline_lc)
            for al in adj_lines:
                ax.draw_artist(al)
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
        self.last_fit_params = None

        # Angles from config (degrees, will be converted to radians)
        nxs_config = GlobalConfig.get_for_class('NXSLoader')
        self.angles_deg = np.array(
            nxs_config.get('angles', np.linspace(0, 337.5, 16).tolist())
            if nxs_config else np.linspace(0, 337.5, 16).tolist()
        )
        self.angles_rad = np.deg2rad(self.angles_deg)

    def set_angles(self, angles_deg):
        """Update the detector-angle mapping (degrees)."""
        self.angles_deg = np.array(angles_deg)
        self.angles_rad = np.deg2rad(self.angles_deg)

    def resizeEvent(self, event):
        """Handle resize events to redraw plots properly"""
        super().resizeEvent(event)
        # Reset background on resize so blitting works correctly
        self.background = None
        self.fig.tight_layout()
        self.draw_idle()
        self.last_fit_params = None

    def init_blit(self):
        """Initialize background for blitting"""
        self.draw()
        self.background = self.copy_from_bbox(self.fig.bbox)

    def update_polar_plot(self, results_df, peak_no=0, value_type='height', beta=2.0, setPlin=None, fitBeta=False, setPhi=None, fitPhi=True):
        """
        Update the polar plot with new data.

        Parameters:
            results_df: DataFrame with peak results
            peak_no: Which peak number to plot (0-indexed)
            value_type: 'height' or 'fwhm area'
            beta: Fixed beta value (used when fitBeta=False)
            setPlin: Fixed Plin value (used when fitBeta=True)
            fitBeta: If True, fit beta2 freely with Plin fixed to setPlin
            setPhi: Fixed phi value in radians (used when fitPhi=False)
            fitPhi: If True, phi is a free fit parameter; if False, phi is fixed to setPhi
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
                self._fit_and_plot(theta, r_values, beta, r_max, setPlin=setPlin, fitBeta=fitBeta, setPhi=setPhi, fitPhi=fitPhi)
            except Exception as e:
                print(f"Fit error: {e}")
                self.fit_text.set_text(f"Fit failed: {str(e)[:30]}")

        self._redraw_artists()

    def _fit_and_plot(self, theta, r_values, beta, r_max, setPlin=None, fitBeta=False, setPhi=None, fitPhi=True):
        """Fit the polarization model and update plot"""
        fit_kws = dict(method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-10, maxfev=5000)
        scale_guess = np.mean(r_values)
        beta0 = beta if beta != 0 else 1.0
        phi0 = setPhi if setPhi is not None else 0.0

        if fitBeta:
            # Plin is fixed
            plin_val = setPlin if setPlin is not None else 1.0
            plin_label = f"Plin: {plin_val:.4f} (fixed)"
            if fitPhi:
                # fit phi, beta2, scale
                def model(theta, phi, beta2, scale):
                    return polarization_model(theta, Plin=plin_val, phi=phi, beta2=beta2, scale=scale)
                initial_guess = [phi0, beta0, scale_guess]
                bounds = ([-np.pi, -4.0, 0], [np.pi, 4.0, np.inf])
                popt, pcov = curve_fit(model, theta, r_values, p0=initial_guess, bounds=bounds, **fit_kws)
                phi_fit, beta2_fit, scale_fit = popt
            else:
                # phi fixed; fit beta2, scale
                phi_fixed = setPhi if setPhi is not None else 0.0
                def model(theta, beta2, scale):
                    return polarization_model(theta, Plin=plin_val, phi=phi_fixed, beta2=beta2, scale=scale)
                initial_guess = [beta0, scale_guess]
                bounds = ([-4.0, 0], [4.0, np.inf])
                popt, pcov = curve_fit(model, theta, r_values, p0=initial_guess, bounds=bounds, **fit_kws)
                beta2_fit, scale_fit = popt
                phi_fit = phi_fixed
            Plin_fit = plin_val
            beta_label = f"β: {beta2_fit:.4f} (fitted)"
        else:
            # beta fixed
            beta2_fit = beta
            beta_label = f"β: {beta:.3f} (fixed)"
            if fitPhi:
                # fit Plin, phi, scale
                def model(theta, Plin, phi, scale):
                    return polarization_model(theta, Plin=Plin, phi=phi, beta2=beta, scale=scale)
                initial_guess = [0.2, phi0, scale_guess]
                bounds = ([0.0, -np.pi, 0], [2.0, np.pi, np.inf])
                popt, pcov = curve_fit(model, theta, r_values, p0=initial_guess, bounds=bounds, **fit_kws)
                Plin_fit, phi_fit, scale_fit = popt
            else:
                # phi fixed; fit Plin, scale
                phi_fixed = setPhi if setPhi is not None else 0.0
                def model(theta, Plin, scale):
                    return polarization_model(theta, Plin=Plin, phi=phi_fixed, beta2=beta, scale=scale)
                initial_guess = [0.2, scale_guess]
                bounds = ([0.0, 0], [2.0, np.inf])
                popt, pcov = curve_fit(model, theta, r_values, p0=initial_guess, bounds=bounds, **fit_kws)
                Plin_fit, scale_fit = popt
                phi_fit = phi_fixed
            plin_label = f"Plin: {Plin_fit:.4f}"

        self.last_fit_params = {
            'Plin': Plin_fit,
            'phi': phi_fit,
            'scale': scale_fit,
            'beta': beta2_fit,
            'pcov': pcov
        }

        # Generate smooth fit curve
        theta_fit = np.linspace(0, 2 * np.pi, 360)
        r_fit = polarization_model(theta_fit, Plin=Plin_fit, phi=phi_fit, beta2=beta2_fit, scale=scale_fit)

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
        fit_info = (f"{plin_label}\n"
                    f"φ: {phi_deg:.1f}°\n"
                    f"{beta_label}\n"
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


class AngularHeatmapCanvas(FigureCanvasQTAgg):
    """Polar pcolormesh canvas showing intensity vs detector angle and sample position.

    Uses the same interpolation logic as ``Plotter._buildIntensityGrid``.
    Redraws fully each update (no blitting) — only active when its tab is shown.
    """

    def __init__(self, parent=None, width=7, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Polar axes + narrow colorbar axes with fixed width ratio
        self.ax = self.fig.add_axes([0.05, 0.05, 0.78, 0.88], projection='polar')
        self.cax = self.fig.add_axes([0.87, 0.15, 0.03, 0.65])  # fixed colorbar slot
        self.ax.set_theta_zero_location('E')
        self.ax.set_theta_direction(1)
        self.ax.set_title('Angular Heatmap', fontsize=12)
        super().__init__(self.fig)

        # Detector-angle mapping from config (same source as PolarPlotCanvas)
        nxs_config = GlobalConfig.get_for_class('NXSLoader')
        self.angles_deg = np.array(
            nxs_config.get('angles', np.linspace(0, 337.5, 16).tolist())
            if nxs_config else np.linspace(0, 337.5, 16).tolist()
        )
        self.angles_rad = np.deg2rad(self.angles_deg)

        self._colorbar = None
        self._cbar_mappable = None

    def set_angles(self, angles_deg):
        """Update the detector-angle mapping (degrees)."""
        self.angles_deg = np.array(angles_deg)
        self.angles_rad = np.deg2rad(self.angles_deg)
        self._cbar_mappable = None

    # ------------------------------------------------------------------
    # Internal helpers (mirrors Plotter._buildIntensityGrid)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_intensity_grid(traces, angles_rad, n_theta=720):
        """Interpolate *traces* (n_det × n_sample) onto a uniform theta grid.

        Returns
        -------
        grid : ndarray, shape (n_theta, n_sample)
        theta_grid : ndarray, shape (n_theta,)
        """
        n_samples = traces.shape[1]
        theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        grid = np.zeros((n_theta, n_samples))

        sort_idx = np.argsort(angles_rad)
        sorted_angles = angles_rad[sort_idx]
        sorted_traces = traces[sort_idx]

        for si in range(n_samples):
            vals = sorted_traces[:, si]
            xp_ext = np.concatenate(
                [sorted_angles - 2 * np.pi, sorted_angles, sorted_angles + 2 * np.pi]
            )
            fp_ext = np.concatenate([vals, vals, vals])
            grid[:, si] = np.interp(theta_grid, xp_ext, fp_ext)

        return grid, theta_grid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_heatmap(
        self,
        plot_data_list: List['PlotData'],
        results_df,
        sample_min: int,
        sample_max: int,
        interpolate: bool = True,
        show_peaks: bool = True,
    ):
        """Rebuild the heatmap from current plot data and results.

        Parameters
        ----------
        plot_data_list : list of PlotData
            Plot-ready detector data (downsampled traces).
        results_df : pandas.DataFrame or None
            Peak-finding results (may be None / empty).
        sample_min, sample_max : int
            Displayed radial range (sample coordinates).
        interpolate : bool
            If True, smooth 720-point theta interpolation; else discrete wedges.
        show_peaks : bool
            Overlay peak scatter + width lines when True.
        """
        # Clear only the polar axes — cax (colorbar slot) keeps its position
        self.ax.cla()
        self.cax.cla()
        self.ax.set_theta_zero_location('E')
        self.ax.set_theta_direction(1)
        self.ax.set_title('Angular Heatmap', fontsize=12)

        # Collect enabled detectors that have data
        det_ids = []
        traces_list = []
        sample_coords = None

        for pd_obj in plot_data_list:
            if not pd_obj.has_data or not pd_obj.is_enabled:
                continue
            if len(pd_obj.samples) == 0:
                continue
            det_id = pd_obj.detector_id
            if det_id >= len(self.angles_deg):
                continue

            # Filter by sample range
            smin_idx = int(np.searchsorted(pd_obj.samples, sample_min))
            smax_idx = int(np.searchsorted(pd_obj.samples, sample_max, side='right'))
            smin_idx = max(0, smin_idx)
            smax_idx = min(len(pd_obj.samples), smax_idx)
            if smax_idx <= smin_idx:
                continue

            trace_slice = pd_obj.values[smin_idx:smax_idx]
            s_slice = pd_obj.samples[smin_idx:smax_idx]

            det_ids.append(det_id)
            traces_list.append(trace_slice)
            if sample_coords is None or len(s_slice) > len(sample_coords):
                sample_coords = s_slice

        if not det_ids or sample_coords is None or len(sample_coords) == 0:
            self.cax.set_visible(False)
            self.draw_idle()
            return
        self.cax.set_visible(True)

        # Align all traces to the common sample_coords length
        n_samples = len(sample_coords)
        traces = np.zeros((len(det_ids), n_samples))
        for idx, t in enumerate(traces_list):
            n = min(len(t), n_samples)
            traces[idx, :n] = t[:n]

        angles_rad = self.angles_rad[det_ids]

        # Radius bin edges
        if n_samples > 1:
            dr = sample_coords[1] - sample_coords[0]
        else:
            dr = 1.0
        r_edges = np.concatenate([[sample_coords[0] - dr / 2], sample_coords + dr / 2])

        vmin = traces.min()
        vmax = traces.max() if traces.max() > vmin else vmin + 1e-9

        if interpolate:
            grid, theta_grid = self._build_intensity_grid(traces, angles_rad, n_theta=720)
            d_theta = theta_grid[1] - theta_grid[0]
            theta_edges = np.append(theta_grid - d_theta / 2, theta_grid[-1] + d_theta / 2)
            mesh = self.ax.pcolormesh(
                theta_edges, r_edges, grid.T,
                cmap='viridis', vmin=vmin, vmax=vmax, shading='auto',
            )
        else:
            # Compute per-detector wedge half-widths as half the gap to each neighbour
            n_dets = len(angles_rad)
            if n_dets > 1:
                sorted_idx = np.argsort(angles_rad)
                sorted_ang = angles_rad[sorted_idx]
                gaps = np.diff(sorted_ang, append=sorted_ang[0] + 2 * np.pi)
                left_half = np.roll(gaps, 1) / 2
                right_half = gaps / 2
                inv_idx = np.argsort(sorted_idx)
                wedge_left = left_half[inv_idx]
                wedge_right = right_half[inv_idx]
            else:
                wedge_left = np.array([np.pi])
                wedge_right = np.array([np.pi])
            for idx in range(len(det_ids)):
                ang = angles_rad[idx]
                theta_edges = np.array([ang - wedge_left[idx], ang + wedge_right[idx]])
                C = traces[idx, :][np.newaxis, :]
                self.ax.pcolormesh(
                    theta_edges, r_edges, C.T,
                    cmap='viridis', vmin=vmin, vmax=vmax, shading='auto',
                )
            mesh = self.ax.pcolormesh(
                [0, 0.01], [r_edges[0], r_edges[-1]], [[vmin]],
                cmap='viridis', vmin=vmin, vmax=vmax, shading='auto',
            )
            mesh.set_visible(False)

        # Colourbar — drawn into the fixed cax slot, never touches self.ax geometry
        self._colorbar = self.fig.colorbar(mesh, cax=self.cax, label='Intensity')

        self.ax.set_rlim(r_edges[0], r_edges[-1])
        self.ax.spines['polar'].set_visible(False)

        # Peak overlay
        if show_peaks and results_df is not None and not results_df.empty:
            for _, peak_row in results_df.iterrows():
                det = int(peak_row['detector'])
                if det >= len(self.angles_rad):
                    continue
                ang = self.angles_rad[det]
                pos = peak_row['pos']
                wl = peak_row['width left']
                wr = peak_row['width right']
                self.ax.scatter([ang], [pos], color='red', s=20, zorder=6)
                self.ax.plot([ang, ang], [pos + wl, pos + wr],
                             color='red', linewidth=0.8, zorder=6)

        self.draw_idle()


class SingleDetectorCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for a single detector with interactive zoom/pan via toolbar"""

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Detector 0', fontsize=10)
        self.ax.tick_params(labelsize=8)
        self.ax.set_ylim([-0.1, 1.1])
        self.ax.set_xlim([0, 1000])
        self.ax.ticklabel_format(style='plain', axis='x', useOffset=False)
        self.ax.xaxis.get_major_formatter().set_scientific(False)
        self.fig.tight_layout(pad=1.5)
        super().__init__(self.fig)
        self.setMaximumHeight(520)

        self.line, = self.ax.plot([], [], color=COLOR_TRACE, linewidth=0.8, alpha=0.9)
        self.scatter = self.ax.scatter([], [], color=COLOR_PEAK, s=40, zorder=5)

        self.fwhm_lc = LineCollection([], colors=COLOR_FWHM, linewidths=1.5, zorder=4)
        self.ax.add_collection(self.fwhm_lc)

        self.baseline_lc = LineCollection([], colors=_COLOR_BASELINE, linewidths=1.2,
                                          linestyles='dashed', zorder=3)
        self.ax.add_collection(self.baseline_lc)

        self.adj_lines = []
        for _ in range(_MAX_BASELINE_PEAKS):
            adj_line, = self.ax.plot([], [], color=_COLOR_ADJUSTED, linestyle='dotted',
                                     linewidth=1.0, alpha=0.7, zorder=3)
            self.adj_lines.append(adj_line)

        self.text_obj = self.ax.text(0.5, 0.5, '', ha='center', va='center',
                                     transform=self.ax.transAxes, fontsize=12, color=COLOR_GRAY)
        self.text_obj.set_visible(False)

        # _user_navigated: set to True when user zooms/pans so we stop overriding limits
        self._user_navigated = False

        # Snapshot reference lines — list of dicts:
        #   {'label': str, 'alpha': float, 'visible': bool,
        #    'line': Line2D, 'plot_data_list': List[PlotData]}
        self.snapshots = []

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fig.tight_layout(pad=1.5)
        self.draw_idle()

    def take_snapshot(self, plot_data_list, alpha=0.3, label=None, current_det_idx=0):
        """Capture plot data as a static reference line for the active detector."""
        color = _SNAPSHOT_COLORS[len(self.snapshots) % len(_SNAPSHOT_COLORS)]
        if label is None:
            label = f"Snap {len(self.snapshots) + 1}"
        (line,) = self.ax.plot([], [], color=color, linewidth=0.8, alpha=alpha, zorder=1.5)
        if current_det_idx < len(plot_data_list):
            pd_snap = plot_data_list[current_det_idx]
            if pd_snap.has_data and pd_snap.is_enabled and len(pd_snap.samples) > 0:
                line.set_data(pd_snap.samples, pd_snap.values)
        self.snapshots.append({
            'label': label, 'alpha': alpha, 'visible': True,
            'line': line, 'plot_data_list': list(plot_data_list)
        })
        self.draw_idle()

    def remove_snapshot(self, idx):
        """Remove a snapshot by index."""
        if 0 <= idx < len(self.snapshots):
            try:
                self.snapshots[idx]['line'].remove()
            except ValueError:
                pass
            self.snapshots.pop(idx)
            self.draw_idle()

    def set_snapshot_visible(self, idx, visible):
        """Toggle a snapshot's visibility."""
        if 0 <= idx < len(self.snapshots):
            self.snapshots[idx]['visible'] = visible
            self.snapshots[idx]['line'].set_visible(visible)
            self.draw_idle()

    def clear_all_snapshots(self):
        """Remove all snapshots."""
        for snap in self.snapshots:
            try:
                snap['line'].remove()
            except ValueError:
                pass
        self.snapshots.clear()
        self.draw_idle()

    def update_plot(self, plot_data: PlotData, show_baseline: bool = True, normalize: bool = True, det_idx: int = 0):
        """Update the canvas with data for a single detector"""
        if plot_data is None or not plot_data.has_data:
            self.line.set_data([], [])
            self.scatter.set_offsets(np.empty((0, 2)))
            self.fwhm_lc.set_segments([])
            self.baseline_lc.set_segments([])
            for al in self.adj_lines:
                al.set_data([], [])
            self.text_obj.set_text('N/A')
            self.text_obj.set_color(COLOR_GRAY)
            self.text_obj.set_visible(True)
            self.ax.set_facecolor('white')
            self.draw_idle()
            return

        if not plot_data.is_enabled:
            self.line.set_data([], [])
            self.scatter.set_offsets(np.empty((0, 2)))
            self.fwhm_lc.set_segments([])
            self.baseline_lc.set_segments([])
            for al in self.adj_lines:
                al.set_data([], [])
            self.text_obj.set_text('OFF')
            self.text_obj.set_color(COLOR_DISABLED)
            self.text_obj.set_visible(True)
            self.ax.set_facecolor('#ffeeee')
            self.draw_idle()
            return

        self.text_obj.set_visible(False)
        self.ax.set_facecolor('white')

        if len(plot_data.samples) > 0:
            self.line.set_data(plot_data.samples, plot_data.values)
            if not self._user_navigated:
                xmin = float(plot_data.samples[0])
                xmax = float(plot_data.samples[-1])
                if xmax > xmin:
                    self.ax.set_xlim([xmin, xmax])
                if normalize:
                    self.ax.set_ylim([-0.1, 1.1])
                else:
                    self.ax.relim()
                    self.ax.autoscale_view(scalex=False, scaley=True)
        else:
            self.line.set_data([], [])

        if plot_data.peak_positions is not None:
            self.scatter.set_offsets(plot_data.peak_positions)
        else:
            self.scatter.set_offsets(np.empty((0, 2)))

        if plot_data.fwhm_lines is not None and len(plot_data.fwhm_lines) > 0:
            segments = []
            for fwhm in plot_data.fwhm_lines:
                pos, widthL, widthR, half_height = fwhm
                segments.append([(pos + widthL, half_height), (pos + widthR, half_height)])
            self.fwhm_lc.set_segments(segments)
        else:
            self.fwhm_lc.set_segments([])

        if show_baseline and plot_data.baseline_data:
            bl_segments = []
            for peak_idx, bd in enumerate(plot_data.baseline_data):
                bl_segments.append([(bd['bl_x'][0], bd['bl_y'][0]),
                                     (bd['bl_x'][1], bd['bl_y'][1])])
                if peak_idx < len(self.adj_lines):
                    self.adj_lines[peak_idx].set_data(bd['adj_x'], bd['adj_y'])
            self.baseline_lc.set_segments(bl_segments)
            n_peaks = len(plot_data.baseline_data)
            for k in range(n_peaks, len(self.adj_lines)):
                self.adj_lines[k].set_data([], [])
        else:
            self.baseline_lc.set_segments([])
            for al in self.adj_lines:
                al.set_data([], [])

        # Update snapshot lines for the active detector
        for snap in self.snapshots:
            pdl = snap['plot_data_list']
            if det_idx < len(pdl):
                pd_snap = pdl[det_idx]
                if pd_snap.has_data and pd_snap.is_enabled and len(pd_snap.samples) > 0:
                    snap['line'].set_data(pd_snap.samples, pd_snap.values)
                else:
                    snap['line'].set_data([], [])
            else:
                snap['line'].set_data([], [])

        self.draw_idle()
