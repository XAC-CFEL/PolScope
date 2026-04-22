import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit
from typing import List

from ToFPipeline.ToFPipeline import GlobalConfig, polarization_model
from colors import (COLOR_TRACE, COLOR_PEAK, COLOR_FWHM, COLOR_DATA,
                    COLOR_FIT, COLOR_PHI, COLOR_DISABLED, COLOR_GRAY)
from models import PlotData


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
        self.last_fit_params = None

        # Angles from config (degrees, will be converted to radians)
        nxs_config = GlobalConfig.get_for_class('NXSLoader')
        self.angles_deg = np.array(
            nxs_config.get('angles', np.linspace(0, 337.5, 16).tolist())
            if nxs_config else np.linspace(0, 337.5, 16).tolist()
        )
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
