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
