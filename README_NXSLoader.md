# NXS File Loader for ToF Pipeline

This document describes how to use the `NXSLoader` class to load time-of-flight histogram data from .nxs files into your pipeline.

## Overview

The `NXSLoader` class loads .nxs files containing time-of-flight histogram data and converts them into an `xarray.DataArray` with the following structure:

```
xarray.DataArray 'adc00'
Dimensions: ('detector', 'pulse', 'sample')
Coordinates:
- detector: int64 array of detector indices (0-14)
- pulse: MultiIndex with levels ['trainId', 'pulseId'] 
- sample: int64 array of sample indices (0-100)
- daq_run: uint32 array with run numbers for each pulse
- trainId: uint32 values (accessible via pulse MultiIndex)
- pulseId: int64 values (accessible via pulse MultiIndex)

Underlying data: dask.array with configurable chunking
```

## Usage

### Basic Usage

```python
from ToFPipeline import GlobalConfig, NXSLoader
from pathlib import Path

# Load configuration
GlobalConfig.load('config.yaml')

# Initialize loader
loader = NXSLoader(data_path=Path('data'))

# Load all .nxs files (auto-detection)
loader.load()

# Access the data
data = loader.data  # xarray.DataArray in required format
```

### Loading Specific Runs

```python
# Load only specific run numbers
loader = NXSLoader(data_path=Path('data'), run_numbers=[1999, 2000, 2119])
loader.load()
```

### Preprocessing

```python
# Apply baseline correction and detector selection
loader.defaultPreprocessing(
    ToF=[0, 1, 2, 3, 4],      # Select detectors 0-4
    baselineRegion=[0, 10]     # Use samples 0-10 for baseline
)
```

## Configuration

Add configuration to your `config.yaml`:

```yaml
NXSLoader:
    ToF: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]  # Available detectors
    angles: [0.0,22.5,45.0,67.5,90.0,112.5,135.0,157.5,180.0,202.5,225.0,247.5,270.0,292.5,315.0]  # Detector angles
    file_pattern: "*_{run_number:05d}.nxs"      # Filename pattern
    n_samples: 101                              # Number of time-of-flight samples
    baselineRegion: [0,10]                      # Default baseline region
    trainStart: null                            # Train slicing (optional)
    trainStop: null
```

## Data Access Patterns

```python
# Access specific detector, pulse, sample
value = data.isel(detector=0, pulse=10, sample=50)

# Get data for specific run
run_data = data.where(data.daq_run == 1999, drop=True)

# Get all data for first detector
detector_data = data.isel(detector=0)

# Get time-of-flight slice
tof_slice = data.isel(sample=slice(40, 60))

# Calculate mean across pulses
detector_means = data.mean(dim='pulse')

# Access MultiIndex components
pulse_index = data.pulse.to_index()
train_ids = pulse_index.get_level_values('trainId')
pulse_ids = pulse_index.get_level_values('pulseId')
```

## File Format Requirements

The NXSLoader expects .nxs files with the following structure:

- Histogram data in `/scan/instrument/histogram_ch01/data` through `/scan/instrument/histogram_ch15/data`
- Time-of-flight axes in `/scan/instrument/histogram_chXX/time_of_flight`
- Timestamps in `/scan/instrument/collection/timestamp`
- Each file represents one run with a unique run number extractable from the filename

## Error Handling

- Missing channels are filled with zeros and a warning is printed
- Files that can't be loaded are skipped with an error message
- Run numbers are extracted from filenames using regex pattern matching
- Automatic fallback patterns are tried if the primary pattern fails

## Performance Notes

- Data is loaded as dask arrays with configurable chunking for memory efficiency
- Large datasets can be processed without loading everything into memory
- Preprocessing operations work with dask arrays for scalability

## Example Script

See `load_nxs_example.py` for a complete working example that demonstrates:
- Loading configuration
- Auto-detecting and loading all .nxs files
- Applying preprocessing
- Data access patterns
- Format verification