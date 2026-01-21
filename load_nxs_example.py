#!/usr/bin/env python3
"""
Example script demonstrating how to load .nxs files using the NXSLoader class.

This script shows how to:
1. Load configuration
2. Initialize the NXSLoader
3. Load data from .nxs files
4. Apply preprocessing
5. Examine the resulting xarray DataArray structure

The resulting DataArray has the format:
- Name: 'adc00'
- Dimensions: ('detector', 'pulse', 'sample')
- Coordinates: detector, pulse (MultiIndex), sample, daq_run
- Pulse MultiIndex: (trainId, pulseId)
"""

import numpy as np
from pathlib import Path
from ToFPipeline.ToFPipeline import GlobalConfig, NXSLoader

def main():
    # Load configuration
    config_path = "ToFPipeline/config.yaml"
    GlobalConfig.load(config_path)
    print(f"Loaded configuration from {config_path}")
    
    # Initialize the NXSLoader
    data_path = Path("data")  # Path to your .nxs files
    
    # Option 1: Auto-detect all runs (recommended)
    loader = NXSLoader(data_path=data_path)
    print("Loading all available .nxs files...")
    loader.load()
    
    # Option 2: Load specific run numbers
    # loader = NXSLoader(data_path=data_path, run_numbers=[1999, 2000, 2119, 2120])
    # loader.load()
    
    print("Data loaded successfully!")
    print(f"Data shape: {loader.data.shape}")
    print(f"Data dimensions: {loader.data.dims}")
    print(f"Data name: {loader.data.name}")
    
    # Show the structure
    print("\n=== DataArray Structure ===")
    print(f"Dimensions: {loader.data.dims}")
    print(f"Coordinates: {list(loader.data.coords.keys())}")
    
    # Pulse MultiIndex details
    pulse_index = loader.data.pulse.to_index()
    print(f"Pulse is MultiIndex: {hasattr(pulse_index, 'levels')}")
    if hasattr(pulse_index, 'levels'):
        print(f"  - Level names: {pulse_index.names}")
        print(f"  - Number of unique trainIds: {len(pulse_index.get_level_values('trainId').unique())}")
        print(f"  - Number of unique pulseIds: {len(pulse_index.get_level_values('pulseId').unique())}")
    
    # Show coordinate ranges
    print(f"\nCoordinate ranges:")
    print(f"  - Detectors: {loader.data.detector.values[0]} to {loader.data.detector.values[-1]} ({len(loader.data.detector)} total)")
    print(f"  - Pulses: {len(loader.data.pulse)} total")
    print(f"  - Samples: {loader.data.sample.values[0]} to {loader.data.sample.values[-1]} ({len(loader.data.sample)} total)")
    print(f"  - Run numbers: {np.unique(loader.data.daq_run.values)}")
    
    # Apply preprocessing (optional)
    print("\n=== Applying Preprocessing ===")
    loader_preprocessed = NXSLoader(data_path=data_path, run_numbers=[1999, 2000])  # Smaller subset for demo
    loader_preprocessed.load()
    
    loader_preprocessed.defaultPreprocessing(
        ToF=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # All detectors
        baselineRegion=[0, 10]  # Use first 10 samples for baseline correction
    )
    
    print(f"Preprocessed data shape: {loader_preprocessed.data.shape}")
    print(f"Selected detectors: {loader_preprocessed.data.detector.values}")
    
    # Example data access patterns
    print("\n=== Data Access Examples ===")
    
    # 1. Access specific detector, pulse, sample
    sample_value = loader.data.isel(detector=0, pulse=0, sample=50)
    print(f"Value at detector=0, pulse=0, sample=50: {sample_value.values}")
    
    # 2. Access all data for first detector
    first_detector = loader.data.isel(detector=0)
    print(f"First detector data shape: {first_detector.shape}")
    
    # 3. Access data for specific run
    first_run = np.unique(loader.data.daq_run.values)[0]
    run_data = loader.data.where(loader.data.daq_run == first_run, drop=True)
    print(f"Data for run {first_run} shape: {run_data.shape}")
    
    # 4. Access slice of sample dimension
    tof_slice = loader.data.isel(sample=slice(40, 60))
    print(f"Time-of-flight slice (samples 40-60) shape: {tof_slice.shape}")
    
    # 5. Mean across pulses for each detector
    detector_means = loader.data.mean(dim='pulse')
    print(f"Detector means shape: {detector_means.shape}")
    
    print("\n=== Format Verification ===")
    print("Checking against your required format:")
    print(f"✓ DataArray name 'adc00': {loader.data.name == 'adc00'}")
    print(f"✓ Dimensions (detector, pulse, sample): {loader.data.dims == ('detector', 'pulse', 'sample')}")
    print(f"✓ Has daq_run coordinate: {'daq_run' in loader.data.coords}")
    print(f"✓ Pulse is MultiIndex with trainId/pulseId: {pulse_index.names == ['trainId', 'pulseId']}")
    print(f"✓ Data type is numeric: {np.issubdtype(loader.data.dtype, np.number)}")
    print(f"✓ Has underlying dask array: {hasattr(loader.data.data, 'chunks')}")
    if hasattr(loader.data.data, 'chunks'):
        print(f"  - Chunk sizes: {loader.data.data.chunks}")
    
    print("\nSUCCESS: Your .nxs files have been loaded into the required format!")
    
    return loader

if __name__ == "__main__":
    loader = main()