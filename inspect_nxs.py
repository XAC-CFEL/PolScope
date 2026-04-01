#!/usr/bin/env python3
import h5py

nxs_file = 'data/branch2_CB_Ne_02119.nxs'

print("=== Detector-Related Information in .nxs File ===\n")
with h5py.File(nxs_file, 'r') as f:
    def search_detector_info(name, obj):
        lower_name = name.lower()
        if any(keyword in lower_name for keyword in ['detector', 'angle', 'channel', 'ch', 'histogram']):
            print(f"\n{name}")
            if isinstance(obj, h5py.Dataset):
                try:
                    data = obj[...]
                    if data.size < 100:
                        print(f"  Data: {data}")
                    else:
                        print(f"  Shape: {data.shape}, dtype: {data.dtype}")
                except:
                    print(f"  (Unable to read data)")
            if obj.attrs:
                for key, value in obj.attrs.items():
                    attr_str = str(value)[:200]  # Truncate long attributes
                    print(f"  @{key}: {attr_str}")
    
    f.visititems(search_detector_info)

print("\n=== Summary ===")
print("Detector numbers: Extracted from histogram channel names (ch01-ch16)")
print("Detector angles: NOT stored in .nxs files - defined in configuration (config.yaml)")
