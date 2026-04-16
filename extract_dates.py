import h5py
import os

def extract_start_end_times(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            start_time = f['scan/start_time'][()]
            end_time = f['scan/end_time'][()]
            print(f"\nFile: {file_path}")
            print(f"Start Time: {start_time}")
            print(f"End Time: {end_time}")
    except KeyError as e:
        print(f"Key not found in file {file_path}: {e}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

def main():
    data_dir = "data"
    files = [f for f in os.listdir(data_dir) if f.endswith('.nxs')]

    for file in files:
        file_path = os.path.join(data_dir, file)
        extract_start_end_times(file_path)

if __name__ == "__main__":
    main()