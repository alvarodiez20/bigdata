import pandas as pd
import time

def try_naive_load(filename):
    print(f"Attempting to load {filename} into memory...")
    try:
        start_time = time.time()
        # This is where the magic (and the crash) happens
        df = pd.read_csv(filename) 
        print(f"Success! Loaded {len(df)} rows in {time.time() - start_time:.2f}s")
    except MemoryError:
        print("CRASH: Python ran out of memory!")
    except Exception as e:
        print(f"The system or kernel likely killed the process: {e}")

if __name__ == "__main__":
    try_naive_load("massive_data.csv")