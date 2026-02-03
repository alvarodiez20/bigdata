import csv
import os

def generate_giant_csv(filename, target_gb=10):
    # Approximate size of one row (~100 bytes)
    row = ["ID", "Name", "Value", "Description", "Status"]
    data_row = [1234567, "John Doe", 99.99, "This is a long string to take up space " * 5, "ACTIVE"]
    
    bytes_per_row = 250 # Rough estimate
    iterations = int((target_gb * 1024**3) / bytes_per_row)

    print(f"Generating ~{target_gb}GB CSV file. This might take a minute...")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        for i in range(iterations):
            writer.writerow([i, f"User_{i}", i * 1.5, "Data" * 20, "OK"])
            if i % 1000000 == 0:
                print(f"Written {i} rows...")

    print(f"Done! Created {filename} ({os.path.getsize(filename) / 1e9:.2f} GB)")

if __name__ == "__main__":
    generate_giant_csv("massive_data.csv", target_gb=12) # Adjust based on your RAM