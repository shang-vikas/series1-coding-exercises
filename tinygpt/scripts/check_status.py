import os
import sys

path = "data/raw/raw.txt"
status_file = "download_status.txt"

if os.path.exists(path):
    size = os.path.getsize(path)
    
    # Count characters
    char_count = 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(1024*1024)  # 1MB chunks
                if not chunk:
                    break
                char_count += len(chunk)
    except Exception as e:
        char_count = size  # Fallback
    
    with open(status_file, 'w') as f:
        f.write(f"File exists: Yes\n")
        f.write(f"Size bytes: {size}\n")
        f.write(f"Size MB: {size / 1024 / 1024:.2f}\n")
        f.write(f"Character count: {char_count}\n")
    
    print(f"Size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
    print(f"Character count: {char_count:,}")
else:
    with open(status_file, 'w') as f:
        f.write("File exists: No\n")
    print("File does not exist yet")
