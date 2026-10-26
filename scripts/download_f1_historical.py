"""
Downloads complete historical Formula 1 World Championship data from 1950 to 2026.
Sources canonical relational tables: races, results, drivers, constructors, status,
pit_stops, circuits, and safety_cars.
"""

import os
import sys
import urllib.request
import ssl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "https://raw.githubusercontent.com/TracingInsights/RaceData/main/data"
TARGET_DIR = "data/f1_historical"

HISTORICAL_FILES = [
    "races.csv",
    "results.csv",
    "drivers.csv",
    "constructors.csv",
    "status.csv",
    "pit_stops.csv",
    "circuits.csv",
    "safety_cars.csv"
]

def download_historical_data():
    os.makedirs(TARGET_DIR, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f"Downloading complete 1950-2026 F1 Historical Archive into '{TARGET_DIR}'...")
    for filename in HISTORICAL_FILES:
        dest_path = os.path.join(TARGET_DIR, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
            print(f"  [EXISTS] {filename} ({os.path.getsize(dest_path):,} bytes)")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"  [FETCHING] {url} -> {dest_path}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp, open(dest_path, "wb") as f_out:
                data = resp.read()
                f_out.write(data)
            print(f"  [DONE] {filename} ({len(data):,} bytes)")
        except Exception as e:
            print(f"  [WARNING] Could not fetch {filename}: {e}")

    print("Historical F1 data download complete.")

if __name__ == "__main__":
    download_historical_data()
