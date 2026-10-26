"""
Script to download authentic Numenta Anomaly Benchmark (NAB) benchmark datasets
and ground-truth anomaly windows from GitHub.
"""

import os
import json
import urllib.request
import ssl

BASE_RAW_URL = "https://raw.githubusercontent.com/numenta/NAB/master"

DATASETS = {
    # File name -> relative path in NAB repo
    "machine_temperature_system_failure.csv": "data/realKnownCause/machine_temperature_system_failure.csv",
    "ec2_cpu_utilization_5f5533.csv": "data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    "art_daily_flatmiddle.csv": "data/artificialWithAnomaly/art_daily_flatmiddle.csv",
    "ambient_temperature_system_failure.csv": "data/realKnownCause/ambient_temperature_system_failure.csv",
    "ec2_cpu_utilization_c6585a.csv": "data/realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv",
    "combined_windows.json": "labels/combined_windows.json",
    "combined_labels.json": "labels/combined_labels.json",
}

# F1 Domain Mapping Metadata
F1_CHANNEL_MAPPING = {
    "machine_temperature_system_failure.csv": {
        "channel_name": "engine_temperature_sensor",
        "description": "Internal Combustion Engine (ICE) cylinder head & coolant temperature sensor (°C)",
        "unit": "°C",
        "normal_range": [70.0, 95.0],
        "anomaly_description": "Catastrophic cooling radiator blockage and thermal runaway warning"
    },
    "ec2_cpu_utilization_5f5533.csv": {
        "channel_name": "ers_mgu_k_load_pct",
        "description": "Motor Generator Unit - Kinetic (MGU-K) electrical duty cycle (% capacity)",
        "unit": "%",
        "normal_range": [20.0, 60.0],
        "anomaly_description": "MGU-K harvesting inverter desynchronization and sudden energy cutoff"
    },
    "art_daily_flatmiddle.csv": {
        "channel_name": "lap_time_stream_delta",
        "description": "Sector 2 to Sector 3 relative lap pace delta vs baseline target pace (seconds)",
        "unit": "s",
        "normal_range": [-0.5, 0.5],
        "anomaly_description": "Safety car deployment / Virtual Safety Car delta time violation"
    },
    "ambient_temperature_system_failure.csv": {
        "channel_name": "tyre_carcass_temp_fl",
        "description": "Front-Left tyre carcass bulk core temperature under lateral load (°C)",
        "unit": "°C",
        "normal_range": [85.0, 110.0],
        "anomaly_description": "Severe blistering and sudden puncture delamination failure"
    }
}

def download_data(target_dir="data/nab"):
    os.makedirs(target_dir, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f"Downloading NAB benchmark datasets into {target_dir}...")
    for filename, rel_path in DATASETS.items():
        out_path = os.path.join(target_dir, filename)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f"  [EXISTS] {filename} ({os.path.getsize(out_path)} bytes)")
            continue
        
        url = f"{BASE_RAW_URL}/{rel_path}"
        print(f"  [FETCHING] {url} -> {out_path}...")
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, context=ctx, timeout=30) as response, open(out_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print(f"  [DONE] {filename} ({len(data)} bytes)")
        except Exception as e:
            print(f"  [ERROR] Failed to download {url}: {e}")

    # Save domain mapping json
    mapping_path = os.path.join(target_dir, "f1_channel_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(F1_CHANNEL_MAPPING, f, indent=2)
    print(f"  [SAVED] F1 domain mapping saved to {mapping_path}")

if __name__ == "__main__":
    download_data()
