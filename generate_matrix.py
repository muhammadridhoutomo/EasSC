import pandas as pd
import numpy as np
import requests
import ast
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def process_dataset_osrm(csv_name, dist_npy, dur_npy):
    print(f"--- Memproses {csv_name} via OSRM (Mobil) ---")
    
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, csv_name))
    except FileNotFoundError:
        print(f"❌ ERROR: File '{csv_name}' tidak ditemukan!")
        return

    # Ekstrak koordinat (longitude,latitude)
    coords = []
    for _, row in df.iterrows():
        if 'Latitude' in df.columns:
            coords.append(f"{row['Longitude']},{row['Latitude']}")
        else:
            c_dict = ast.literal_eval(row['Coordinate'])
            coords.append(f"{c_dict['lng']},{c_dict['lat']}")

    coords_string = ";".join(coords)
    print(f"Mengambil data jalan raya & durasi mobil untuk {len(coords)} lokasi...")

    # Tembak OSRM Table API (Car profile)
    # annotations=distance (meter) dan duration (detik)
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_string}?annotations=distance,duration"

    try:
        response = requests.get(url, timeout=30)
        data = response.json()

        if data['code'] == 'Ok':
            # 1. Matriks Jarak (Meter -> Kilometer)
            dist_matrix = np.array(data['distances']) / 1000.0
            dist_matrix = np.nan_to_num(dist_matrix, nan=999.0)
            np.save(os.path.join(BASE_DIR, dist_npy), dist_matrix)
            
            # 2. Matriks Durasi (Detik -> Menit)
            dur_matrix = np.array(data['durations']) / 60.0
            dur_matrix = np.nan_to_num(dur_matrix, nan=120.0) # Default 2 jam jika terisolasi
            np.save(os.path.join(BASE_DIR, dur_npy), dur_matrix)
            
            print(f"✅ Sukses! Jarak disimpan di '{dist_npy}', Durasi di '{dur_npy}'.")
        else:
            print(f"❌ OSRM error: {data.get('message')}")
            
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    # Dataset Lama
    process_dataset_osrm('wisata_sejarah.csv', 'matriks_wisata.npy', 'matriks_durasi.npy')
    # Dataset Baru
    process_dataset_osrm('historical_new.csv', 'matriks_wisata_new.npy', 'matriks_durasi_new.npy')
