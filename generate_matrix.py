import pandas as pd
import numpy as np
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def haversine(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak antara dua titik koordinat dalam Kilometer 
    menggunakan rumus Haversine.
    """
    # Radius bumi dalam kilometer
    R = 6371.0
    
    # Ubah koordinat ke radian
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    
    a = math.sin(d_lat / 2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(d_lon / 2)**2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def process_dataset_haversine(csv_name, npy_name):
    print(f"--- Memproses {csv_name} menggunakan Rumus Haversine ---")
    
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, csv_name))
    except FileNotFoundError:
        print(f"❌ ERROR: File '{csv_name}' tidak ditemukan!")
        return

    num_locations = len(df)
    dist_matrix = np.zeros((num_locations, num_locations))

    # Ambil koordinat
    coords = []
    for _, row in df.iterrows():
        if 'Latitude' in df.columns:
            coords.append((row['Latitude'], row['Longitude']))
        else:
            # Format lama (parsing Coordinate string)
            import ast
            c_dict = ast.literal_eval(row['Coordinate'])
            coords.append((c_dict['lat'], c_dict['lng']))

    # Hitung matriks jarak titik-ke-titik
    for i in range(num_locations):
        for j in range(num_locations):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                dist_matrix[i, j] = haversine(lat1, lon1, lat2, lon2)

    # Simpan matriks
    np.save(os.path.join(BASE_DIR, npy_name), dist_matrix)
    print(f"✅ Matriks sukses dihitung (Haversine) dan disimpan sebagai '{npy_name}'!")

if __name__ == "__main__":
    process_dataset_haversine('wisata_sejarah.csv', 'matriks_wisata.npy')
    process_dataset_haversine('historical_new.csv', 'matriks_wisata_new.npy')
