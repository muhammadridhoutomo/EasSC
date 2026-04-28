import pandas as pd
import numpy as np
import requests
import ast
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Memulai proses pembuatan matriks jarak...")

# 1. Baca dataset 32 titik
try:
    df = pd.read_csv(os.path.join(BASE_DIR, 'wisata_sejarah.csv'))
except FileNotFoundError:
    print("❌ ERROR: File 'wisata_sejarah.csv' tidak ditemukan!")
    exit()

# 2. Ekstrak koordinat (OSRM membutuhkan format: longitude,latitude)
coords = []
for coord_str in df['Coordinate']:
    coord_dict = ast.literal_eval(coord_str)
    coords.append(f"{coord_dict['lng']},{coord_dict['lat']}")

coords_string = ";".join(coords)

print(f"Mengambil jarak asli jalan raya untuk {len(coords)} lokasi...")

# 3. Tembak endpoint OSRM Table API
url = f"http://router.project-osrm.org/table/v1/driving/{coords_string}?annotations=distance"

try:
    response = requests.get(url)
    data = response.json()

    if data['code'] == 'Ok':
        # Jarak dari OSRM dalam satuan meter, kita konversi ke kilometer
        distance_matrix = np.array(data['distances']) / 1000.0
        
        # Opsi: Jika ada titik yang terisolasi (None/Null), ubah jadi jarak tak terhingga
        distance_matrix = np.nan_to_num(distance_matrix, nan=float('inf'))
        
        # 4. Simpan matriks ke file .npy
        nama_file_npy = 'matriks_wisata.npy'
        np.save(os.path.join(BASE_DIR, nama_file_npy), distance_matrix)
        print(f"✅ Matriks sukses dibuat dan disimpan sebagai '{nama_file_npy}'!")
    else:
        print(f"❌ OSRM merespon dengan error: {data.get('message', 'Unknown Error')}")

except Exception as e:
    print(f"❌ Terjadi kesalahan saat menghubungi API: {e}")