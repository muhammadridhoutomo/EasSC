import pandas as pd
import numpy as np
import random
import time
import ast
import folium
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# BLOK 1: PERSIAPAN DATA (FULL REAL DISTANCE)
# =====================================================================

print("==========================================================")
print("  ROUTE PLANNING GA - WISATA BUDAYA SURABAYA & SEMARANG")
print("==========================================================")
print("Loading data asli jalan raya...")

try:
    df_label = pd.read_csv(os.path.join(BASE_DIR, 'daftar_tempat_full.csv'))
    daftar_tempat = df_label['Nama Tempat'].tolist()
    jumlah_tempat = len(daftar_tempat)
except FileNotFoundError:
    print("❌ ERROR: File 'daftar_tempat_full.csv' tidak ditemukan!")
    exit()

try:
    distance_matrix = np.load(os.path.join(BASE_DIR, 'matriks_full_sby_smg.npy'))
except FileNotFoundError:
    print("❌ ERROR: File 'matriks_full_sby_smg.npy' tidak ditemukan!")
    exit()

def inisialisasi_populasi(ukuran_populasi, jumlah_tempat):
    populasi = []
    for _ in range(ukuran_populasi):
        individu = list(range(jumlah_tempat))
        random.shuffle(individu)
        populasi.append(individu)
    return populasi

def hitung_total_jarak(rute):
    jarak = 0
    for i in range(len(rute) - 1):
        jarak += distance_matrix[rute[i]][rute[i+1]]
    jarak += distance_matrix[rute[-1]][rute[0]]
    return jarak

def evaluasi_fitness(populasi):
    fitness_list = []
    for rute in populasi:
        jarak = hitung_total_jarak(rute)
        fitness = 1.0 / float(jarak) if jarak > 0 else 0
        fitness_list.append(fitness) 
    return fitness_list

# =====================================================================
# VVVVV === START: BAGIAN MODIFIKASI OPERATOR GA === VVVVV
# =====================================================================

# --- VERSI 2 (ORANG 2: Kompetitif) ---

# 1. SELEKSI: Tournament Selection
def seleksi(populasi, fitness_list):
    ukuran_turnamen = 3  # <--- UBAH DARI 5 JADI 3
    
    idx_turnamen_1 = random.sample(range(len(populasi)), ukuran_turnamen)
    best_idx_1 = max(idx_turnamen_1, key=lambda i: fitness_list[i])
    parent1 = populasi[best_idx_1]
    
    idx_turnamen_2 = random.sample(range(len(populasi)), ukuran_turnamen)
    best_idx_2 = max(idx_turnamen_2, key=lambda i: fitness_list[i])
    parent2 = populasi[best_idx_2]
    
    return parent1, parent2

# 2. CROSSOVER: Partially Mapped Crossover (PMX) - DIPERBAIKI
def crossover(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1.copy()

    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    
    # a. Copy segmen utama dari parent1
    child[start:end] = parent1[start:end]
    
    # b. Mapping untuk mengisi nilai agar tidak ada duplikat
    for i in range(start, end):
        # Jika elemen di parent2 belum ada di child
        if parent2[i] not in child:
            current_val = parent1[i]
            idx_in_p2 = parent2.index(current_val)
            
            # Telusuri jika posisi di parent2 ternyata jatuh di dalam segmen
            while start <= idx_in_p2 < end:
                current_val = parent1[idx_in_p2]
                idx_in_p2 = parent2.index(current_val)
                
            # Tempatkan elemen parent2[i] di posisi yang aman
            child[idx_in_p2] = parent2[i]
            
    # c. Sisa tempat kosong (yang bernilai -1) diisi dengan parent2
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
            
    return child

# 3. MUTASI: Inversion Mutation
def mutasi(individu, mutation_rate):
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(individu)), 2))
        individu[start:end] = reversed(individu[start:end])
    return individu

# =====================================================================
# ^^^^^ === END: BAGIAN MODIFIKASI OPERATOR GA === ^^^^^
# =====================================================================

# =====================================================================
# BLOK 3: MAIN LOOP GA (EVOLUSI)
# =====================================================================

ukuran_populasi = 200 
generasi_maks = 1000  
crossover_rate = 0.8
mutation_rate = 0.3

print(f"\nMemulai Pencarian Rute Optimal...")
print(f"Total Tempat   : {jumlah_tempat} Lokasi (SBY & SMG)")
waktu_mulai = time.time()

populasi = inisialisasi_populasi(ukuran_populasi, jumlah_tempat)
best_route_overall = None
best_distance_overall = float('inf')

for gen in range(generasi_maks):
    fitness_list = evaluasi_fitness(populasi)
    
    max_fitness_idx = np.argmax(fitness_list)
    current_best_route = populasi[max_fitness_idx]
    current_best_distance = hitung_total_jarak(current_best_route)
    
    if current_best_distance < best_distance_overall:
        best_distance_overall = current_best_distance
        best_route_overall = current_best_route.copy() # <--- TAMBAHKAN .copy() agar tidak tertimpa
        
    new_populasi = [best_route_overall] # Elitism
    
    while len(new_populasi) < ukuran_populasi:
        p1, p2 = seleksi(populasi, fitness_list)
        child = crossover(p1, p2, crossover_rate)
        
        if p1 == p2:
            child = mutasi(child, 1.0)
        else:
            child = mutasi(child, mutation_rate)
            
        new_populasi.append(child)
        
    populasi = new_populasi
    
    if (gen + 1) % 100 == 0:
        print(f"Generasi {gen + 1:<4} | Jarak Terpendek Sementara: {best_distance_overall:.2f} km")

durasi = time.time() - waktu_mulai

print("\n" + "=" * 58)
print("✨ HASIL AKHIR RUTE OPTIMAL ✨")
print("=" * 58)
print(f"Waktu Komputasi    : {durasi:.2f} detik")
print(f"Total Jarak Tempuh : {best_distance_overall:.2f} km\n")

# =====================================================================
# BLOK 4: AUTO-GENERATE VISUALISASI PETA DENGAN ANGKA URUTAN
# =====================================================================
import os
import requests

print("\n⏳ Sedang mengunduh jalur jalan raya asli dan membuat peta...")

# 1. Mengambil data koordinat dari dataset asli
df_asli = pd.read_csv(os.path.join(BASE_DIR, 'wisata_budaya.csv'))
koordinat_list = []

for nama_lengkap in daftar_tempat:
    nama_bersih = nama_lengkap.split(' (')[0].strip()
    match = df_asli[df_asli['Place_Name'].str.contains(nama_bersih, case=False, na=False)]
    
    if not match.empty:
        coord_dict = ast.literal_eval(match.iloc[0]['Coordinate'])
        koordinat_list.append((coord_dict['lat'], coord_dict['lng']))
    else:
        koordinat_list.append((-7.25, 112.75)) # Fallback Surabaya

# 2. Urutkan Koordinat Sesuai Rute
rute_urut = []
for indeks in best_route_overall:
    rute_urut.append(koordinat_list[indeks])
# Tutup rute agar melingkar kembali ke awal
rute_urut.append(koordinat_list[best_route_overall[0]]) 

# 3. Minta Geometri dari OSRM
osrm_coords = ";".join([f"{lng},{lat}" for lat, lng in rute_urut])
route_url = f"http://router.project-osrm.org/route/v1/driving/{osrm_coords}?geometries=geojson&overview=full"

# 4. Setup Peta
peta = folium.Map(location=[-7.1, 111.5], zoom_start=8)

# 5. Pasang Pin dengan ANGKA URUTAN
for urutan, indeks_tempat in enumerate(best_route_overall):
    lat, lng = koordinat_list[indeks_tempat]
    nama_tempat = daftar_tempat[indeks_tempat]
    
    # Tentukan warna: Merah untuk Surabaya, Biru untuk Semarang
    warna_garis = '#d32f2f' if '(SBY)' in nama_tempat else '#1976d2'
    warna_bg = '#ffcdd2' if '(SBY)' in nama_tempat else '#bbdefb'
    
    # Membuat icon angka kustom menggunakan HTML (DivIcon)
    icon_angka = folium.DivIcon(
        html=f'''
        <div style="
            font-size: 11pt; 
            font-weight: bold; 
            color: black; 
            background-color: {warna_bg}; 
            border: 3px solid {warna_garis}; 
            border-radius: 50%; 
            text-align: center; 
            line-height: 28px;
            width: 30px; 
            height: 30px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
        ">
            {urutan + 1}
        </div>
        '''
    )
    
    folium.Marker(
        location=[lat, lng],
        popup=f"<b>Urutan ke-{urutan+1}</b><br>{nama_tempat}",
        tooltip=f"{urutan+1}. {nama_tempat}", # Saat di-hover muncul urutan & nama
        icon=icon_angka
    ).add_to(peta)

# 6. Gambar Jalur di Peta
try:
    req = requests.get(route_url)
    res = req.json()
    if res['code'] == 'Ok':
        geometri_jalan = res['routes'][0]['geometry']
        folium.GeoJson(
            geometri_jalan,
            name='Rute Jalan Raya',
            style_function=lambda x: {'color': '#118AB2', 'weight': 5, 'opacity': 0.8}
        ).add_to(peta)
    else:
        folium.PolyLine(locations=rute_urut, color='green', weight=3).add_to(peta)
except Exception as e:
    folium.PolyLine(locations=rute_urut, color='green', weight=3).add_to(peta)

# 7. Simpan File Peta
nama_file_peta = 'Peta_Rute_Berurutan2.html'
lokasi_simpan = os.path.abspath(nama_file_peta)
peta.save(nama_file_peta)

print("\n" + "=" * 58)
print("✅ VISUALISASI PETA SELESAI!")
print("Silakan COPY lokasi file di bawah ini dan PASTE di browser (Chrome) Anda:")
print(f"👉  file:///{lokasi_simpan.replace(chr(92), '/')}")
print("=" * 58)