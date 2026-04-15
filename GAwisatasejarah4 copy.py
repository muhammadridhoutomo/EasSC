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
# BLOK 2: OPERATOR GA (VERSI 4)
# =====================================================================

# --- VERSI 4 (ORANG 4: Hibrida) ---

# 1. SELEKSI: Strong Tournament Selection
def seleksi(populasi, fitness_list):
    ukuran_turnamen = 3 
    idx_p1 = max(random.sample(range(len(populasi)), ukuran_turnamen), key=lambda i: fitness_list[i])
    idx_p2 = max(random.sample(range(len(populasi)), ukuran_turnamen), key=lambda i: fitness_list[i])
    return populasi[idx_p1], populasi[idx_p2]

# 2. CROSSOVER: Edge Recombination Crossover (ERX)
def crossover(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1.copy()

    size = len(parent1)
    
    # a. Buat Edge Map (Daftar tetangga dari setiap titik wisata)
    edge_map = {i: set() for i in range(jumlah_tempat)} # jumlah_tempat dari Blok 1
    for p in [parent1, parent2]:
        for i in range(size):
            node = p[i]
            kiri = p[i-1]
            kanan = p[(i+1) % size]
            edge_map[node].add(kiri)
            edge_map[node].add(kanan)
            
    # b. Bentuk rute anak
    child = []
    current_node = parent1[0] # Mulai dari node pertama parent 1
    
    while len(child) < size:
        child.append(current_node)
        
        # Hapus node ini dari semua daftar tetangga
        for node in edge_map:
            edge_map[node].discard(current_node)
            
        tetangga = edge_map[current_node]
        if not tetangga:
            # Jika buntu, ambil wisata acak yang belum dikunjungi
            belum_dikunjungi = [n for n in range(jumlah_tempat) if n not in child]
            if belum_dikunjungi:
                current_node = random.choice(belum_dikunjungi)
        else:
            # Pilih tetangga yang memiliki daftar tetangga paling sedikit (Heuristik)
            min_len = min(len(edge_map[n]) for n in tetangga)
            kandidat = [n for n in tetangga if len(edge_map[n]) == min_len]
            current_node = random.choice(kandidat)
            
    return child

# 3. MUTASI: Insert Mutation
def mutasi(individu, mutation_rate):
    if random.random() < mutation_rate:
        size = len(individu)
        # Ambil satu titik wisata
        idx_ambil = random.randint(0, size - 1)
        kota = individu.pop(idx_ambil)
        
        # Sisipkan ke tempat baru
        idx_sisip = random.randint(0, size - 1)
        individu.insert(idx_sisip, kota)
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
        # WAJIB TAMBAH .copy() AGAR REKOR TERBAIK TIDAK RUSAK
        best_route_overall = current_best_route.copy() 
        
    new_populasi = [best_route_overall] # Elitism
    
    while len(new_populasi) < ukuran_populasi:
        p1, p2 = seleksi(populasi, fitness_list)
        child = crossover(p1, p2, crossover_rate)
        
        # JURUS ANTI-STUCK: Paksa mutasi jika parent kloningan
        if p1 == p2:
            child = mutasi(child, 1.0) # Pasti mutasi 100%
            
            # Khusus Versi 4: Karena Insert Mutasi sangat lemah, 
            # kita lakukan Insert Mutasi 3 kali lipat agar rute benar-benar berubah!
            child = mutasi(child, 1.0) 
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
nama_file_peta = 'Peta_Rute_Berurutan4.html'
lokasi_simpan = os.path.abspath(nama_file_peta)
peta.save(nama_file_peta)

print("\n" + "=" * 58)
print("✅ VISUALISASI PETA SELESAI!")
print("Silakan COPY lokasi file di bawah ini dan PASTE di browser (Chrome) Anda:")
print(f"👉  file:///{lokasi_simpan.replace(chr(92), '/')}")
print("=" * 58)