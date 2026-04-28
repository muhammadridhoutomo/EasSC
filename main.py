import pandas as pd
import numpy as np
import ast
import folium
import os
import requests

from ga_v1 import GeneticAlgorithm
from ga_v2 import TournamentGA
from ga_v3 import AdvancedMutationGA
from ga_v4 import AdaptiveGA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("==========================================================")
print("  MULTI-VERSION GA ROUTE PLANNING")
print("==========================================================")

# 1. Load Data
try:
    df_lokasi = pd.read_csv(os.path.join(BASE_DIR, 'wisata_sejarah.csv'))
    distance_matrix = np.load(os.path.join(BASE_DIR, 'matriks_wisata.npy'))
except FileNotFoundError as e:
    print(f"❌ ERROR: File data tidak ditemukan! ({e})")
    exit()

# 2. Daftar Algoritma
algos = [
    GeneticAlgorithm("Versi 1 (Klasik)", distance_matrix),
    TournamentGA("Versi 2 (Tournament Selection)", distance_matrix),
    AdvancedMutationGA("Versi 3 (Inversion Mut)", distance_matrix),
    AdaptiveGA("Versi 4 (Adaptive Rate)", distance_matrix)
]

results = {}
for algo in algos:
    route, dist = algo.run()
    results[algo.name] = (route, dist)
    print(f"   ✅ {algo.name} Selesai. Total Jarak: {dist:.2f} km\n")

# 3. Visualisasi Peta
print("⏳ Membuat Peta Interaktif dengan Rute Jalan Raya Presisi (Mohon tunggu beberapa saat)...")
peta = folium.Map(location=[-7.6, 112.5], zoom_start=9)

colors = ['#118AB2', '#EF476F', '#FFD166', '#06D6A0']
warna_kota = {
    'Surabaya': ('#d32f2f', '#ffcdd2'),  
    'Sidoarjo': ('#1976d2', '#bbdefb'),  
    'Mojokerto': ('#388e3c', '#c8e6c9'), 
    'Malang': ('#f57c00', '#ffe0b2')     
}

koordinat_list = []
for coord_str in df_lokasi['Coordinate']:
    c = ast.literal_eval(coord_str)
    koordinat_list.append((c['lat'], c['lng']))

for i, (name, (rute, dist)) in enumerate(results.items()):
    fg = folium.FeatureGroup(name=f"{name} ({dist:.2f} km)")
    
    rute_coords = [koordinat_list[idx] for idx in rute]
    rute_coords.append(rute_coords[0]) 
    
    # Array untuk menampung koordinat geometri utuh
    full_route_geometry = []
    
    # Minta rute OSRM secara titik-ke-titik agar presisi dan tidak ditolak server
    for j in range(len(rute_coords) - 1):
        lat1, lon1 = rute_coords[j]
        lat2, lon2 = rute_coords[j+1]
        
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?geometries=geojson&overview=full"
        
        try:
            res = requests.get(url).json()
            if res['code'] == 'Ok':
                coords = res['routes'][0]['geometry']['coordinates']
                # OSRM merespon dengan [lon, lat], Folium butuh [lat, lon]
                full_route_geometry.extend([[c[1], c[0]] for c in coords])
            else:
                full_route_geometry.extend([[lat1, lon1], [lat2, lon2]])
        except:
            full_route_geometry.extend([[lat1, lon1], [lat2, lon2]])
            
    # Gambar garis utuh yang sudah menempel di jalan raya
    folium.PolyLine(full_route_geometry, color=colors[i], weight=5, opacity=0.8).add_to(fg)

    # Tambahkan Marker dengan Nomor Urutan
    for urutan, indeks_tempat in enumerate(rute):
        lat, lng = koordinat_list[indeks_tempat]
        nama_tempat = df_lokasi.iloc[indeks_tempat]['Place_Name']
        kota_tempat = df_lokasi.iloc[indeks_tempat]['City']
        
        warna_kota_fill = warna_kota.get(kota_tempat, ('#616161', '#e0e0e0'))[1]
        
        icon_angka = folium.DivIcon(
            html=f'''
            <div style="
                font-size: 10pt; font-weight: bold; color: black; 
                background-color: {warna_kota_fill}; 
                border: 3px solid {colors[i]}; 
                border-radius: 50%; text-align: center; 
                line-height: 24px; width: 25px; height: 25px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
            ">
                {urutan + 1}
            </div>
            '''
        )
        
        folium.Marker(
            location=[lat, lng],
            popup=f"<b>{name}</b><br>Urutan ke-{urutan+1}: {nama_tempat}",
            icon=icon_angka
        ).add_to(fg)
        
    fg.add_to(peta)

folium.LayerControl(collapsed=False).add_to(peta)
peta.save("Perbandingan_Rute_GA.html")
print("=" * 58)
print("✨ SELESAI!.")