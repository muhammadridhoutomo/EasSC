import pandas as pd
import numpy as np
import ast
import folium
import os
import requests
import datetime
import json
import random
import matplotlib.pyplot as plt

from ga_v1 import GeneticAlgorithm
from ga_v2 import TournamentGA
from ga_v3 import AdvancedMutationGA
from ga_v4 import AdaptiveGA
from discrete_pso import PSOAlgorithm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("==========================================================")
print("  MULTI-VERSION GA & PSO - TRAVEL ITINERARY PLANNER")
print("==========================================================")

# Input User untuk PSO
print("\n--- Kustomisasi Perjalanan (Khusus PSO) ---")
start_city_user = input("Masukkan Kota Keberangkatan (Surabaya/Sidoarjo/Mojokerto/Malang/Batu): ") or "Surabaya"
max_days_user = int(input("Berapa hari rencana perjalanan Anda? ") or "3")

# 1. Load Data
try:
    # Dataset Lama
    df_old = pd.read_csv(os.path.join(BASE_DIR, 'wisata_sejarah.csv'))
    matrix_old = np.load(os.path.join(BASE_DIR, 'matriks_wisata.npy'))
    
    # Dataset Baru
    df_new = pd.read_csv(os.path.join(BASE_DIR, 'historical_new.csv'))
    matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_wisata_new.npy'))
except FileNotFoundError as e:
    print(f"❌ ERROR: File data tidak ditemukan! ({e})")
    exit()

# 2. Inisialisasi Algoritma
algos = [
    GeneticAlgorithm("Versi 1 (Klasik)", df_old, matrix_old),
    TournamentGA("Versi 2 (Tournament Selection)", df_old, matrix_old),
    AdvancedMutationGA("Versi 3 (Inversion Mut)", df_old, matrix_old),
    AdaptiveGA("Versi 4 (Adaptive Rate)", df_old, matrix_old),
    PSOAlgorithm("Versi 5 (Custom PSO)", df_new, matrix_new, start_city=start_city_user, max_days=max_days_user) 
]

results_data = {} 
results_meta = {}
results_df = {} # Simpan reference DF untuk plotting peta nanti

for algo in algos:
    route, dist, itinerary, days = algo.run()
    results_data[algo.name] = itinerary
    results_meta[algo.name] = f"{dist:.2f} km"
    results_df[algo.name] = algo.df
    print(f"   ✅ {algo.name} | Jarak: {dist:.2f} km | Waktu: {days} Hari\n")

# ====================================================================
# 3. MEMBUAT GRAFIK KONVERGENSI (BARU)
# ====================================================================
print("⏳ Membuat Grafik Konvergensi...")
fig, ax1 = plt.subplots(figsize=(12, 7))

warna_grafik = ['#118AB2', '#EF476F', '#FFD166', '#06D6A0', '#073B4C']

# Sumbu Y untuk GA (Jarak)
ax1.set_xlabel('Generasi')
ax1.set_ylabel('Total Jarak (km) - GA Versions', color='black')

# Sumbu Y untuk PSO (Jumlah Wisata)
ax2 = ax1.twinx()
ax2.set_ylabel('Jumlah Wisata Terkunjungi - Custom PSO', color='#073B4C')

for i, algo in enumerate(algos):
    if "PSO" in algo.name:
        # Plot PSO di sumbu Y kedua
        ax2.plot(algo.history, label=f"{algo.name} (Wisata)", color=warna_grafik[i], linewidth=2, linestyle='--')
        nilai_akhir = algo.history[-1]
        ax2.text(len(algo.history) - 1 + 10, nilai_akhir, f'{nilai_akhir} Wisata', 
                 color=warna_grafik[i], fontsize=10, fontweight='bold', va='center')
    else:
        # Plot GA di sumbu Y pertama
        ax1.plot(algo.history, label=f"{algo.name} (Jarak)", color=warna_grafik[i], linewidth=2)
        nilai_akhir = algo.history[-1]
        ax1.text(len(algo.history) - 1 + 10, nilai_akhir, f'{nilai_akhir:.2f} km', 
                 color=warna_grafik[i], fontsize=10, fontweight='bold', va='center')

plt.title('Grafik Konvergensi Kinerja Algoritma (GA vs Custom PSO)', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 1150) 
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()

# Simpan grafik sebagai gambar PNG
waktu_sekarang = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
nama_file_grafik = f"Grafik_Konvergensi_{waktu_sekarang}.png"
plt.savefig(nama_file_grafik, dpi=300)
print(f"✅ Grafik konvergensi berhasil disimpan sebagai '{nama_file_grafik}'\n")
# ====================================================================

# 4. Persiapan Peta
print("⏳ Membuat Peta Interaktif dengan Jadwal Dinamis...")
peta = folium.Map(location=[-7.6, 112.5], zoom_start=9)

colors = ['#118AB2', '#EF476F', '#FFD166', '#06D6A0', '#073B4C']
warna_kota = {
    'Surabaya': ('#d32f2f', '#ffcdd2'),  
    'Sidoarjo': ('#1976d2', '#bbdefb'),  
    'Mojokerto': ('#388e3c', '#c8e6c9'), 
    'Malang': ('#f57c00', '#ffe0b2'),
    'Batu': ('#7b1fa2', '#e1bee7')     
}

layer_mapping = {}

# 5. Generate Layer untuk setiap Versi
for i, (name, itinerary) in enumerate(results_data.items()):
    dist_str = results_meta[name]
    layer_name = f"{name} ({dist_str})"
    layer_mapping[layer_name] = name 
    
    fg = folium.FeatureGroup(name=layer_name)
    
    current_df = results_df[name]
    name_col = 'Place_Name' if 'Place_Name' in current_df.columns else 'Nama Tempat'
    city_col = 'City' if 'City' in current_df.columns else 'Kota'
    
    rute = []
    rute_coords = []
    for item in itinerary:
        idx = current_df[current_df[name_col] == item['place']].index[0]
        rute.append(idx)
        
        if 'Coordinate' in current_df.columns:
            c = ast.literal_eval(current_df.iloc[idx]['Coordinate'])
            rute_coords.append((c['lat'], c['lng']))
        else:
            rute_coords.append((current_df.iloc[idx]['Latitude'], current_df.iloc[idx]['Longitude']))
    
    if not rute_coords: continue
    rute_coords.append(rute_coords[0]) 

    # --- Gambar Garis Presisi (OSRM) ---
    full_route_geometry = []
    for j in range(len(rute_coords) - 1):
        lat1, lon1 = rute_coords[j]
        lat2, lon2 = rute_coords[j+1]
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?geometries=geojson&overview=full"
        try:
            res = requests.get(url).json()
            if res['code'] == 'Ok':
                coords = res['routes'][0]['geometry']['coordinates']
                full_route_geometry.extend([[c[1], c[0]] for c in coords])
            else: full_route_geometry.extend([[lat1, lon1], [lat2, lon2]])
        except: full_route_geometry.extend([[lat1, lon1], [lat2, lon2]])
            
    folium.PolyLine(full_route_geometry, color=colors[i], weight=5, opacity=0.8).add_to(fg)

    # --- Tambahkan Marker Angka Urutan ---
    for urutan, idx_tempat in enumerate(rute):
        lat, lng = rute_coords[urutan]
        nama_tempat = current_df.iloc[idx_tempat][name_col]
        kota_tempat = current_df.iloc[idx_tempat][city_col]
        
        warna_kota_fill = warna_kota.get(kota_tempat, ('#616161', '#e0e0e0'))[1]
        
        icon_angka = folium.DivIcon(
            html=f'''
            <div style="font-size: 10pt; font-weight: bold; color: black; 
                background-color: {warna_kota_fill}; border: 3px solid {colors[i]}; 
                border-radius: 50%; text-align: center; line-height: 24px; 
                width: 25px; height: 25px; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);">
                {urutan + 1}
            </div>'''
        )
        folium.Marker(
            location=[lat, lng], 
            popup=f"<b>{name}</b><br>Urutan ke-{urutan+1}: {nama_tempat}", 
            icon=icon_angka
        ).add_to(fg)
        
    fg.add_to(peta)

folium.LayerControl(position='topleft', collapsed=False).add_to(peta)

# 6. Injeksi Panel HTML & JavaScript
itinerary_panel = """
<div id="itinerary-panel" style="position: fixed; top: 10px; right: 10px; width: 330px; height: 90vh; 
    background: white; z-index: 9999; overflow-y: auto; padding: 15px; 
    border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    <h3 style="margin-top:0; color:#333; text-align:center;">Jadwal Perjalanan</h3>
    <div id="itinerary-content">
        <p style="text-align:center; color:#888; margin-top:50px;">Pilih salah satu versi GA/PSO di sebelah kiri untuk melihat detail jadwal.</p>
    </div>
</div>
"""
peta.get_root().html.add_child(folium.Element(itinerary_panel))

data_json = json.dumps(results_data)
mapping_json = json.dumps(layer_mapping)

script_js = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
    var dataItin = {data_json};
    var nameMapping = {mapping_json};
    var contentDiv = document.getElementById('itinerary-content');

    function renderItinerary(versionName) {{
        var items = dataItin[versionName];
        var html = '<h4 style="text-align:center; color:#118AB2; margin-bottom:5px;">' + versionName + '</h4><hr>';
        var currentDay = 0;

        items.forEach(function(item) {{
            if (item.day !== currentDay) {{
                currentDay = item.day;
                html += '<h5 style="background:#118AB2; color:white; padding:5px; border-radius:5px; margin-top:15px;">Hari ' + currentDay + ' - ' + item.city + '</h5>';
            }}
            html += '<div style="margin-bottom: 8px; font-size: 12px; border-left: 3px solid #EF476F; padding-left: 8px;">';
            html += '<b>' + item.arrive + ' - ' + item.depart + '</b><br>';
            html += '<span style="color:#222;">' + item.place + '</span><br>';
            html += '<span style="color:#777; font-size:10px;">Durasi: ' + item.duration + ' mnt</span>';
            html += '</div>';
        }});
        contentDiv.innerHTML = html;
    }}

    var map_instance = null;
    for (var key in window) {{
        if (key.startsWith('map_') && window[key] instanceof L.Map) {{
            map_instance = window[key];
            break;
        }}
    }}

    if (map_instance) {{
        map_instance.on('overlayadd', function(e) {{
            var realVersionName = nameMapping[e.name];
            if (realVersionName) {{
                renderItinerary(realVersionName);
            }}
        }});
    }}
}});
</script>
"""
peta.get_root().html.add_child(folium.Element(script_js))

# 7. Simpan File
waktu_sekarang = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
nama_file = f"Rute_Wisata_GA_PSO_{waktu_sekarang}.html"
peta.save(nama_file)

print("=" * 58)
print(f"✨ BERHASIL! File disimpan sebagai '{nama_file}'.")
