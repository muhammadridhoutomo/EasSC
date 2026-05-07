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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("==========================================================")
print("  MULTI-VERSION GA - TRAVEL ITINERARY PLANNER")
print("==========================================================")

# 1. Load Data
try:
    df_lokasi = pd.read_csv(os.path.join(BASE_DIR, 'wisata_sejarah.csv'))
    distance_matrix = np.load(os.path.join(BASE_DIR, 'matriks_wisata.npy'))
except FileNotFoundError as e:
    print(f"❌ ERROR: File data tidak ditemukan! ({e})")
    exit()

# 2. Inisialisasi Algoritma
algos = [
    GeneticAlgorithm("Versi 1 (Klasik)", df_lokasi, distance_matrix),
    TournamentGA("Versi 2 (Tournament Selection)", df_lokasi, distance_matrix),
    AdvancedMutationGA("Versi 3 (Inversion Mut)", df_lokasi, distance_matrix),
    AdaptiveGA("Versi 4 (Adaptive Rate)", df_lokasi, distance_matrix)
]

results_data = {} 
results_meta = {}

for algo in algos:
    route, dist, itinerary, days = algo.run()
    results_data[algo.name] = itinerary
    results_meta[algo.name] = f"{dist:.2f} km"
    print(f"   ✅ {algo.name} | Jarak: {dist:.2f} km | Waktu: {days} Hari\n")

# ====================================================================
# 3. MEMBUAT GRAFIK KONVERGENSI (BARU)
# ====================================================================
print("⏳ Membuat Grafik Konvergensi...")
plt.figure(figsize=(11, 6)) # Lebarkan sedikit dari 10 ke 11 agar teks di kanan tidak terpotong

warna_grafik = ['#118AB2', '#EF476F', '#FFD166', '#06D6A0']

for i, algo in enumerate(algos):
    # Plot garisnya
    plt.plot(algo.history, label=f"{algo.name}", color=warna_grafik[i], linewidth=2)
    
    # Ambil nilai jarak terakhir (di generasi ke-1000)
    nilai_akhir = algo.history[-1]
    generasi_akhir = len(algo.history) - 1
    
    # Menambahkan teks angka persis di ujung kanan garis
    plt.text(generasi_akhir + 10, nilai_akhir, f'{nilai_akhir:.2f} km', 
             color=warna_grafik[i], fontsize=10, fontweight='bold', va='center')

plt.title('Grafik Konvergensi Kinerja Genetic Algorithm', fontsize=14, fontweight='bold')
plt.xlabel('Generasi', fontsize=12)
plt.ylabel('Total Jarak Tempuh (km)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Melebarkan batas sumbu X agar ada ruang kosong di sebelah kanan untuk teks angka
plt.xlim(0, 1150) 

plt.legend(loc='upper right')
plt.tight_layout()

# Simpan grafik sebagai gambar PNG
waktu_sekarang = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
nama_file_grafik = f"Grafik_Konvergensi_{waktu_sekarang}.png"
plt.savefig(nama_file_grafik, dpi=300)
print(f"✅ Grafik konvergensi berhasil disimpan sebagai '{nama_file_grafik}'\n")
# ====================================================================

# 3. Persiapan Peta
print("⏳ Membuat Peta Interaktif dengan Jadwal Dinamis...")
peta = folium.Map(location=[-7.6, 112.5], zoom_start=9)

colors = ['#118AB2', '#EF476F', '#FFD166', '#06D6A0']
warna_kota = {
    'Surabaya': ('#d32f2f', '#ffcdd2'),  
    'Sidoarjo': ('#1976d2', '#bbdefb'),  
    'Mojokerto': ('#388e3c', '#c8e6c9'), 
    'Malang': ('#f57c00', '#ffe0b2')     
}
# =====================================

koordinat_list = []
for coord_str in df_lokasi['Coordinate']:
    c = ast.literal_eval(coord_str)
    koordinat_list.append((c['lat'], c['lng']))

layer_mapping = {}

# 4. Generate Layer untuk setiap Versi
for i, (name, itinerary) in enumerate(results_data.items()):
    dist_str = results_meta[name]
    layer_name = f"{name} ({dist_str})"
    layer_mapping[layer_name] = name 
    
    fg = folium.FeatureGroup(name=layer_name)
    
    rute = []
    for item in itinerary:
        idx = df_lokasi[df_lokasi['Place_Name'] == item['place']].index[0]
        rute.append(idx)
    
    rute_coords = [koordinat_list[idx] for idx in rute]
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
    for urutan, indeks_tempat in enumerate(rute):
        lat, lng = koordinat_list[indeks_tempat]
        nama_tempat = df_lokasi.iloc[indeks_tempat]['Place_Name']
        kota_tempat = df_lokasi.iloc[indeks_tempat]['City']
        
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

# Pindahkan Layer Control ke Kiri Atas
folium.LayerControl(position='topleft', collapsed=False).add_to(peta)

# 5. Injeksi Panel HTML & JavaScript
itinerary_panel = """
<div id="itinerary-panel" style="position: fixed; top: 10px; right: 10px; width: 330px; height: 90vh; 
    background: white; z-index: 9999; overflow-y: auto; padding: 15px; 
    border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    <h3 style="margin-top:0; color:#333; text-align:center;">Jadwal Perjalanan</h3>
    <div id="itinerary-content">
        <p style="text-align:center; color:#888; margin-top:50px;">Pilih salah satu versi GA di sebelah kiri untuk melihat detail jadwal.</p>
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

# 6. Simpan File
waktu_sekarang = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
nama_file = f"Rute_Wisata_GA_{waktu_sekarang}.html"
peta.save(nama_file)

print("=" * 58)
print(f"✨ BERHASIL! File disimpan sebagai '{nama_file}'.")