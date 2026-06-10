import streamlit as st
import pandas as pd
import numpy as np
import folium
import streamlit.components.v1 as components
import os
import ast
import json
import datetime
import matplotlib.pyplot as plt
from ga_v1 import GeneticAlgorithm
from ga_v2 import TournamentGA
from ga_v3 import AdvancedMutationGA
from ga_v4 import AdaptiveGA
from pso import PSOAlgorithm

# Set Page Config
st.set_page_config(page_title="Travel Itinerary Planner PRO", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("🗺️ Custom Travel Planner (Particle Swarm Optimization)")
st.markdown("Optimasi rute wisata sejarah berbasis kustomisasi jumlah hari dan kota keberangkatan.")

# --- SIDEBAR (Customization) ---
st.sidebar.header("⚙️ Pengaturan Wisatawan")
start_city = st.sidebar.selectbox(
    "Kota Keberangkatan (Start)",
    ["Surabaya", "Sidoarjo", "Mojokerto", "Malang", "Batu"]
)
max_days = st.sidebar.slider("Durasi Liburan (Hari)", 1, 7, 3)

st.sidebar.divider()
st.sidebar.info("Klik tombol di bawah untuk menghitung rute terbaik menggunakan algoritma PSO.")
run_button = st.sidebar.button("🚀 Generate Rute Terbaik", use_container_width=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df_old = pd.read_csv(os.path.join(BASE_DIR, 'wisata_sejarah.csv'))
    matrix_old = np.load(os.path.join(BASE_DIR, 'matriks_wisata.npy'))
    df_new = pd.read_csv(os.path.join(BASE_DIR, 'historical_new.csv'))
    matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_wisata_new.npy'))
    return df_old, matrix_old, df_new, matrix_new

df_old, matrix_old, df_new, matrix_new = load_data()

if run_button:
    with st.spinner("Algoritma PSO sedang bekerja mencari rute paling efisien..."):
        # 1. RUN ALGORITHMS (Hanya PSO)
        algos = [
            PSOAlgorithm("Custom PSO Optimization", df_new, matrix_new, start_city=start_city, max_days=max_days)
        ]
        
        results_data = {}
        results_meta = {}
        results_df = {}
        history_data = []

        for algo in algos:
            route, dist, itinerary, days = algo.run()
            results_data[algo.name] = itinerary
            results_meta[algo.name] = f"{dist:.2f} km"
            results_df[algo.name] = algo.df
            history_data.append({"name": algo.name, "history": algo.history})

        # --- 2. GRAFIK KONVERGENSI ---
        st.subheader("📈 Grafik Optimasi PSO")
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        warna_grafik = ['#073B4C']
        
        for i, h in enumerate(history_data):
            ax1.plot(h['history'], label=f"Progress: {h['name']}", color=warna_grafik[i], linewidth=2.5)
            ax1.text(len(h['history'])-1, h['history'][-1], f" {h['history'][-1]} Wisata", color=warna_grafik[i], fontweight='bold', fontsize=12)
        
        ax1.set_xlabel('Generasi (Iterasi)')
        ax1.set_ylabel('Jumlah Wisata Terkunjungi')
        ax1.set_title('Peningkatan Jumlah Lokasi Wisata yang Dapat Dikunjungi')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='lower right')
        st.pyplot(fig)

        # --- 3. PETA INTERAKTIF ---
        st.divider()
        st.subheader("🗺️ Visualisasi Peta & Jalwal Perjalanan (Real Road Distance)")
        
        status_road = st.status("Mengambil data jalan asli dari OSRM...")
        
        peta = folium.Map(location=[-7.6, 112.5], zoom_start=9)
        colors = ['#EF476F'] 
        warna_kota = {
            'Surabaya': ('#d32f2f', '#ffcdd2'),  'Sidoarjo': ('#1976d2', '#bbdefb'),  
            'Mojokerto': ('#388e3c', '#c8e6c9'), 'Malang': ('#f57c00', '#ffe0b2'),
            'Batu': ('#7b1fa2', '#e1bee7')     
        }
        
        layer_mapping = {}

        for i, (name, itinerary) in enumerate(results_data.items()):
            dist_str = results_meta[name]
            layer_name = f"Rute PSO ({len(itinerary)} Wisata - {dist_str})"
            layer_mapping[layer_name] = name 
            fg = folium.FeatureGroup(name=layer_name)
            
            current_df = results_df[name]
            name_col = 'Nama Tempat'
            
            rute_coords = []
            for item in itinerary:
                idx = current_df[current_df[name_col] == item['place']].index[0]
                rute_coords.append((current_df.iloc[idx]['Latitude'], current_df.iloc[idx]['Longitude']))
            
            if not rute_coords: continue
            rute_coords.append(rute_coords[0]) # Kembali ke titik awal

            # --- Gambar Garis Presisi (OSRM) ---
            full_route_geometry = []
            for j in range(len(rute_coords) - 1):
                lat1, lon1 = rute_coords[j]
                lat2, lon2 = rute_coords[j+1]
                url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?geometries=geojson&overview=full"
                try:
                    res_osrm = requests.get(url, timeout=5).json()
                    if res_osrm['code'] == 'Ok':
                        coords = res_osrm['routes'][0]['geometry']['coordinates']
                        full_route_geometry.extend([[c[1], c[0]] for c in coords])
                    else: 
                        full_route_geometry.extend([[lat1, lon1], [lat2, lon2]])
                except: 
                    full_route_geometry.extend([[lat1, lon1], [lat2, lon2]])
            
            folium.PolyLine(full_route_geometry, color=colors[i], weight=5, opacity=0.8).add_to(fg)
            status_road.update(label="✅ Data jalan asli berhasil dimuat!", state="complete")

            # Marker angka
            for urutan in range(len(itinerary)):
                coord = rute_coords[urutan]
                nama_tempat = itinerary[urutan]['place']
                kota_tempat = itinerary[urutan]['city']
                fill_color = warna_kota.get(kota_tempat, ('#616161', '#e0e0e0'))[1]
                
                icon_angka = folium.DivIcon(html=f'''
                    <div style="font-size: 9pt; font-weight: bold; color: black; background-color: {fill_color}; 
                    border: 2px solid {colors[i]}; border-radius: 50%; text-align: center; line-height: 20px; 
                    width: 22px; height: 22px; box-shadow: 1px 1px 3px rgba(0,0,0,0.3);">{urutan+1}</div>''')
                
                folium.Marker(location=coord, popup=f"<b>{name}</b><br>{urutan+1}: {nama_tempat}", icon=icon_angka).add_to(fg)
            
            fg.add_to(peta)

        folium.LayerControl(position='topleft', collapsed=False).add_to(peta)

        # Injeksi Panel Jadwal
        itinerary_panel = """
        <div id="itinerary-panel" style="position: fixed; top: 10px; right: 10px; width: 300px; height: 550px; 
            background: white; z-index: 9999; overflow-y: auto; padding: 15px; 
            border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.2); font-family: sans-serif; font-size: 13px;">
            <h3 style="margin-top:0; font-size:16px; text-align:center;">Jadwal Perjalanan</h3>
            <div id="itinerary-content"><p style="color:#888; text-align:center;">Jadwal akan muncul otomatis di sini.</p></div>
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
            
            // Auto-render the first result
            var firstKey = Object.keys(dataItin)[0];
            if (firstKey) {{
                var items = dataItin[firstKey];
                var html = '<h4 style="color:#EF476F; margin:5px 0;">Rute Optimal PSO</h4><hr>';
                var curDay = 0;
                items.forEach(function(item) {{
                    if (item.day !== curDay) {{
                        curDay = item.day;
                        html += '<div style="background:#EF476F; color:white; padding:3px 8px; border-radius:4px; margin-top:10px;">Hari ' + curDay + '</div>';
                    }}
                    html += '<div style="margin: 8px 0; border-left: 3px solid #073B4C; padding-left: 8px;">';
                    html += '<b>' + item.arrive + ' - ' + item.depart + '</b><br>' + item.place + '</div>';
                }});
                contentDiv.innerHTML = html;
            }}
        }});
        </script>
        """
        peta.get_root().html.add_child(folium.Element(script_js))

        components.html(peta._repr_html_(), height=600)

else:
    st.info("💡 Selamat datang! Atur kota keberangkatan dan durasi liburan Anda di sidebar kiri, lalu klik 'Generate Rute Terbaik'.")
