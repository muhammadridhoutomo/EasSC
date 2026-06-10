import streamlit as st
import pandas as pd
import numpy as np
import folium
import streamlit.components.v1 as components
import os
import ast
import json
import requests
import matplotlib.pyplot as plt
from pso import PSOAlgorithm
from tabu_search import TabuSearch

# Set Page Config
st.set_page_config(page_title="Travel Itinerary Planner PRO", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("🗺️ Custom Travel Planner (PSO & Tabu Search)")
st.markdown("Bandingkan rute terbaik antara algoritma **Particle Swarm Optimization** dan **Tabu Search**.")

# --- LOGIKA RESET: Jika input sidebar berubah, hapus hasil lama ---
def reset_results():
    st.session_state.results = None

# --- SIDEBAR ---
st.sidebar.header("⚙️ Pengaturan Wisatawan")
start_city = st.sidebar.selectbox(
    "Kota Keberangkatan (Start)",
    ["Surabaya", "Sidoarjo", "Mojokerto", "Malang", "Batu"],
    on_change=reset_results # Reset hasil jika kota diganti
)
max_days = st.sidebar.slider(
    "Durasi Liburan (Hari)", 1, 7, 3,
    on_change=reset_results # Reset hasil jika hari diganti
)

st.sidebar.divider()
run_button = st.sidebar.button("🚀 Generate Perbandingan Rute", use_container_width=True)

# --- INITIALIZE SESSION STATE ---
if 'results' not in st.session_state:
    st.session_state.results = None

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df_new = pd.read_csv(os.path.join(BASE_DIR, 'historical_new.csv'))
    matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_wisata_new.npy'))
    return df_new, matrix_new

df_new, matrix_new = load_data()

# --- LOGIC: RUN ALGORITHMS ---
if run_button:
    with st.spinner("Mencari rute paling optimal (500 Iterasi)..."):
        # Jalankan PSO (Generasi dibatasi 500)
        pso = PSOAlgorithm("Particle Swarm Optimization", df_new, matrix_new, 
                          start_city=start_city, max_days=max_days, generations=500)
        _, pso_dist, pso_itin, pso_days = pso.run()
        
        # Jalankan Tabu Search (Iterasi dibatasi 500)
        tabu = TabuSearch("Tabu Search", df_new, matrix_new, 
                         start_city=start_city, max_days=max_days, iterations=500)
        _, tabu_dist, tabu_itin, tabu_days = tabu.run()

        # Simpan ke Session State beserta info input yang digunakan
        st.session_state.results = {
            "PSO": {"itinerary": pso_itin, "distance": pso_dist, "history": pso.history, "color": "#EF476F"},
            "Tabu Search": {"itinerary": tabu_itin, "distance": tabu_dist, "history": tabu.history, "color": "#118AB2"},
            "meta": {"city": start_city, "days": max_days}
        }

# --- DISPLAY RESULTS (IF AVAILABLE) ---
if st.session_state.results:
    results_map = st.session_state.results
    meta = results_map["meta"]

    # Notifikasi info rute saat ini
    st.success(f"Menampilkan rute dari **{meta['city']}** untuk durasi **{meta['days']} hari**.")

    # 1. GRAFIK KONVERGENSI
    st.subheader("📈 Perbandingan Progress Optimasi (500 Iterasi)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(results_map["PSO"]["history"], label="PSO", color=results_map["PSO"]["color"], linewidth=2)
    ax.plot(results_map["Tabu Search"]["history"], label="Tabu Search", color=results_map["Tabu Search"]["color"], linewidth=2)
    ax.set_xlabel('Iterasi')
    ax.set_ylabel('Jumlah Wisata')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # 2. PEMILIHAN ALGORITMA UNTUK PETA
    st.divider()
    selected_algo = st.radio("🔍 **Pilih Algoritma untuk Visualisasi Peta:**", ["PSO", "Tabu Search"], horizontal=True)
    
    data_view = results_map[selected_algo]
    itinerary = data_view["itinerary"]

    st.subheader(f"🗺️ Visualisasi Peta: {selected_algo} ({len(itinerary)} Wisata)")
    
    # 3. VISUALISASI PETA (REAL ROAD)
    peta = folium.Map(location=[-7.6, 112.5], zoom_start=9)
    warna_kota = {
        'Surabaya': ('#d32f2f', '#ffcdd2'),  'Sidoarjo': ('#1976d2', '#bbdefb'),  
        'Mojokerto': ('#388e3c', '#c8e6c9'), 'Malang': ('#f57c00', '#ffe0b2'),
        'Batu': ('#7b1fa2', '#e1bee7')     
    }

    rute_coords = []
    for item in itinerary:
        match = df_new[df_new['Nama Tempat'] == item['place']].iloc[0]
        rute_coords.append((float(match['Latitude']), float(match['Longitude'])))
    
    if rute_coords:
        rute_coords_loop = rute_coords + [rute_coords[0]]
        full_geometry = []
        
        with st.status(f"Mengambil geometri jalan raya {selected_algo}...", expanded=False) as status:
            for j in range(len(rute_coords_loop) - 1):
                lat1, lon1 = rute_coords_loop[j]
                lat2, lon2 = rute_coords_loop[j+1]
                url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?geometries=geojson&overview=full"
                try:
                    res = requests.get(url, timeout=10).json()
                    if res['code'] == 'Ok':
                        segment_coords = [[p[1], p[0]] for p in res['routes'][0]['geometry']['coordinates']]
                        full_geometry.extend(segment_coords)
                    else: full_geometry.extend([[lat1, lon1], [lat2, lon2]])
                except: full_geometry.extend([[lat1, lon1], [lat2, lon2]])
            status.update(label="✅ Geometri dimuat!", state="complete", expanded=False)
        
        folium.PolyLine(full_geometry, color=data_view["color"], weight=5, opacity=0.85).add_to(peta)

        for i in range(len(itinerary)):
            coord = rute_coords[i]
            item = itinerary[i]
            fill_color = warna_kota.get(item['city'], ('#616161', '#e0e0e0'))[1]
            icon_html = f'''
                <div style="font-size: 9pt; font-weight: bold; color: black; background-color: {fill_color}; 
                border: 2px solid {data_view['color']}; border-radius: 50%; text-align: center; line-height: 20px; 
                width: 22px; height: 22px; box-shadow: 1px 1px 3px rgba(0,0,0,0.3);">{i+1}</div>'''
            folium.Marker(location=coord, popup=f"<b>{item['place']}</b>", icon=folium.DivIcon(html=icon_html)).add_to(peta)

    # Injeksi Panel Jadwal
    results_json = json.dumps({"active": itinerary})
    itinerary_panel = f"""
    <div id="itinerary-panel" style="position: fixed; top: 10px; right: 10px; width: 280px; height: 500px; 
        background: white; z-index: 9999; overflow-y: auto; padding: 15px; 
        border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.2); font-family: sans-serif;">
        <h3 style="margin-top:0; font-size:16px; text-align:center; color:#073B4C;">Jadwal {selected_algo}</h3>
        <div id="itin-content"></div>
    </div>
    <script>
        var data = {results_json};
        var content = document.getElementById('itin-content');
        var items = data["active"];
        var html = '';
        var curDay = 0;
        items.forEach(function(item) {{
            if (item.day !== curDay) {{
                curDay = item.day;
                html += '<div style="background:{data_view['color']}; color:white; padding:4px 8px; border-radius:4px; margin-top:10px; font-weight:bold;">Hari ' + curDay + '</div>';
            }}
            html += '<div style="margin: 8px 0; border-left: 3px solid #073B4C; padding-left: 8px; font-size:12px;">';
            html += '<b>' + item.arrive + ' - ' + item.depart + '</b><br>' + item.place + '</div>';
        }});
        content.innerHTML = html;
    </script>
    """
    peta.get_root().html.add_child(folium.Element(itinerary_panel))
    components.html(peta._repr_html_(), height=600)

else:
    st.info("💡 Atur rencana perjalanan di sidebar, lalu klik 'Generate Perbandingan Rute'.")
    st.image("https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&q=80&w=1200")
