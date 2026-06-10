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
from discrete_pso import PSOAlgorithm
from tabu_search import TabuSearch

# Set Page Config
st.set_page_config(page_title="Travel Itinerary Planner PRO", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("🗺️ Custom Travel Planner (Multi-Algorithm Comparison)")
st.markdown("Bandingkan rute terbaik antara algoritma **Particle Swarm Optimization** dan **Tabu Search** secara bersamaan.")

# --- LOGIKA RESET ---
def reset_results():
    st.session_state.results = None

# --- SIDEBAR ---
st.sidebar.header("⚙️ Pengaturan Wisatawan")
start_city = st.sidebar.selectbox(
    "Kota Keberangkatan (Start)",
    ["Surabaya", "Sidoarjo", "Mojokerto", "Malang", "Batu"],
    on_change=reset_results
)
max_days = st.sidebar.slider(
    "Durasi Liburan (Hari)", 1, 7, 3,
    on_change=reset_results
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
    dur_matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_durasi_new.npy'))
    return df_new, matrix_new, dur_matrix_new

df_new, matrix_new, dur_matrix_new = load_data()

# --- LOGIC: RUN ALGORITHMS ---
if run_button:
    with st.spinner("Mencari rute paling optimal (500 Iterasi)..."):
        # Jalankan PSO
        pso = PSOAlgorithm("PSO", df_new, matrix_new, dur_matrix_new,
                          start_city=start_city, max_days=max_days, generations=500)
        _, pso_dist, pso_itin, pso_days = pso.run()
        
        # Jalankan Tabu Search
        tabu = TabuSearch("Tabu Search", df_new, matrix_new, dur_matrix_new,
                         start_city=start_city, max_days=max_days, iterations=500)
        _, tabu_dist, tabu_itin, tabu_days = tabu.run()

        st.session_state.results = {
            "PSO": {"itinerary": pso_itin, "distance": pso_dist, "history": pso.history, "color": "#EF476F"},
            "Tabu Search": {"itinerary": tabu_itin, "distance": tabu_dist, "history": tabu.history, "color": "#118AB2"},
            "meta": {"city": start_city, "days": max_days}
        }

# --- DISPLAY RESULTS ---
if st.session_state.results:
    results_map = st.session_state.results
    meta = results_map["meta"]

    st.success(f"Berhasil mengoptimasi rute dari **{meta['city']}** ({meta['days']} hari).")

    # 1. GRAFIK KONVERGENSI
    st.subheader("📈 Progress Optimasi")
    fig, ax = plt.subplots(figsize=(12, 3))
    for name in ["PSO", "Tabu Search"]:
        ax.plot(results_map[name]["history"], label=name, color=results_map[name]["color"], linewidth=2)
    ax.set_ylabel('Jumlah Wisata')
    ax.legend()
    st.pyplot(fig)

    # 2. PEMILIHAN ALGORITMA UNTUK PETA
    st.divider()
    selected_algos = st.multiselect(
        "🔍 **Pilih Algoritma yang ingin ditampilkan di Peta:**", 
        ["PSO", "Tabu Search"], 
        default=["PSO", "Tabu Search"]
    )

    if not selected_algos:
        st.warning("Pilih minimal satu algoritma untuk melihat rute di peta.")
    else:
        # 3. VISUALISASI PETA (MULTI-LAYER)
        peta = folium.Map(location=[-7.6, 112.5], zoom_start=9)
        warna_kota = {
            'Surabaya': ('#d32f2f', '#ffcdd2'),  'Sidoarjo': ('#1976d2', '#bbdefb'),  
            'Mojokerto': ('#388e3c', '#c8e6c9'), 'Malang': ('#f57c00', '#ffe0b2'),
            'Batu': ('#7b1fa2', '#e1bee7')     
        }

        layer_itineraries = {}

        for name in selected_algos:
            data_view = results_map[name]
            itin = data_view["itinerary"]
            layer_name = f"{name} ({len([x for x in itin if not x.get('is_mobilisasi')])} Wisata)"
            layer_itineraries[layer_name] = itin
            
            fg = folium.FeatureGroup(name=layer_name)
            
            # Ambil koordinat wisata
            coords = []
            for item in itin:
                if not item.get('is_mobilisasi'):
                    match = df_new[df_new['Nama Tempat'] == item['place']].iloc[0]
                    coords.append((float(match['Latitude']), float(match['Longitude'])))
            
            if coords:
                coords_loop = coords + [coords[0]]
                full_geom = []
                # Ambil geometri jalan (langsung tanpa status bar agar UI bersih saat ganti multiselect)
                for j in range(len(coords_loop) - 1):
                    c1, c2 = coords_loop[j], coords_loop[j+1]
                    url = f"http://router.project-osrm.org/route/v1/driving/{c1[1]},{c1[0]};{c2[1]},{c2[0]}?geometries=geojson&overview=full"
                    try:
                        res = requests.get(url, timeout=5).json()
                        if res['code'] == 'Ok':
                            full_geom.extend([[p[1], p[0]] for p in res['routes'][0]['geometry']['coordinates']])
                        else: full_geom.extend([[c1[0], c1[1]], [c2[0], c2[1]]])
                    except: full_geom.extend([[c1[0], c1[1]], [c2[0], c2[1]]])
                
                folium.PolyLine(full_geom, color=data_view["color"], weight=5, opacity=0.8).add_to(fg)

                # Marker Wisata
                wisata_idx = 1
                for item in itin:
                    if not item.get('is_mobilisasi'):
                        match = df_new[df_new['Nama Tempat'] == item['place']].iloc[0]
                        c = (float(match['Latitude']), float(match['Longitude']))
                        fill = warna_kota.get(item['city'], ('#616161', '#e0e0e0'))[1]
                        icon_html = f'''<div style="font-size:8pt; font-weight:bold; background:{fill}; border:2px solid {data_view['color']}; border-radius:50%; text-align:center; line-height:18px; width:20px; height:20px;">{wisata_idx}</div>'''
                        folium.Marker(location=c, popup=f"{name}: {item['place']}", icon=folium.DivIcon(html=icon_html)).add_to(fg)
                        wisata_idx += 1
            
            fg.add_to(peta)

        folium.LayerControl(position='topleft', collapsed=False).add_to(peta)

        # Injeksi Panel Jadwal (Advanced Interactive)
        data_json = json.dumps(layer_itineraries)
        itinerary_panel = """
        <div id="itinerary-panel" style="position: fixed; top: 10px; right: 10px; width: 280px; height: 500px; 
            background: white; z-index: 9999; overflow-y: auto; padding: 15px; 
            border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.2); font-family: sans-serif;">
            <h3 style="margin-top:0; font-size:16px; text-align:center; color:#073B4C;">Detail Perjalanan</h3>
            <div id="itin-content" style="font-size:12px; color:#666; text-align:center; margin-top:50px;">Pilih layer algoritma di pojok kiri peta untuk melihat jadwal.</div>
        </div>
        """
        peta.get_root().html.add_child(folium.Element(itinerary_panel))
        
        script_js = f"""
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var dataItin = {data_json};
            var contentDiv = document.getElementById('itin-content');
            
            window.updateItin = function(name) {{
                var items = dataItin[name];
                if(!items) return;
                var html = '<h4 style="text-align:center; color:#073B4C; margin-bottom:5px;">' + name + '</h4><hr>';
                var curDay = 0;
                items.forEach(function(item) {{
                    if (item.day !== curDay) {{
                        curDay = item.day;
                        html += '<div style="background:#073B4C; color:white; padding:3px 8px; border-radius:4px; margin-top:10px; font-weight:bold;">Hari ' + curDay + '</div>';
                    }}
                    var isMob = item.place.includes('Mobilisasi');
                    var color = isMob ? '#888' : '#EF476F';
                    var icon = isMob ? '' : '📍 ';
                    html += '<div style="margin: 8px 0; border-left: 3px solid ' + color + '; padding-left: 8px;">';
                    html += '<b>' + item.arrive + ' - ' + item.depart + '</b><br>' + icon + item.place + '</div>';
                }});
                contentDiv.innerHTML = html;
                contentDiv.style.textAlign = 'left';
                contentDiv.style.marginTop = '0';
            }};

            var map_inst = null;
            for (var key in window) {{ if (key.startsWith('map_')) {{ map_inst = window[key]; break; }} }}
            if (map_inst) {{
                map_inst.on('overlayadd', function(e) {{
                    window.updateItin(e.name);
                }});
            }}
        }});
        </script>
        """
        peta.get_root().html.add_child(folium.Element(script_js))
        components.html(peta._repr_html_(), height=600)

else:
    st.info("💡 Atur rencana perjalanan di sidebar, lalu klik 'Generate Perbandingan Rute'.")
