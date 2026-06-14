import pandas as pd
import numpy as np
import time
import os

# Import algorithms
from discrete_pso import PSOAlgorithm
from tabu_search import TabuSearch
from hybrid_GA import HybridGA
from ACO import ACOAlgorithm

def run_individual_algo():
    # --- 1. LOAD DATA ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df_new = pd.read_csv(os.path.join(BASE_DIR, 'historical_new.csv'))
    matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_wisata_new.npy'))
    dur_matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_durasi_new.npy'))

    print("==============================================")
    print("      INDIVIDUAL ALGORITHM RUNNER")
    print("==============================================")
    print("Pilih Algoritma:")
    print("1. PSO (Particle Swarm Optimization)")
    print("2. Tabu Search")
    print("3. Hybrid GA (Genetic Algorithm)")
    print("4. ACO (Ant Colony Optimization)")
    
    choice = input("\nMasukkan pilihan (1-4): ")
    
    start_city = input("Masukkan Kota Keberangkatan (Surabaya/Sidoarjo/dll): ") or "Surabaya"
    max_days = int(input("Berapa hari perjalanan? (1-7): ") or "3")
    iters = int(input("Jumlah Iterasi/Generasi: ") or "500")

    algo = None
    if choice == '1':
        algo = PSOAlgorithm("PSO", df_new, matrix_new, dur_matrix_new, start_city=start_city, max_days=max_days, generations=iters)
    elif choice == '2':
        algo = TabuSearch("Tabu Search", df_new, matrix_new, dur_matrix_new, start_city=start_city, max_days=max_days, iterations=iters)
    elif choice == '3':
        algo = HybridGA("Hybrid GA", df_new, matrix_new, dur_matrix_new, start_city=start_city, max_days=max_days, generations=iters)
    elif choice == '4':
        algo = ACOAlgorithm("ACO", df_new, matrix_new, dur_matrix_new, start_city=start_city, max_days=max_days, iterations=iters)
    else:
        print("Pilihan tidak valid!")
        return

    print(f"\n🚀 Menjalankan {algo.name}...")
    start_time = time.time()
    route, dist, itinerary, days = algo.run()
    end_time = time.time()

    print("\n" + "="*40)
    print(f"HASIL AKHIR {algo.name.upper()}")
    print("="*40)
    print(f"Total Jarak    : {dist:.2f} km")
    print(f"Total Hari     : {days} hari")
    print(f"Running Time   : {end_time - start_time:.2f} detik")
    print(f"Jumlah Wisata  : {len([x for x in itinerary if not x.get('is_mobilisasi')])} lokasi")
    print("-" * 40)
    
    print("\nDetail Itinerary:")
    current_day = 0
    for item in itinerary:
        if item['day'] != current_day:
            current_day = item['day']
            print(f"\n--- HARI {current_day} ---")
        
        prefix = "[ Wisata ]" if not item.get('is_mobilisasi') else "[ Travel ]"
        print(f"{item['arrive']} - {item['depart']} | {prefix} {item['place']} ({item['duration']} mnt)")

if __name__ == "__main__":
    run_individual_algo()
