import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Import algorithms
from discrete_pso import PSOAlgorithm
from tabu_search import TabuSearch
from adaptive_GA import AdaptiveGA
from ACO import ACOAlgorithm

def run_detailed_benchmark():
    # --- 1. LOAD DATA ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df_new = pd.read_csv(os.path.join(BASE_DIR, 'historical_new.csv'))
    matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_wisata_new.npy'))
    dur_matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_durasi_new.npy'))

    print("==============================================")
    print("      DETAILED ALGORITHM BENCHMARK (10X)")
    print("==============================================")
    print("Pilih Algoritma yang ingin dianalisis:")
    print("1. Discrete PSO (Particle Swarm Optimization)")
    print("2. Tabu Search")
    print("3. Adaptive GA (Genetic Algorithm)")
    print("4. ACO (Ant Colony Optimization)")

    choice = input("\nMasukkan pilihan (1-4): ").strip()
    
    start_city = input("Masukkan Kota Keberangkatan: ").strip() or "Surabaya"
    
    raw_days = input("Berapa hari perjalanan? (1-7): ").strip()
    max_days = int(''.join(filter(str.isdigit, raw_days))) if raw_days else 3
    
    raw_iters = input("Jumlah Iterasi/Generasi (100-500): ").strip()
    iters = int(''.join(filter(str.isdigit, raw_iters))) if raw_iters else 200

    algo_map = {
        '1': ("Discrete PSO", PSOAlgorithm),
        '2': ("Tabu Search", TabuSearch),
        '3': ("Adaptive GA", AdaptiveGA),
        '4': ("ACO", ACOAlgorithm)
    }

    if choice not in algo_map:
        print("Pilihan tidak valid!")
        return

    name, algo_class = algo_map[choice]
    
    distances = []
    runtimes = []
    
    print(f"\n🚀 Memulai analisis detail untuk {name}...")
    print(f"Konfigurasi: 10 kali running, {iters} iterasi.\n")

    for i in range(10):
        # Initialize
        if name in ["Tabu Search", "ACO"]:
            algo = algo_class(name, df_new, matrix_new, dur_matrix_new, 
                              start_city=start_city, max_days=max_days, iterations=iters)
        else:
            algo = algo_class(name, df_new, matrix_new, dur_matrix_new, 
                              start_city=start_city, max_days=max_days, generations=iters)

        start_time = time.time()
        _, dist, _, _ = algo.run()
        end_time = time.time()
        
        exec_time = end_time - start_time
        distances.append(dist)
        runtimes.append(exec_time)
        
        print(f"✅ Run {i+1}/10: Jarak = {dist:.2f} km | Waktu = {exec_time:.2f} s")

    # --- 2. HITUNG STATISTIK ---
    best_dist = np.min(distances)
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)
    avg_time = np.mean(runtimes)

    print("\n" + "="*50)
    print(f"HASIL ANALISIS DETAIL: {name}")
    print("="*50)
    print(f"1. Nilai Terbaik (Minimum) : {best_dist:.2f} km")
    print(f"2. Nilai Rata-rata Jarak   : {avg_dist:.2f} km")
    print(f"3. Standar Deviasi Jarak   : {std_dist:.2f}")
    print(f"4. Rata-rata Running Time  : {avg_time:.2f} detik")
    print("="*50)

    # --- 3. VISUALISASI ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Jarak per Run
    ax1.plot(range(1, 11), distances, marker='o', linestyle='-', color='b', label='Jarak tiap Run')
    ax1.axhline(avg_dist, color='r', linestyle='--', label=f'Rata-rata ({avg_dist:.2f})')
    ax1.set_title(f'Stabilitas Jarak {name} (10 Run)')
    ax1.set_xlabel('Run ke-')
    ax1.set_ylabel('Jarak (km)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Running Time per Run
    ax2.bar(range(1, 11), runtimes, color='orange', alpha=0.7)
    ax2.axhline(avg_time, color='red', linestyle='--', label=f'Rata-rata ({avg_time:.2f}s)')
    ax2.set_title(f'Running Time {name} per Run')
    ax2.set_xlabel('Run ke-')
    ax2.set_ylabel('Waktu (detik)')
    ax2.legend()

    plt.tight_layout()
    plot_filename = f'Analisis_Detail_{name.replace(" ", "_")}.png'
    plt.savefig(plot_filename)
    print(f"\n✅ Grafik analisis detail disimpan sebagai: {plot_filename}")
    plt.show()

if __name__ == "__main__":
    run_detailed_benchmark()
