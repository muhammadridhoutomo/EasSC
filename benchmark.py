import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Import algorithms
from discrete_pso import PSOAlgorithm
from tabu_search import TabuSearch
from hybrid_GA import HybridGA
from ACO import ACOAlgorithm

# --- 1. LOAD DATA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_new = pd.read_csv(os.path.join(BASE_DIR, 'historical_new.csv'))
matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_wisata_new.npy'))
dur_matrix_new = np.load(os.path.join(BASE_DIR, 'matriks_durasi_new.npy'))

# --- 2. BENCHMARK CONFIGURATION ---
NUM_RUNS = 10
START_CITY = "Surabaya"
MAX_DAYS = 3
GENERATIONS = 200 # Using 200 for faster execution while maintaining comparability

# Algorithm list
algorithm_classes = [
    ("PSO", PSOAlgorithm),
    ("Tabu Search", TabuSearch),
    ("Hybrid GA", HybridGA),
    ("ACO", ACOAlgorithm)
]

# Results storage
all_results = {name: {"distances": [], "times": []} for name, _ in algorithm_classes}

print(f"🚀 Starting Benchmark ({NUM_RUNS}x runs per algorithm)...")
print(f"Config: Start={START_CITY}, Duration={MAX_DAYS} Days, Iterations/Gens={GENERATIONS}\n")

# --- 3. EXECUTION LOOP ---
for name, algo_class in algorithm_classes:
    print(f"--- Running {name} ---")
    for i in range(NUM_RUNS):
        # Initialize
        if name == "Tabu Search" or name == "ACO":
            algo = algo_class(name, df_new, matrix_new, dur_matrix_new, 
                              start_city=START_CITY, max_days=MAX_DAYS, iterations=GENERATIONS)
        else:
            algo = algo_class(name, df_new, matrix_new, dur_matrix_new, 
                              start_city=START_CITY, max_days=MAX_DAYS, generations=GENERATIONS)

        start_time = time.time()
        _, best_dist, _, _ = algo.run()
        end_time = time.time()

        exec_time = end_time - start_time
        all_results[name]["distances"].append(best_dist)
        all_results[name]["times"].append(exec_time)
        
        print(f"   Run {i+1}/{NUM_RUNS} | Distance: {best_dist:.2f} km | Time: {exec_time:.2f} s")
    print()

# --- 4. CALCULATE STATISTICS ---
stats_summary = []
for name in all_results:
    dists = all_results[name]["distances"]
    times = all_results[name]["times"]
    
    summary = {
        "Algorithm": name,
        "Best (Min) Distance": np.min(dists),
        "Avg Distance": np.mean(dists),
        "Std Dev Distance": np.std(dists),
        "Avg Run Time (s)": np.mean(times)
    }
    stats_summary.append(summary)

df_stats = pd.DataFrame(stats_summary)
print("\n" + "="*70)
print("FINAL COMPARISON RESULTS")
print("="*70)
print(df_stats.to_string(index=False))
print("="*70)

# --- 5. GRAPH GENERATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Distance Comparison
x = np.arange(len(df_stats))
width = 0.35

ax1.bar(x - width/2, df_stats["Best (Min) Distance"], width, label='Best (Min)', color='#06D6A0')
ax1.bar(x + width/2, df_stats["Avg Distance"], width, label='Average', color='#118AB2')
ax1.set_xticks(x)
ax1.set_xticklabels(df_stats["Algorithm"])
ax1.set_ylabel('Total Distance (km)')
ax1.set_title('Distance Performance (Lower is Better)')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Running Time Comparison
ax2.bar(df_stats["Algorithm"], df_stats["Avg Run Time (s)"], color='#EF476F')
ax2.set_ylabel('Average Time (seconds)')
ax2.set_title('Average Running Time')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('Hasil_Komparasi_Algoritma.png', dpi=300)
print("\n✅ Graphics saved as: 'Hasil_Komparasi_Algoritma.png'")
