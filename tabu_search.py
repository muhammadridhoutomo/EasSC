import random
import numpy as np
from ga_v1 import GeneticAlgorithm

class TabuSearch(GeneticAlgorithm):
    def __init__(self, name, df_lokasi, distance_matrix, duration_matrix, start_city='Surabaya', max_days=3, tenure=15, iterations=500):
        # Gunakan parameter yang konsisten dengan PSO
        super().__init__(name, df_lokasi, distance_matrix)
        self.dur_matrix = duration_matrix
        self.start_city = start_city
        self.max_days = max_days
        self.tenure = tenure
        self.max_iterations = iterations
        
        # Mapping kolom dataset baru (historical_new.csv)
        self.city_col = 'Kota'
        self.name_col = 'Nama Tempat'
        self.duration_col = 'Durasi Kunjungan (menit)'
        self.open_col = 'Jam Buka'
        self.close_col = 'Jam Tutup'

        # Ekstrak data (Sama dengan PSO)
        self.cities = self.df[self.city_col].tolist()
        self.names = self.df[self.name_col].tolist()
        self.durations = self.df[self.duration_col].astype(int).tolist()
        
        self.open_mins = []
        self.close_mins = []
        for _, row in self.df.iterrows():
            b_h, b_m = map(int, str(row[self.open_col]).split(':'))
            t_h, t_m = map(int, str(row[self.close_col]).split(':'))
            self.open_mins.append(b_h * 60 + b_m)
            self.close_mins.append(t_h * 60 + t_m)

    def init_solution(self):
        """Inisialisasi rute dengan memastikan kota start di urutan pertama, 
        diikuti oleh campuran acak semua lokasi untuk mendorong multi-city trip."""
        indices = list(range(self.jumlah_tempat))
        start_indices = self.df[self.df[self.city_col] == self.start_city].index.tolist()
        if not start_indices: start_indices = [0]
        
        # Acak urutan di dalam kota start
        random.shuffle(start_indices)
        
        # Ambil sisa tempat (termasuk dari kota lain)
        remaining = [idx for idx in indices if idx not in start_indices]
        random.shuffle(remaining)
        
        return start_indices + remaining

    def hitung_itinerary(self, rute):
        """
        Logika penjadwalan dengan menyertakan waktu MOBILISASI antar lokasi.
        """
        current_day = 1
        current_time = 8 * 60 
        itinerary = []
        total_distance = 0
        
        for i in range(len(rute)):
            idx = rute[i]
            kota = self.cities[idx]
            if i > 0:
                travel_dist = self.matrix[rute[i-1]][idx]
                total_distance += travel_dist
                # Gunakan durasi nyata dari OSRM
                travel_time = self.dur_matrix[rute[i-1]][idx]
                
                start_h, start_m = int(current_time//60), int(current_time%60)
                end_time = current_time + travel_time
                end_h, end_m = int(end_time//60), int(end_time%60)
                
                itinerary.append({
                    'day': current_day,
                    'city': kota,
                    'place': '🚗 Mobilisasi',
                    'arrive': f"{start_h:02d}:{start_m:02d}",
                    'depart': f"{end_h:02d}:{end_m:02d}",
                    'duration': int(travel_time),
                    'is_mobilisasi': True
                })
                current_time = end_time
            else:
                travel_time = 0
            
            arrival_time = current_time
            if arrival_time < self.open_mins[idx]:
                arrival_time = self.open_mins[idx]
                
            finish_time = arrival_time + self.durations[idx]
            
            if finish_time > self.close_mins[idx] or finish_time > 21 * 60:
                current_day += 1
                current_time = 8 * 60
                if current_day > self.max_days:
                    if itinerary and itinerary[-1].get('is_mobilisasi'):
                        itinerary.pop()
                    break
                arrival_time = current_time + 30 
                if arrival_time < self.open_mins[idx]:
                    arrival_time = self.open_mins[idx]
                finish_time = arrival_time + self.durations[idx]

            arr_h, arr_m = int(arrival_time//60), int(arrival_time%60)
            fin_h, fin_m = int(finish_time//60), int(finish_time%60)
            
            itinerary.append({
                'day': current_day,
                'city': kota,
                'place': self.names[idx],
                'arrive': f"{arr_h:02d}:{arr_m:02d}",
                'depart': f"{fin_h:02d}:{fin_m:02d}",
                'duration': self.durations[idx],
                'is_mobilisasi': False
            })
            current_time = finish_time

        jumlah_wisata = len([item for item in itinerary if not item.get('is_mobilisasi')])
        fitness = (jumlah_wisata * 1000) / (total_distance + 1)
        return fitness, itinerary, total_distance, current_day

    def get_neighbors(self, solution):
        """Membangun tetangga dengan menukar dua tempat (Swap Neighborhood)"""
        neighbors = []
        # Batasi jumlah tetangga agar tidak terlalu lambat (misal 50 tetangga)
        for _ in range(50):
            idx1, idx2 = random.sample(range(len(solution)), 2)
            # Jangan ganggu posisi pertama jika itu kota start
            if idx1 == 0 or idx2 == 0: continue
            
            neighbor = list(solution)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            neighbors.append((neighbor, (idx1, idx2)))
        return neighbors

    def run(self):
        print(f"🚀 Menjalankan {self.name}...")
        current_solution = self.init_solution()
        best_solution = list(current_solution)
        
        # Evaluasi awal
        best_fit, best_itin, best_dist, best_d = self.hitung_itinerary(best_solution)
        
        tabu_list = {} # format {(idx1, idx2): expire_iteration}
        self.history = []

        for i in range(self.max_iterations):
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_fit = -1
            best_move = None
            
            for neighbor, move in neighbors:
                fit, itin, dist, days = self.hitung_itinerary(neighbor)
                
                # Cek apakah move ada di Tabu List
                is_tabu = False
                if move in tabu_list or (move[1], move[0]) in tabu_list:
                    # Aspiration Criterion: Jika tetangga ini lebih baik dari gbest, abaikan tabu
                    if fit > best_fit:
                        is_tabu = False
                    else:
                        is_tabu = True
                
                if not is_tabu:
                    if fit > best_neighbor_fit:
                        best_neighbor = neighbor
                        best_neighbor_fit = fit
                        best_move = move
            
            if best_neighbor:
                current_solution = best_neighbor
                # Update Global Best
                if best_neighbor_fit > best_fit:
                    best_solution = list(best_neighbor)
                    best_fit = best_neighbor_fit
                    _, best_itin, best_dist, best_d = self.hitung_itinerary(best_solution)
                
                # Update Tabu List
                tabu_list[best_move] = i + self.tenure
                
            # Clean tabu list
            tabu_list = {m: exp for m, exp in tabu_list.items() if exp > i}
            
            # Catat history (jumlah wisata)
            self.history.append(len(best_itin))
            
            if (i + 1) % 100 == 0:
                print(f"      Iter {i + 1:<4} | Wisata: {len(best_itin)} | Jarak: {best_dist:.2f} km")

        self.best_route = best_solution
        self.best_distance = best_dist
        self.best_itinerary = best_itin
        self.best_days = best_d
        return self.best_route, self.best_distance, self.best_itinerary, self.best_days
