import random
import numpy as np
import pandas as pd
from ga_v1 import GeneticAlgorithm

class Particle:
    def __init__(self, route):
        self.position = list(route)
        self.pbest_position = list(route)
        self.pbest_fitness = -1
        self.velocity = []

class PSOAlgorithm(GeneticAlgorithm):
    def __init__(self, name, df_lokasi, distance_matrix, duration_matrix, start_city='Surabaya', max_days=3, pop_size=100, generations=1000, w=0.7, c1=1.5, c2=1.5):
        super().__init__(name, df_lokasi, distance_matrix, pop_size, generations)
        self.dur_matrix = duration_matrix
        self.start_city = start_city
        self.max_days = max_days
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Mapping kolom dataset baru (historical_new.csv)
        self.city_col = 'Kota'
        self.name_col = 'Nama Tempat'
        self.duration_col = 'Durasi Kunjungan (menit)'
        self.open_col = 'Jam Buka'
        self.close_col = 'Jam Tutup'

        # Ekstrak data
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

    def init_populasi(self):
        """
        Inisialisasi populasi dengan memastikan rute dimulai dari kota yang dipilih user.
        """
        populasi = []
        indices = list(range(self.jumlah_tempat))
        
        # Cari semua tempat di kota awal
        start_indices = self.df[self.df[self.city_col] == self.start_city].index.tolist()
        
        if not start_indices:
            print(f"⚠️ Warning: Kota {self.start_city} tidak ditemukan. Menggunakan Surabaya sebagai default.")
            start_indices = self.df[self.df[self.city_col] == 'Surabaya'].index.tolist()

        for _ in range(self.pop_size):
            remaining = [idx for idx in indices if idx not in start_indices]
            random.shuffle(start_indices)
            random.shuffle(remaining)
            
            # Gabungkan: Tempat di kota awal diletakkan di depan
            individu = start_indices + remaining
            populasi.append(individu)
        return populasi

    def hitung_itinerary(self, rute):
        """
        Logika penjadwalan dengan menyertakan waktu MOBILISASI antar lokasi.
        """
        current_day = 1
        current_time = 8 * 60 # Mulai jam 08:00
        itinerary = []
        total_distance = 0
        
        for i in range(len(rute)):
            idx = rute[i]
            kota = self.cities[idx]
            
            if i > 0:
                travel_dist = self.matrix[rute[i-1]][idx]
                total_distance += travel_dist
                # Gunakan durasi nyata dari OSRM (menit)
                travel_time = self.dur_matrix[rute[i-1]][idx]
                
                # Tambahkan entri MOBILISASI ke itinerary
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
            buka_menit = self.open_mins[idx]
            tutup_menit = self.close_mins[idx]
            durasi = self.durations[idx]
            
            if arrival_time < buka_menit:
                arrival_time = buka_menit
                
            finish_time = arrival_time + durasi
            
            if finish_time > tutup_menit or finish_time > 21 * 60:
                current_day += 1
                current_time = 8 * 60 # Reset ke jam 8 pagi
                if current_day > self.max_days:
                    # Jika sudah lewat hari, hapus entri mobilisasi terakhir jika ada
                    if itinerary and itinerary[-1].get('is_mobilisasi'):
                        itinerary.pop()
                    break
                    
                arrival_time = current_time + 30 
                if arrival_time < buka_menit:
                    arrival_time = buka_menit
                finish_time = arrival_time + durasi

            arr_h, arr_m = int(arrival_time//60), int(arrival_time%60)
            fin_h, fin_m = int(finish_time//60), int(finish_time%60)
            
            itinerary.append({
                'day': current_day,
                'city': kota,
                'place': self.names[idx],
                'arrive': f"{arr_h:02d}:{arr_m:02d}",
                'depart': f"{fin_h:02d}:{fin_m:02d}",
                'duration': durasi,
                'is_mobilisasi': False
            })
            current_time = finish_time

        jumlah_wisata = len([item for item in itinerary if not item.get('is_mobilisasi')])
        fitness = (jumlah_wisata * 1000) / (total_distance + 1)
        return fitness, itinerary, total_distance, current_day

    def get_swap_sequence(self, source, target):
        swaps = []
        temp_source = list(source)
        for i in range(len(temp_source)):
            if temp_source[i] != target[i]:
                try:
                    idx_target = temp_source.index(target[i])
                    swaps.append((i, idx_target))
                    temp_source[i], temp_source[idx_target] = temp_source[idx_target], temp_source[i]
                except ValueError:
                    continue
        return swaps

    def apply_swaps(self, position, swaps, probability):
        new_position = list(position)
        for s1, s2 in swaps:
            if random.random() < probability:
                new_position[s1], new_position[s2] = new_position[s2], new_position[s1]
        return new_position

    def run(self):
        print(f"🚀 Menjalankan {self.name} (Start: {self.start_city}, Max: {self.max_days} Hari)...")
        routes = self.init_populasi()
        particles = [Particle(r) for r in routes]
        
        self.history = []
        gbest_position = None
        gbest_fitness = -1

        for gen in range(self.generations):
            for p in particles:
                fitness, itin, dist, days = self.hitung_itinerary(p.position)
                
                if fitness > p.pbest_fitness:
                    p.pbest_fitness = fitness
                    p.pbest_position = list(p.position)
                
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = list(p.position)
                    self.best_fitness = fitness
                    self.best_route = list(p.position)
                    self.best_distance = dist
                    self.best_itinerary = itin
                    self.best_days = days

            for p in particles:
                ss_pbest = self.get_swap_sequence(p.position, p.pbest_position)
                ss_gbest = self.get_swap_sequence(p.position, gbest_position)
                
                prob_pbest = self.c1 / (self.c1 + self.c2 + self.w)
                prob_gbest = self.c2 / (self.c1 + self.c2 + self.w)
                
                p.position = self.apply_swaps(p.position, ss_pbest, prob_pbest)
                p.position = self.apply_swaps(p.position, ss_gbest, prob_gbest)

                if random.random() < 0.1:
                    # Mutasi acak tapi jangan ganggu indeks pertama (kota start) jika ingin konsisten
                    # Tapi PSO biasanya bebas, yang penting evaluasi rutenya tetap efisien
                    idx1, idx2 = random.sample(range(len(p.position)), 2)
                    p.position[idx1], p.position[idx2] = p.position[idx2], p.position[idx1]

            # Di PSO custom ini, history mencatat jumlah wisata yang bisa dikunjungi
            self.history.append(len(self.best_itinerary))
            
            if (gen + 1) % 100 == 0:
                print(f"      Gen {gen + 1:<4} | Wisata: {len(self.best_itinerary)} | Jarak: {self.best_distance:.2f} km")

        return self.best_route, self.best_distance, self.best_itinerary, self.best_days
