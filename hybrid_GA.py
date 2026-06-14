import numpy as np
import random

class HybridGA:
    """
    Hybrid Genetic Algorithm - Standalone Implementation
    Mencari rute optimal dengan:
    - Elite preservation (elitism)
    - Tournament selection
    - Adaptive mutation rate
    - Adaptive crossover (Order, Cycle)
    Sama seperti PSO dan Tabu Search (dengan mobilisasi)
    """
    
    def __init__(self, name, df_lokasi, distance_matrix, duration_matrix, start_city='Surabaya', max_days=3, pop_size=200, generations=500, cx_rate=0.8, mut_rate=0.1, elite_size=20, tournament_size=3):
        self.name = name
        self.df = df_lokasi
        self.matrix = distance_matrix
        self.dur_matrix = duration_matrix
        self.start_city = start_city
        self.max_days = max_days
        self.pop_size = pop_size
        self.generations = generations
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.jumlah_tempat = len(distance_matrix)
        
        self.best_route = None
        self.best_fitness = -1
        self.best_distance = float('inf')
        self.best_itinerary = []
        self.best_days = 0
        self.history = []
        
        # Mapping kolom dataset
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
        """Inisialisasi populasi: Selesaikan satu kota baru pindah ke kota lain.
        Dimulai dari start_city."""
        populasi = []
        indices = list(range(self.jumlah_tempat))
        all_cities_in_df = list(self.df[self.city_col].unique())
        
        if self.start_city in all_cities_in_df:
            all_cities_in_df.remove(self.start_city)
        ordered_cities = [self.start_city] + all_cities_in_df
        
        city_groups = {c: self.df[self.df[self.city_col] == c].index.tolist() for c in ordered_cities}

        for _ in range(self.pop_size):
            individu = []
            for c in ordered_cities:
                group = city_groups[c][:]
                random.shuffle(group)
                individu.extend(group)
            populasi.append(individu)
        return populasi

    def hitung_itinerary(self, rute):
        """Hitung itinerary dengan mobilisasi (sama seperti PSO/Tabu)"""
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
                travel_time = self.dur_matrix[rute[i-1]][idx]
                
                # Tambahkan mobilisasi
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
        
        # Hitung fitness (sama seperti PSO/Tabu)
        visited_cities_list = [item['city'] for item in itinerary if not item.get('is_mobilisasi')]
        total_rating = sum([float(self.df[self.df[self.name_col] == item['place']]['Rating'].values[0]) 
                           for item in itinerary 
                           if not item.get('is_mobilisasi')])
        
        unique_cities_visited = set(visited_cities_list)
        unique_cities_count = len(unique_cities_visited)
        city_reward = (unique_cities_count ** 2) * 1000
        jumlah_wisata = len([item for item in itinerary if not item.get('is_mobilisasi')])
        
        # PENALTY: City Jumping
        city_sequence = []
        for c in visited_cities_list:
            if not city_sequence or c != city_sequence[-1]:
                city_sequence.append(c)
        city_jump_penalty = (len(city_sequence) - unique_cities_count) * 10000

        fitness = (total_rating * 150 + city_reward + jumlah_wisata * 100) / (total_distance + city_jump_penalty + 1)
        
        return fitness, itinerary, total_distance, current_day

    def evaluasi(self, populasi):
        """Evaluasi fitness untuk semua individu"""
        fitness_list = []
        for rute in populasi:
            fit, _, _, _ = self.hitung_itinerary(rute)
            fitness_list.append(fit)
        return np.array(fitness_list)

    def tournament_selection(self, pop, fitness):
        """Tournament selection - pilih parent terbaik"""
        candidates_idx = random.sample(range(len(pop)), self.tournament_size)
        best_idx = max(candidates_idx, key=lambda i: fitness[i])
        return pop[best_idx][:]

    def adaptive_crossover(self, p1, p2):
        """Adaptive crossover dengan multiple methods"""
        crossover_type = random.choice(['order_crossover', 'cycle_crossover'])
        
        if crossover_type == 'order_crossover':
            size = len(p1)
            a, b = sorted(random.sample(range(size), 2))
            
            child = [-1] * size
            child[a:b+1] = p1[a:b+1]
            
            pointer = (b + 1) % size
            for gene in p2:
                if gene not in child:
                    while child[pointer] != -1:
                        pointer = (pointer + 1) % size
                    child[pointer] = gene
            
            return child
        else:  # cycle crossover
            size = len(p1)
            child = [-1] * size
            visited = [False] * size
            
            cycle_start = 0
            while -1 in child:
                idx = cycle_start
                while True:
                    child[idx] = p1[idx]
                    visited[idx] = True
                    idx = p2.index(p1[idx])
                    if idx == cycle_start:
                        break
                
                for i in range(size):
                    if not visited[i]:
                        cycle_start = i
                        break
                
                if -1 not in child:
                    break
                for i in range(size):
                    if child[i] == -1:
                        child[i] = p2[i]
            
            return child

    def adaptive_mutation(self, child, gen):
        """Adaptive mutation dengan decreasing rate. Proteksi titik start (indeks 0)."""
        current_mut_rate = self.mut_rate * (1 - (gen / self.generations) ** 1.5)
        current_mut_rate = max(0.01, current_mut_rate)
        
        if random.random() < current_mut_rate:
            mutation_type = random.choice(['swap', 'insert', 'inversion'])
            
            if mutation_type == 'swap':
                # Swap tidak boleh melibatkan indeks 0
                idx1, idx2 = random.sample(range(1, len(child)), 2)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            
            elif mutation_type == 'insert':
                # Insert tidak boleh melibatkan indeks 0
                idx = random.randint(1, len(child) - 1)
                place = child.pop(idx)
                new_idx = random.randint(1, len(child))
                child.insert(new_idx, place)
            
            elif mutation_type == 'inversion':
                # Inversion tidak boleh melibatkan indeks 0
                idx1, idx2 = sorted(random.sample(range(1, len(child)), 2))
                child[idx1:idx2 + 1] = child[idx1:idx2 + 1][::-1]
        
        return child

    def run(self):
        """Main GA execution"""
        print(f"🚀 Menjalankan {self.name}...")
        pop = self.init_populasi()
        self.history = []
        self.best_fitness = -1
        
        for gen in range(self.generations):
            # Evaluasi fitness
            fit = self.evaluasi(pop)
            
            # Cari best individual
            best_idx = np.argmax(fit)
            
            # Update best solution
            if fit[best_idx] > self.best_fitness:
                self.best_fitness = fit[best_idx]
                self.best_route = pop[best_idx][:]
                _, itin, dist, days = self.hitung_itinerary(self.best_route)
                self.best_distance = dist
                self.best_itinerary = itin
                self.best_days = days
            
            # Simpan history (jumlah wisata, bukan jarak)
            num_wisata = len([x for x in self.best_itinerary if not x.get('is_mobilisasi')])
            self.history.append(num_wisata)
            
            # Print progress
            if (gen + 1) % 100 == 0:
                num_wisata = len([x for x in self.best_itinerary if not x.get('is_mobilisasi')])
                print(f"      Gen {gen + 1:<4} | Wisata: {num_wisata} | Jarak: {self.best_distance:.2f} km")
            
            # Elitism: simpan top individuals
            elite_indices = np.argsort(fit)[-self.elite_size:]
            elite_pop = [pop[i][:] for i in elite_indices]
            
            # Generate new population
            new_pop = elite_pop[:]
            
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_selection(pop, fit)
                p2 = self.tournament_selection(pop, fit)
                
                if random.random() < self.cx_rate:
                    child = self.adaptive_crossover(p1, p2)
                else:
                    child = p1[:]
                
                child = self.adaptive_mutation(child, gen)
                new_pop.append(child)
            
            pop = new_pop[:self.pop_size]
        
        return self.best_route, self.best_distance, self.best_itinerary, self.best_days
