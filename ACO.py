import numpy as np
import random


class ACOAlgorithm:
    def __init__(self, name, df_lokasi, distance_matrix, duration_matrix,
                 start_city='Surabaya', max_days=3,
                 n_ants=28, iterations=500,
                 alpha=1.0, beta=2.0, rho=0.1, q=100.0,
                 elite_weight=2.0):
        self.name = name
        self.df = df_lokasi
        self.matrix = distance_matrix
        self.dur_matrix = duration_matrix
        self.start_city = start_city
        self.max_days = max_days

        #ACO hyperparameters
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha                  #bobot pheromone
        self.beta = beta                    #bobot heuristic (rating/distance)
        self.rho = rho                      #evaporation rate
        self.q = q                          #konstanta deposit pheromone
        self.elite_weight = elite_weight    #multiplier deposit untuk elite route

        self.jumlah_tempat = len(distance_matrix)

        #Hasil terbaik
        self.best_route = None
        self.best_fitness = -1
        self.best_distance = float('inf')
        self.best_itinerary = []
        self.best_days = 0
        self.history = []

        #Mapping kolom dataset
        self.city_col = 'Kota'
        self.name_col = 'Nama Tempat'
        self.duration_col = 'Durasi Kunjungan (menit)'
        self.open_col = 'Jam Buka'
        self.close_col = 'Jam Tutup'

        #Ekstrak data ke list Python
        self.cities = self.df[self.city_col].tolist()
        self.names = self.df[self.name_col].tolist()
        self.durations = self.df[self.duration_col].astype(int).tolist()
        self.ratings = self.df['Rating'].astype(float).tolist()

        self.open_mins = []
        self.close_mins = []
        for _, row in self.df.iterrows():
            b_parts = str(row[self.open_col]).split(':')
            t_parts = str(row[self.close_col]).split(':')
            b_h, b_m = int(b_parts[0]), int(b_parts[1])
            t_h, t_m = int(t_parts[0]), int(t_parts[1])
            self.open_mins.append(b_h * 60 + b_m)
            self.close_mins.append(t_h * 60 + t_m)

        #Inisialisasi pheromone & heuristic matrix
        self.max_rating = max(self.ratings) if self.ratings else 1.0
        self._init_pheromone()
        self._init_heuristic()

        #Cari kandidat start indices (tempat di start_city)
        self.start_indices = [i for i in range(self.jumlah_tempat)
                              if self.cities[i] == self.start_city]
        if not self.start_indices:
            self.start_indices = [0]

    def _init_pheromone(self):
        base_pheromone = 1.0
        self.pheromone = np.ones((self.jumlah_tempat, self.jumlah_tempat)) * base_pheromone

        #Bias dengan rating tempat tujuan
        for j in range(self.jumlah_tempat):
            rating_factor = self.ratings[j] / self.max_rating
            self.pheromone[:, j] *= rating_factor

        #Tutup self-loop
        np.fill_diagonal(self.pheromone, 0.0)

    def _init_heuristic(self):
        self.heuristic = np.zeros((self.jumlah_tempat, self.jumlah_tempat))
        for i in range(self.jumlah_tempat):
            for j in range(self.jumlah_tempat):
                if i == j:
                    continue
                dist = self.matrix[i][j]
                if dist <= 0:
                    dist = 0.1  #hindari division by zero/kasus jarak nol
                self.heuristic[i][j] = self.ratings[j] / dist

    def hitung_itinerary(self, rute):
        current_day = 1
        current_time = 8 * 60  #Mulai jam 08:00
        itinerary = []
        total_distance = 0

        for i in range(len(rute)):
            idx = rute[i]
            kota = self.cities[idx]

            if i > 0:
                travel_dist = self.matrix[rute[i-1]][idx]
                total_distance += travel_dist
                travel_time = self.dur_matrix[rute[i-1]][idx]

                #Tambahkan entri Mobilisasi
                start_h, start_m = int(current_time // 60), int(current_time % 60)
                end_time = current_time + travel_time
                end_h, end_m = int(end_time // 60), int(end_time % 60)

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
                current_time = 8 * 60  #Reset ke jam 08:00
                if current_day > self.max_days:
                    #Hapus mobilisasi terakhir jika ada
                    if itinerary and itinerary[-1].get('is_mobilisasi'):
                        itinerary.pop()
                    break

                arrival_time = current_time + 30
                if arrival_time < buka_menit:
                    arrival_time = buka_menit
                finish_time = arrival_time + durasi

            arr_h, arr_m = int(arrival_time // 60), int(arrival_time % 60)
            fin_h, fin_m = int(finish_time // 60), int(finish_time % 60)

            itinerary.append({
                'day': current_day,
                'city': kota,
                'place': self.names[idx],
                'place_idx': idx,
                'arrive': f"{arr_h:02d}:{arr_m:02d}",
                'depart': f"{fin_h:02d}:{fin_m:02d}",
                'duration': durasi,
                'is_mobilisasi': False
            })
            current_time = finish_time

        # --- HITUNG FITNESS BERDASARKAN RATING & DIVERSITAS KOTA ---
        # 1. Total Rating (Diberi pangkat agar rating tinggi jauh lebih berharga)
        total_rating = sum([(self.ratings[item['place_idx']] ** 2)
                            for item in itinerary if not item.get('is_mobilisasi')])

        # 2. Diversitas Kota & Penalti Kelengkapan
        cities_visited_list = [item['city'] for item in itinerary if not item.get('is_mobilisasi')]
        cities_visited_unique = set(cities_visited_list)
        unique_cities_count = len(cities_visited_unique)
        
        # Ambil daftar kota yang seharusnya dikunjungi (dari dataset yang sudah difilter di app.py)
        selected_cities_count = len(set(self.cities))
        
        missing_city_penalty = 0
        if unique_cities_count < selected_cities_count:
            missing_city_penalty = (selected_cities_count - unique_cities_count) * 100000

        city_reward = (unique_cities_count ** 3) * 5000

        # 3. Kuantitas Wisata
        jumlah_wisata = len([item for item in itinerary if not item.get('is_mobilisasi')])

        # 4. PENALTY: City Jumping (kembali ke kota yang sudah ditinggalkan)
        city_sequence = []
        for c in cities_visited_list:
            if not city_sequence or c != city_sequence[-1]:
                city_sequence.append(c)
        
        city_jump_penalty = (len(city_sequence) - unique_cities_count) * 10000

        fitness = (total_rating * 200 + city_reward + jumlah_wisata * 100) / (total_distance + city_jump_penalty + missing_city_penalty + 1)

        return fitness, itinerary, total_distance, current_day

    def construct_ant_route(self):
        # Mulai dari tempat random di start_city
        current = random.choice(self.start_indices)
        route = [current]
        unvisited = set(range(self.jumlah_tempat))
        unvisited.remove(current)

        while unvisited:
            current_city = self.cities[current]
            candidates = list(unvisited)
            
            # PRIORITAS: Selesaikan kota saat ini sebelum pindah
            same_city_candidates = [j for j in candidates if self.cities[j] == current_city]
            
            # Jika masih ada tempat di kota yang sama, paksa pilih dari situ
            search_pool = same_city_candidates if same_city_candidates else candidates

            # Hitung skor untuk search_pool
            tau_row = self.pheromone[current]
            eta_row = self.heuristic[current]

            scores = np.array([
                (tau_row[j] ** self.alpha) * (eta_row[j] ** self.beta)
                for j in search_pool
            ])

            total = scores.sum()
            if total <= 0 or not np.isfinite(total):
                next_place = random.choice(search_pool)
            else:
                probs = scores / total
                next_place = int(np.random.choice(search_pool, p=probs))

            route.append(next_place)
            unvisited.remove(next_place)
            current = next_place

        return route

    def update_pheromone(self, all_routes, all_fitnesses):
        #1. Evaporasi
        self.pheromone *= (1.0 - self.rho)

        #2. Deposit dari semua semut (proportional ke fitness)
        max_fit = max(all_fitnesses) if all_fitnesses else 1.0
        if max_fit <= 0:
            max_fit = 1.0

        for route, fitness in zip(all_routes, all_fitnesses):
            if fitness <= 0:
                continue
            normalized_fit = fitness / max_fit
            deposit = self.q * normalized_fit
            #Deposit di edge yang dilalui semut
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                self.pheromone[a][b] += deposit

        #3. Elitist deposit: best route ever dapat deposit ekstra
        if self.best_route is not None and self.best_fitness > 0:
            elite_deposit = self.q * self.elite_weight
            for i in range(len(self.best_route) - 1):
                a, b = self.best_route[i], self.best_route[i + 1]
                self.pheromone[a][b] += elite_deposit

        #4. Clip pheromone agar tidak meledak/menghilang (MMAS-style soft clip)
        np.clip(self.pheromone, 1e-6, 1e6, out=self.pheromone)

    def run(self):
        print(f"🚀 Menjalankan {self.name}...")
        self.history = []
        self.best_fitness = -1

        for iter_num in range(self.iterations):
            all_routes = []
            all_fitnesses = []

            #Setiap semut bangun rute & dievaluasi
            for _ in range(self.n_ants):
                route = self.construct_ant_route()
                fitness, itin, dist, days = self.hitung_itinerary(route)

                all_routes.append(route)
                all_fitnesses.append(fitness)

                #Update global best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_route = list(route)
                    self.best_distance = dist
                    self.best_itinerary = itin
                    self.best_days = days

            #Update pheromone setelah semua semut selesai (offline update)
            self.update_pheromone(all_routes, all_fitnesses)

            #Catat history: jumlah wisata pada best route saat ini
            wisata_count = len([x for x in self.best_itinerary if not x.get('is_mobilisasi')])
            self.history.append(wisata_count)

            if (iter_num + 1) % 100 == 0:
                print(f"      Iter {iter_num + 1:<4} | Wisata: {wisata_count} | Jarak: {self.best_distance:.2f} km")

        return self.best_route, self.best_distance, self.best_itinerary, self.best_days