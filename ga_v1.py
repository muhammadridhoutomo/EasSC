import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, name, df_lokasi, distance_matrix, pop_size=200, generations=1000, cx_rate=0.8, mut_rate=0.1):
        self.name = name
        self.df = df_lokasi
        self.matrix = distance_matrix
        self.pop_size = pop_size
        self.generations = generations
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.jumlah_tempat = len(distance_matrix)
        
        self.best_route = None
        self.best_fitness = -1
        self.best_distance = float('inf')
        self.best_itinerary = []
        self.best_days = 0

        # === OPTIMASI KECEPATAN: Ekstrak Pandas ke Python List murni di awal ===
        self.cities = self.df['City'].tolist()
        self.names = self.df['Place_Name'].tolist()
        self.durations = self.df['Visit_Duration'].astype(int).tolist()
        
        self.open_mins = []
        self.close_mins = []
        for _, row in self.df.iterrows():
            b_h, b_m = map(int, str(row['Opening_Hours']).split(':'))
            t_h, t_m = map(int, str(row['Closing_Hours']).split(':'))
            self.open_mins.append(b_h * 60 + b_m)
            self.close_mins.append(t_h * 60 + t_m)

    def init_populasi(self):
        populasi = []
        sby = self.df[self.df['City'] == 'Surabaya'].index.tolist()
        sda = self.df[self.df['City'] == 'Sidoarjo'].index.tolist()
        mjk = self.df[self.df['City'] == 'Mojokerto'].index.tolist()
        mlg = self.df[self.df['City'] == 'Malang'].index.tolist()
        
        for _ in range(self.pop_size):
            random.shuffle(sby)
            random.shuffle(sda)
            random.shuffle(mjk)
            random.shuffle(mlg)
            individu = sby + sda + mjk + mlg
            populasi.append(individu)
        return populasi

    def hitung_itinerary(self, rute):
        current_day = 1
        current_time = 8 * 60 
        
        itinerary = []
        total_distance = 0
        city_changes = 0
        
        # Baca dari List, bukan dari dataframe
        current_city = self.cities[rute[0]]
        days_in_current_city = 1
        city_days = {}
        
        for i in range(len(rute)):
            idx = rute[i]
            kota = self.cities[idx]
            
            if i > 0:
                travel_dist = self.matrix[rute[i-1]][idx]
                total_distance += travel_dist
                travel_time = travel_dist * 1.5 
            else:
                travel_time = 0
                
            if kota != current_city:
                city_changes += 1
                city_days[current_city] = city_days.get(current_city, 0) + days_in_current_city
                current_city = kota
                days_in_current_city = 1
                current_day += 1 
                current_time = 8 * 60
                
            arrival_time = current_time + travel_time
            
            # Akses jam dari array jauh lebih cepat
            buka_menit = self.open_mins[idx]
            tutup_menit = self.close_mins[idx]
            durasi = self.durations[idx]
            
            if arrival_time < buka_menit:
                arrival_time = buka_menit
                
            finish_time = arrival_time + durasi
            
            if finish_time > tutup_menit or finish_time > 20 * 60:
                current_day += 1
                days_in_current_city += 1
                current_time = 8 * 60
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
                'duration': durasi
            })
            
            current_time = finish_time
            
        city_days[current_city] = city_days.get(current_city, 0) + days_in_current_city
        
        penalty = 0
        if current_day > 8: 
            penalty += (current_day - 8) * 1000 
        if city_changes > 3:
            penalty += (city_changes - 3) * 500 
        for c, d in city_days.items():
            if d > 2: 
                penalty += (d - 2) * 500 
                
        fitness = 10000.0 / (total_distance + penalty + 1)
        
        return fitness, itinerary, total_distance, current_day

    def evaluasi(self, populasi):
        fitness_list = []
        for rute in populasi:
            fit, _, _, _ = self.hitung_itinerary(rute)
            fitness_list.append(fit)
        return fitness_list

    def selection(self, populasi, fitness_list):
        total_f = sum(fitness_list)
        prob = [f/total_f for f in fitness_list]
        idx = np.random.choice(len(populasi), size=2, p=prob, replace=False)
        return populasi[idx[0]], populasi[idx[1]]

    def crossover(self, p1, p2):
        if random.random() > self.cx_rate: return p1.copy()
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end] = p1[start:end]
        p2_f = [item for item in p2 if item not in child]
        child[:start] = p2_f[:start]
        child[end:] = p2_f[start:]
        return child

    def mutation(self, individu):
        if random.random() < self.mut_rate:
            idx1, idx2 = random.sample(range(len(individu)), 2)
            individu[idx1], individu[idx2] = individu[idx2], individu[idx1]
        return individu

    def run(self):
        print(f"🚀 Menjalankan {self.name}...")
        pop = self.init_populasi()
        
        for gen in range(self.generations):
            fit = self.evaluasi(pop)
            best_idx = np.argmax(fit)
            
            if fit[best_idx] > self.best_fitness:
                self.best_fitness = fit[best_idx]
                self.best_route = pop[best_idx]
                _, itin, dist, days = self.hitung_itinerary(self.best_route)
                self.best_distance = dist
                self.best_itinerary = itin
                self.best_days = days

            new_pop = [self.best_route] 
            while len(new_pop) < self.pop_size:
                p1, p2 = self.selection(pop, fit)
                child = self.crossover(p1, p2)
                child = self.mutation(child)
                new_pop.append(child)
            pop = new_pop
            
            if (gen + 1) % 100 == 0:
                print(f"      Gen {gen + 1:<4} | Jarak: {self.best_distance:.2f} km | Hari: {self.best_days}")
                
        return self.best_route, self.best_distance, self.best_itinerary, self.best_days