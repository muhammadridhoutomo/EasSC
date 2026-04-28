import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, name, distance_matrix, pop_size=200, generations=1000, cx_rate=0.8, mut_rate=0.1):
        self.name = name
        self.matrix = distance_matrix
        self.pop_size = pop_size
        self.generations = generations
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.jumlah_tempat = len(distance_matrix)
        self.best_route = None
        self.best_distance = float('inf')

    def init_populasi(self):
        populasi = []
        for _ in range(self.pop_size):
            individu = list(range(self.jumlah_tempat))
            random.shuffle(individu)
            populasi.append(individu)
        return populasi

    def hitung_jarak(self, rute):
        jarak = 0
        for i in range(len(rute) - 1):
            jarak += self.matrix[rute[i]][rute[i+1]]
        jarak += self.matrix[rute[-1]][rute[0]]
        return jarak

    def evaluasi(self, populasi):
        fitness_list = []
        for rute in populasi:
            jarak = self.hitung_jarak(rute)
            fitness = 1.0 / float(jarak) if jarak > 0 else 0
            fitness_list.append(fitness)
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
            curr_dist = self.hitung_jarak(pop[best_idx])
            
            if curr_dist < self.best_distance:
                self.best_distance = curr_dist
                self.best_route = pop[best_idx]

            new_pop = [self.best_route]
            while len(new_pop) < self.pop_size:
                p1, p2 = self.selection(pop, fit)
                child = self.crossover(p1, p2)
                child = self.mutation(child)
                new_pop.append(child)
            pop = new_pop
            
            # --- TAMBAHAN: UPDATE TERMINAL PER 100 GENERASI ---
            if (gen + 1) % 100 == 0:
                print(f"      Generasi {gen + 1:<4} | Jarak Terpendek Sementara: {self.best_distance:.2f} km")
                
        return self.best_route, self.best_distance