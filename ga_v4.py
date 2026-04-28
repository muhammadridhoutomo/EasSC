import numpy as np
from ga_v1 import GeneticAlgorithm

class AdaptiveGA(GeneticAlgorithm):
    def run(self):
        print(f"🚀 Menjalankan {self.name}...")
        pop = self.init_populasi()
        for gen in range(self.generations):
            # Mutasi adaptif: semakin lama generasi, rate makin kecil
            self.mut_rate = max(0.01, 0.1 * (1 - gen/self.generations))
            
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
            
            # --- TAMBAHAN: Tampilkan ke terminal per 100 generasi ---
            if (gen + 1) % 100 == 0:
                print(f"      Generasi {gen + 1:<4} | Jarak Terpendek Sementara: {self.best_distance:.2f} km")
                
        return self.best_route, self.best_distance