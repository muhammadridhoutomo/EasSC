import numpy as np
from ga_v1 import GeneticAlgorithm

class AdaptiveGA(GeneticAlgorithm):
    def run(self):
        print(f"🚀 Menjalankan {self.name}...")
        pop = self.init_populasi()

        self.history = []

        for gen in range(self.generations):
            self.mut_rate = max(0.01, 0.1 * (1 - gen/self.generations))
            
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
            
            self.history.append(self.best_distance)

            if (gen + 1) % 100 == 0:
                print(f"      Gen {gen + 1:<4} | Jarak: {self.best_distance:.2f} km | Hari: {self.best_days}")
                
        return self.best_route, self.best_distance, self.best_itinerary, self.best_days