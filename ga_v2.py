import random
from ga_v1 import GeneticAlgorithm

class TournamentGA(GeneticAlgorithm):
    def selection(self, populasi, fitness_list):
        k = 5 # Ukuran turnamen
        selected = []
        for _ in range(2):
            candidates_idx = random.sample(range(len(populasi)), k)
            best_cand = max(candidates_idx, key=lambda i: fitness_list[i])
            selected.append(populasi[best_cand])
        return selected[0], selected[1]