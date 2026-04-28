import random
from ga_v1 import GeneticAlgorithm

class AdvancedMutationGA(GeneticAlgorithm):
    def mutation(self, individu):
        if random.random() < self.mut_rate:
            # 50% Swap, 50% Inversion
            if random.random() < 0.5:
                i1, i2 = random.sample(range(len(individu)), 2)
                individu[i1], individu[i2] = individu[i2], individu[i1]
            else:
                start, end = sorted(random.sample(range(len(individu)), 2))
                individu[start:end] = individu[start:end][::-1]
        return individu