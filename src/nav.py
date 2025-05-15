import random
import numpy as np
from utils import a_star
from constants import MAP_RESOLUTION
class GeneticAlgorithm:
    """
    Generic Genetic Algorithm for evolving controller parameters
    Concepts from Evolution & Learning slides:
    - Slide 4: Combining slow adaptation (evolution) with fast adaptation (learning)
    - Slide 10: Baldwin effect (learning guiding evolution)
    - Slide 12: Hinton & Nowlan model (smooth fitness landscape with learnable genes)
    """
    def __init__(self, pop_size, genome_length, eval_function,
                 crossover_rate=0.8, mutation_rate=0.1, mutation_scale=0.1):
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.eval_function = eval_function
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        # Initialize population: random genomes in [-1,1]
        self.population = [np.random.uniform(-1, 1, genome_length) for _ in range(pop_size)]
        self.fitness = [0.0] * pop_size

    def evaluate(self):
        """
        Evaluate each individual using provided eval_function:
        - eval_function(genome) -> fitness value
        """
        for i, genome in enumerate(self.population):
            # Slide 13: Evolutionary Reinforcement Learning (ERL) concept
            self.fitness[i] = self.eval_function(genome)

    def select_parents(self):
        """
        Tournament selection: choose two parents
        """
        parents = []
        for _ in range(self.pop_size):
            # pick two candidates and select the fitter
            i, j = random.sample(range(self.pop_size), 2)
            parents.append(self.population[i] if self.fitness[i] > self.fitness[j] else self.population[j])
        return parents

    def crossover(self, parent1, parent2):
        """
        Single-point crossover
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.genome_length - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, genome):
        """
        Gaussian mutation per gene
        """
        for idx in range(self.genome_length):
            if random.random() < self.mutation_rate:
                genome[idx] += np.random.normal(0, self.mutation_scale)
        return genome

    def run(self, generations):
        """
        Run the evolutionary loop
        """
        best_genome = None
        best_fitness = float('-inf')
        for gen in range(generations):
            self.evaluate()
            # record best (Slide 16: Evolution of Learning)
            gen_best = max(self.fitness)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_genome = self.population[self.fitness.index(gen_best)].copy()
            print(f"Gen {gen}: Best Fitness = {gen_best:.3f}")

            # Selection
            parents = self.select_parents()
            # Create next population
            next_pop = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = parents[i], parents[(i+1) % self.pop_size]
                c1, c2 = self.crossover(p1, p2)
                next_pop.append(self.mutate(c1))
                next_pop.append(self.mutate(c2))
            self.population = next_pop[:self.pop_size]

        return best_genome, best_fitness



def genetic_algorithm(start, goal, walls, population_size, generations):
    """
    Wrapper to compute a path from start to goal using A* on the occupancy grid.
    """
    # Convert to grid coordinates (row, col)
    sr = int(start[1] // MAP_RESOLUTION)
    sc = int(start[0] // MAP_RESOLUTION)
    gr = int(goal[1] // MAP_RESOLUTION)
    gc = int(goal[0] // MAP_RESOLUTION)
    path, found = a_star((sr, sc), (gr, gc), walls, MAP_RESOLUTION)
    if not found:
        return None
    # Convert back to world coordinates (x,y)
    return [((c + 0.5) * MAP_RESOLUTION, (r + 0.5) * MAP_RESOLUTION) for r, c in path]


def limit_waypoints(path, max_points=20):
    """Keep at most max_points, evenly sampled (always include start & end)."""
    n = len(path)
    if n <= max_points:
        return path
    step = (n - 1) / float(max_points - 1)
    limited = [ path[0] ]
    for i in range(1, max_points - 1):
        idx = int(round(i * step))
        limited.append(path[idx])
    limited.append(path[-1])
    return limited

# Example usage placeholder:
# def simulate_and_score(genome):
#     # Integrate genome as parameters for navigation controller
#     # Run simulation using self-localization and mapping
#     # Return fitness (e.g., time to reach goal or distance traveled)
#     pass

# if __name__ == "__main__":
#     ga = GeneticAlgorithm(pop_size=50, genome_length=10, eval_function=simulate_and_score)
#     best, fitness = ga.run(generations=100)
#     print("Best fitness:", fitness)
