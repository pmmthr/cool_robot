import math
import random
import heapq
import numpy as np
from constants import MAP_RESOLUTION, WIDTH, HEIGHT
from utils import a_star

class GeneticAlgorithm:
    """
    Generic Genetic Algorithm for evolving controller parameters
    """
    def __init__(self, pop_size, genome_length, eval_function,
                 crossover_rate=0.8, mutation_rate=0.1, mutation_scale=0.1):
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.eval_function = eval_function
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        # Initialize population: random genomes in grid coords
        # genome: sequence of (row, col) tuples
        self.population = [self._random_genome() for _ in range(pop_size)]
        self.fitness = [float('inf')] * pop_size

    def _random_genome(self):
        # uniform random waypoints in grid space
        max_r = HEIGHT // MAP_RESOLUTION - 1
        max_c = WIDTH  // MAP_RESOLUTION - 1
        return [(random.randint(0, max_r), random.randint(0, max_c))
                for _ in range(self.genome_length)]

    def evaluate(self):
        for i, genome in enumerate(self.population):
            self.fitness[i] = self.eval_function(genome)

    def select_parents(self):
        # tournament selection
        parents = []
        for _ in range(self.pop_size):
            i, j = random.sample(range(self.pop_size), 2)
            parents.append(self.population[i] if self.fitness[i] < self.fitness[j] else self.population[j])
        return parents

    def crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            pt = random.randint(1, self.genome_length - 1)
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1.copy(), p2.copy()

    def mutate(self, genome):
        for idx in range(len(genome)):
            if random.random() < self.mutation_rate:
                r, c = genome[idx]
                genome[idx] = (min(max(r + random.choice([-1,1]), 0), HEIGHT//MAP_RESOLUTION -1),
                               min(max(c + random.choice([-1,1]), 0), WIDTH //MAP_RESOLUTION -1))
        return genome

    def run(self, generations):
        best_genome, best_fit = None, float('inf')
        for gen in range(generations):
            # print generation count for debugging
            print(f"Generation {gen}")
            self.evaluate()
            # record best
            idx = min(range(self.pop_size), key=lambda i: self.fitness[i])
            if self.fitness[idx] < best_fit:
                best_fit = self.fitness[idx]
                best_genome = list(self.population[idx])
            # create next generation
            parents = self.select_parents()
            new_pop = []
            for i in range(0, self.pop_size, 2):
                o1, o2 = parents[i], parents[(i+1)%self.pop_size]
                c1, c2 = self.crossover(o1, o2)
                new_pop.extend([self.mutate(c1), self.mutate(c2)])
            self.population = new_pop[:self.pop_size]
        return best_genome, best_fit


def genome_to_path(genome, start, goal, walls):
    # stitch A* between start -> each genome point -> goal
    points = [start] + [((c+0.5)*MAP_RESOLUTION, (r+0.5)*MAP_RESOLUTION) for r,c in genome] + [goal]
    path = []
    for a, b in zip(points, points[1:]):
        sr, sc = int(a[1]//MAP_RESOLUTION), int(a[0]//MAP_RESOLUTION)
        gr, gc = int(b[1]//MAP_RESOLUTION), int(b[0]//MAP_RESOLUTION)
        seg, found = a_star((sr, sc), (gr, gc), walls, MAP_RESOLUTION)
        if not found:
            return None
        path.extend([((c+0.5)*MAP_RESOLUTION, (r+0.5)*MAP_RESOLUTION) for r,c in seg])
    return path


def fitness_function(genome, start, goal, walls):
    path = genome_to_path(genome, start, goal, walls)
    if path is None:
        return float('inf')
    # path length
    L = sum(math.hypot(x2-x1, y2-y1) for (x1,y1),(x2,y2) in zip(path, path[1:]))
    # heuristic to goal
    dx, dy = goal[0]-path[-1][0], goal[1]-path[-1][1]
    return L + abs(dx)+abs(dy)


def genetic_algorithm(start, goal, walls, population_size=50, generations=100, genome_length=5):
    # wrap fitness
    def eval_fn(genome):
        return fitness_function(genome, start, goal, walls)

    ga = GeneticAlgorithm(population_size, genome_length, eval_fn)
    best_genome, best_fit = ga.run(generations)
    if best_genome is None or best_fit==float('inf'):
        return None
    return genome_to_path(best_genome, start, goal, walls)


def limit_waypoints(path, max_points=20):
    if not path:
        return []
    n = len(path)
    if n <= max_points:
        return path
    step = (n-1)/(max_points-1)
    limited = [path[0]]
    for i in range(1, max_points-1):
        limited.append(path[int(round(i*step))])
    limited.append(path[-1])
    return limited
