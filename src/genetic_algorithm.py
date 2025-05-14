import random
from tqdm import tqdm

def genetic_algorithm(start, goal, maze_walls, population_size=50, generations=100, mutation_rate=0.1):
    def fitness(path):
        if not path or any(wall.collidepoint(step[0], step[1]) for step in path for wall in maze_walls):
            return float('inf')  # Invalid path
        return len(path) + heuristic(path[-1], goal)

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    def mutate(path):
        if random.random() < mutation_rate:
            idx = random.randint(0, len(path) - 1)
            new_step = (path[idx][0] + random.choice([-1, 1]), path[idx][1] + random.choice([-1, 1]))
            if new_step not in maze_walls:  #  mutation doesn't create invalid steps
                path[idx] = new_step
        return path

    def crossover(parent1, parent2):
        split = random.randint(0, len(parent1) - 1)
        child = parent1[:split] + parent2[split:]
        # make sure child path doesn't pass through walls
        child = [step for step in child if not any(wall.collidepoint(step[0], step[1]) for wall in maze_walls)]
        return child if child else parent1  # just in case child is empty

    population = [[start] for _ in range(population_size)]

    for _ in tqdm(range(generations)):
        population = sorted(population, key=fitness)
        new_population = population[:population_size // 2]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(new_population, 2)
            child = mutate(crossover(parent1, parent2))
            new_population.append(child)

        population = new_population

        print(f"Best path length: {fitness(population[0])}")

    best_path = min(population, key=fitness)
    return best_path if fitness(best_path) < float('inf') else None