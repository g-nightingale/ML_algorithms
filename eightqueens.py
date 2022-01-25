import time
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt


class GeneticAlgorithmBasic:
    """Applies a genetic algorithm to solve the 8 queens puzzle."""
    
    def __init__(self, n=8, population_size=100, mutate_prob=0.8, bits_to_mutate=1, max_iter=5000,
                 time_limit=30, power_factor=5, verbose_level=0, random_seed=None):
        self.n = n
        self.population_size = population_size
        population = []
        if random_seed is not None:
            self.rng = default_rng(seed=random_seed)
        else:
            self.rng = default_rng()
        for i in range(self.population_size):
            # Ensure elements do not repeat
            population.append(self.rng.choice(n, n, replace=False).astype(np.int8))
        self.population = population
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.power_factor = power_factor
        self.mutate_prob = mutate_prob
        self.bits_to_mutate = bits_to_mutate
        self.costs = np.zeros(self.population_size, dtype=np.int8)
        self.generation_costs = []
        self.verbose_level = verbose_level

    def cost(self, state):
        """Calculates the number of pairs attacking."""
        count = 0
        for i in range(len(state) - 1):
            # for each queen, look in columns to the right
            # add one to the count if there is another queen in the same row
            count += (state[i] == np.array(state[i + 1:], dtype=np.int8)).sum()

            # add one to the count for each queen on the upper or lower diagonal
            upper_diagonal = state[i] + np.arange(1, self.n - i)
            lower_diagonal = state[i] - np.arange(1, self.n - i)
            count += (np.array(state[i + 1:], dtype=np.int8) == upper_diagonal).sum()
            count += (np.array(state[i + 1:], dtype=np.int8) == lower_diagonal).sum()
        # Double the count
        count *= 2
        return count

    def calculate_costs(self, population):
        """Calculate the cost for the population."""
        return np.array([self.cost(p) for p in population], dtype=np.int8)
    
    def fitness(self, costs):
        """Calculates the fitness of all indviduals."""
        return (self.n * (self.n - 1) - costs) / 2

    def fitness_population(self, fitness):
        """Fitness of all individuals expressed as a % of the population."""
        return np.power(fitness, self.power_factor)/np.sum(np.power(fitness, self.power_factor))
    
    def min_cost(self, costs):
        """Get the minimum cost of all the individuals."""
        return min([c for c in costs])
    
    def find_solution(self, costs):
        """Find solution if one exists."""
        for i, c in enumerate(costs):
            if c == 0:
                return i
        return None
        
    def print_population(self):
        """Print population."""
        for p in self.population:
            print(p)
            
    def calculate_time(self, start):
        """Calculates how long the program has been running."""
        return time.perf_counter() - start
            
    def reproduce(self, parent1, parent2):
        """Reproduce given a pair of parents."""
        
        # Create child arrays
        child1 = np.zeros(self.n, dtype=np.int8)
        child2 = np.zeros(self.n, dtype=np.int8)
        
        # Determine random crossover point
        n_x = self.rng.integers(0, self.n)
        
        # Generate the new states 
        child1[:n_x] = parent1[:n_x]
        child1[n_x:] = parent2[n_x:]
        child2[:n_x] = parent2[:n_x]
        child2[n_x:] = parent1[n_x:]
        
        # Mutate
        if self.rng.random() < self.mutate_prob:
            child1 = self.mutate(child1)
            
        if self.rng.random() < self.mutate_prob:
            child2 = self.mutate(child2)
            
        # Print verbosity
        if self.verbose_level in [1]:
            print(f"crossover point: {n_x}")
            print(f"parent1: {parent1}")
            print(f"parent2: {parent2}")
            print(f"child1: {child1}")
            print(f"child2: {child2}")
            print()
            
        return child1, child2
    
    def mutate(self, a):
        """Mutate random bits."""
        idx = self.rng.integers(0, self.n, self.bits_to_mutate)
        bit_values = self.rng.integers(0, self.n, self.bits_to_mutate)
        
        # Generate the new states 
        for i, bit_idx in enumerate(idx):
            a[bit_idx] = bit_values[i]
            
        return a
    
    def genetic_algorithm(self):
        """Genetic algorithm."""
        
        start = time.perf_counter() 
        
        # Initialise costs and fitness
        population = self.population.copy()
        costs = self.calculate_costs(population)
        fitness = self.fitness(costs)
        fitness_population = self.fitness_population(fitness)
        
        # Loop through iterations
        for i in range(self.max_iter):
            new_population = []
            
            # Generate a new population
            for j in range(self.population_size//2):
                p = self.rng.choice(range(self.population_size), size=2, replace=False, p=fitness_population)
                child1, child2 = self.reproduce(population[p[0]], population[p[1]])
                new_population.extend((child1, child2))
                
            population = new_population.copy()
            
            # Recalculate costs and fitness of new generation
            costs = self.calculate_costs(population)
            fitness = self.fitness(costs)
            fitness_population = self.fitness_population(fitness)
            min_cost = self.min_cost(costs)
            self.generation_costs.append(min_cost)
            time_taken = self.calculate_time(start)
            
            if i % 10 == 0 and self.verbose_level in [1, 2]:
                print(f"Generation {i} min cost: {min_cost}")
            
            # Stop if solution found or time limit reached
            if min_cost == 0 or time_taken > self.time_limit:
                self.population = population
                self.costs = costs
                solution_idx = self.find_solution(costs)
                if self.verbose_level in [1, 2, 3]:
                    if solution_idx is None:
                        print(f"No solution found :( Generations: {i}  Time taken: {time_taken}")
                    else:
                        print(f"Solution found! Generations: {i} Time taken: {time_taken}")
                if solution_idx is None:
                    return None, time_taken
                else:
                    return self.population[solution_idx], time_taken
            
        # Catch-all when max iterations reached
        self.population = population
        self.costs = costs
        if self.verbose_level in [1, 2, 3]:
            print(f"No solution found :( Generations: {i}  Time taken: {time_taken}")
        return None, time_taken
    
    def plot_costs(self):
        """Plot the minimum population cost per generation."""
        plt.figure(figsize=(10, 6))
        plt.title("Minimum population cost per generation")
        plt.ylabel("Minimum cost")
        plt.xlabel("Generation")
        plt.plot(self.generation_costs, color='green')
        plt.show()


class GeneticAlgorithmEnhanced(GeneticAlgorithmBasic):
    """Applies a genetic algorithm to solve the 8 queens puzzle."""
    
    def __init__(self, n=8, population_size=100, sample_percent=0.2, mutate_prob=0.8, 
                 bits_to_mutate=1, max_iter=5000, time_limit=30, power_factor=5, 
                 verbose_level=0, random_seed=None):
        super().__init__(n, population_size, mutate_prob, bits_to_mutate, max_iter, 
                         time_limit, power_factor, verbose_level, random_seed)
        self.sample_percent = sample_percent
        
    def genetic_algorithm(self):
        """Genetic algorithm."""
        
        start = time.perf_counter() 
        sample_n = int(self.population_size * self.sample_percent)
        
        # Initialise costs and fitness
        population = self.population.copy()
        costs = self.calculate_costs(population)
        fitness = self.fitness(costs)
        
        # Loop through iterations
        for i in range(self.max_iter):
            
            new_population = []

            # Sort individuals by fitness and create samples
            sorted_idx = np.argsort(-fitness)[:sample_n]
            sample_population = [population[i] for i in sorted_idx]
            sample_fitness = [fitness[i] for i in sorted_idx]
            fitness_sample = self.fitness_population(sample_fitness)

            # Generate children from sample of parents
            for j in range(int((1 - self.sample_percent) * self.population_size) // 2):
                p = self.rng.choice(range(len(sample_population)), size=2, replace=False, p=fitness_sample)
                child1, child2 = self.reproduce(sample_population[p[0]], sample_population[p[1]])
                new_population.extend((child1, child2))
                
            new_population.extend(sample_population)
            population = new_population.copy()
            
            # Recalculate costs and fitness of new generation
            costs = self.calculate_costs(population)
            fitness = self.fitness(costs)
            min_cost = self.min_cost(costs)
            self.generation_costs.append(min_cost)
            time_taken = self.calculate_time(start)
            
            if i % 10 == 0 and self.verbose_level in [1, 2]:
                print(f"Generation {i} min cost: {min_cost}")
            
            # Stop if solution found or time limit reached
            if min_cost == 0 or time_taken > self.time_limit:
                self.population = population
                self.costs = costs
                solution_idx = self.find_solution(costs)  
                if self.verbose_level in [1, 2, 3]:
                    if solution_idx is None:
                        print(f"No solution found :( Generations: {i}  Time taken: {time_taken}")
                    else:
                        print(f"Solution found! Generations: {i} Time taken: {time_taken}")
                if solution_idx is None:
                    return None, time_taken
                else:
                    return self.population[solution_idx], time_taken
            
        # Catch-all when max iterations reached
        self.population = population
        self.costs = costs 
        if self.verbose_level in [1, 2, 3]:
            print(f"No solution found :( Generations: {i}  Time taken: {time_taken}")
        return None, time_taken
    