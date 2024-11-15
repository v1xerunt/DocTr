import numpy as np
import random
import math
from utils import max_sim
random.seed(0)

def calculate_entropy(ratios):
    entropy = 0.0
    
    for ratio in ratios:
        if ratio > 0:
            entropy -= ratio * math.log2(ratio)
    
    return entropy

def calculate_fairness_score(locations):
    # N * F
    num_locations = len(locations)
    
    location_entropies = []
    
    for location in locations:
        entropy = calculate_entropy(location)
        location_entropies.append(entropy)
        
    avg_entropy = sum(location_entropies) / num_locations
    max_entropy = math.log2(len(locations[0]))
    
    fairness_score = avg_entropy / max_entropy
    
    return fairness_score

def calculate_geo_entropy(selection):
    geo_set = list(set(selection))
    total_groups = len(geo_set)
    counts = np.zeros(total_groups)
    for group in selection:
        cur_idx = geo_set.index(group)
        counts[cur_idx] += 1
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(p * np.log(p) for p in probabilities if p > 0)
    
    max_entropy = math.log2(total_groups)
    entropy = entropy / max_entropy
    
    return entropy

def fitness(solution):
    total_score = sum(item[0] for item in solution)
    gender_entropy = calculate_fairness_score([item[1] for item in solution])
    race_entropy = calculate_fairness_score([item[2] for item in solution])
    ethnicity_entropy = calculate_fairness_score([item[3] for item in solution])
    geo_entropy = calculate_geo_entropy([item[4] for item in solution])
    competiting_score = sum([item[5] for item in solution])
    cur_score = total_score + 10*(gender_entropy + race_entropy + ethnicity_entropy + geo_entropy) - 0.1 * competiting_score
    return cur_score if cur_score > 0 else 0   # You can adjust the weights if needed

def create_initial_population(items, population_size, K):
    return [random.sample(items, K) for _ in range(population_size)]

def select_parents(population):
    fitness_scores = [fitness(sol) for sol in population]
    total_fitness = sum(fitness_scores)
    selection_probs = [f / total_fitness for f in fitness_scores]
    selected_indices = np.random.choice(range(len(population)), size=2, replace=False, p=selection_probs)
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2, K):
    crossover_point = random.randint(1, K - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(solution, items, mutation_rate):
    if random.random() < mutation_rate:
        mutate_index = random.randint(0, len(solution) - 1)
        solution[mutate_index] = random.choice(items)
    return solution

def genetic_algorithm(items, K, population_size=50, generations=100, mutation_rate=0.01):
    population = create_initial_population(items, population_size, K)

    for _ in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2, K)
            new_population.extend([mutate(child1, items, mutation_rate), mutate(child2, items, mutation_rate)])
        population = new_population

    best_solution = max(population, key=fitness)
    return best_solution

def calc_metrics(best_solution, scores):
    solution_idx = [item[-1] for item in best_solution]
    total_score = np.mean([scores[idx] for idx in solution_idx])
    avg_gender_entropy = calculate_fairness_score([item[1] for item in best_solution])
    avg_race_entropy = calculate_fairness_score([item[2] for item in best_solution])
    avg_ethnicity_entropy = calculate_fairness_score([item[3] for item in best_solution])
    avg_geo_entropy = calculate_geo_entropy([item[4] for item in best_solution])
    avg_competiting_score = np.mean([item[5] for item in best_solution])
    return total_score, avg_gender_entropy, avg_race_entropy, avg_ethnicity_entropy, avg_geo_entropy, avg_competiting_score