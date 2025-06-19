"""This Script will contain the logic for the Genetic Algorithm. It will do the following:
1. A dictionary of hyperparameters.
2. A list of these individuals.
3. Fitness functoon: **train_and_evaluate_individual** from train_evaluate.py
4. Tournament selection is a common and effective choice.
5. Single point or two point crossover for dictionaries.
6. Randomly changing hyperparameter values within their defined ranges/choices.
"""


import random
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.hpo_config import HYPERPARAMETER_SPACE, GA_CONFIG, sample_hyperparameters, FITNESS_TRACKING_CONFIG
from src.train_evaluate import train_and_evaluate_individual

class GeneticAlgorithmHPO():
    def __init__(self, hp_space, ga_config, fitness_function):
        self.hp_space = hp_space
        self.ga_config = ga_config
        self.fitness_function = fitness_function    # train_and_evaluate_individual

        self.population_size = self.ga_config["population_size"]
        self.num_generations = self.ga_config["num_generations"]
        self.mutation_rate = self.ga_config["mutation_rate"]
        self.crossover_rate = self.ga_config["crossover_rate"]
        self.elitism_count = self.ga_config["elitism_count"]
        self.tournament_size = self.ga_config["tournament_size"]

        self.population = []
        self.fitness_history = []       # Store best fitness per geneation
        self.best_individual_overall = None
        self.best_individual_overall = -1.0
    
    def _initialize_population(self):
        """Initializes the population with random individuals"""
        self.population = []
        for _ in range(self.population_size):
            individual = sample_hyperparameters(self.hp_space)
            self.population.append({"hp": individual, "fitness": -1.0})
        print(f"Initialised population with {len(self.population)} individuals.")
    
    def _calculate_fitness(self, individual, generation_num, individual_num):
        """Calculates (or retrieves cached) fitness for an individual"""
        fitness_score = self.fitness_function(individual["hp"], generation_num, individual_num)
        individual["fitness"] = fitness_score

        return fitness_score
    
    def _evaluate_population(self, generation_num):
        """Evaluates the fitness of the entire population"""
        print(f"\n-- Evaluating Population for Generation {generation_num + 1} ---")
        for i, individual in enumerate(tqdm(self.population, desc="Evaluating Population")):
            if individual["fitness"] == -1.0:
                self._calculate_fitness(individual, generation_num, i)

        self.population.sort(key=lambda ind: ind["fitness"], reverse=True)
    
    def _selection(self):
        """Performs tournament selection to choose parents."""
        selected_parents = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, self.tournament_size)
            tournament.sort(key=lambda ind: ind["fitness"], reverse=True)
            selected_parents.append(tournament[0])
        return selected_parents
    
    def _crossover(self, parent1_hp, parent2_hp):
        """Performs dictionary-based crossover"""
        child1_hp = copy.deepcopy(parent1_hp)
        child2_hp = copy.deepcopy(parent2_hp)

        if random.random() < self.crossover_rate:
            keys = list(self.hp_space.keys())
            random.shuffle(keys)        # shuffle keys to make crossover point random regarding parameter type
            # simple crossover: swap values for a subset of keys or choose a random subset of keys to swap

            num_keys_to_swap = random.randint(1, len(keys) // 2)
            keys_to_swap = random.sample(keys, num_keys_to_swap)

            for key in keys_to_swap:
                child1_hp[key], child2_hp[key] = parent2_hp[key], parent1_hp[key]
        return child1_hp, child2_hp

    def _mutate(self, individual_hp):
        """Mutates an individual's hyperparameters."""
        mutated_hp = copy.deepcopy(individual_hp)
        for param_name in self.hp_space.keys():
            if random.random() < self.mutation_rate:
                # Resample this spefific hyperparameter
                spec = self.hp_space[param_name]
                if spec["type"] == "choices":
                    mutated_hp[param_name] = random.choice(spec["values"])
                elif spec["type"] == "range":
                    low, high = spec["bounds"]
                    step = spec.get("step")
                    if isinstance(low, int) and isinstance(high, int) and (step is None or isinstance(step, int)):
                        if step:
                            val = random.choice(np.arange(low, high + step, step))
                        else:
                            val = random.randint(low, high)
                    else:
                        if step:
                            num_steps = int((high - low) / step)
                            val = low + random.randint(0, num_steps) * step
                            val = round(val, len(str(step).split(".")[-1]) if "." in str(step) else 0)
                        else:
                            val = random.uniform(low, high)
                    mutated_hp[param_name] = val
                elif spec["type"] == "log_range":
                    log_low, log_high = spec["bounds"]
                    mutated_hp[param_name] = 10 ** random.uniform(log_low, log_high)

                if param_name == "batch_size":
                    mutated_hp[param_name] = int(mutated_hp[param_name])
        return mutated_hp

    def run(self):
        """Runs the genetic algorithm for HPO"""    
        print("Starting Genetic Algorithm for Hyperparameter Optimization...")
        self._initialize_population()

        # Evaluate initial population
        self._evaluate_population(generation_num=-1)    # -1 for initial population evaluation pass
        if not self.population:
            print("Population is empty after initialization or evaluation. Existing.")
            return None, -1.0
        
        self.best_individual_overall = copy.deepcopy(self.population[0])
        self.best_fitness_overall = self.population[0]["fitness"]
        self.fitness_history.append(self.best_fitness_overall)
        
        print(f"Initial Best Fitness: {self.best_fitness_overall:.4f} with HP: {self.best_individual_overall["hp"]}")

        for gen in range(self.num_generations):
            print(f"\n=== Generation {gen + 1}/{self.num_generations} ===")

            # Selection 
            parents = self._selection()
            next_population = []

            # Elitism: Carry over the best individuals
            if self.elitism_count > 0:
                elites = copy.deepcopy(self.population[:self.elitism_count])
                next_population.extend(elites)
            
            # Crossover and mutation to fill the rest of the population
            num_offspring_needed = self.population_size - len(next_population)

            current_offspring = []
            while len(current_offspring) < num_offspring_needed:
                p1_container, p2_container = random.sample(parents, 2)  # get parent containers
                child1_hp, child2_hp = self._crossover(p1_container["hp"], p2_container["hp"])

                child1_hp = self._mutate(child1_hp)
                child2_hp = self._mutate(child2_hp)

                current_offspring.append({"hp": child1_hp, "fitness": -1.0})    # fitness to be evaluated
                if len(current_offspring) < num_offspring_needed:
                    current_offspring.append({"hp": child2_hp, "fitness": -1.0})
            
            next_population.extend(current_offspring)
            self.population = next_population[:self.population_size]    
            self._evaluate_population(generation_num=gen)

            # Update overall best
            current_best_fitness_in_gen = self.population[0]["fitness"]
            if current_best_fitness_in_gen > self.best_fitness_overall:
                self.best_fitness_overall = current_best_fitness_in_gen
                self.best_individual_overall = copy.deepcopy(self.population[0])
           
            self.fitness_history.append(self.population[0]["fitness"])  # stores best fitness of current gen
            print(f"Generation {gen + 1} - Best Fitness: {self.population[0]["fitness"]:.4f}")
            print(f"Overall Best HP: {self.best_individual_overall["hp"]}")
        
        print("\n--- Genetic Algorithm HPO Finished ---")
        print(f"Best Hyperparameters Found: {self.best_individual_overall["hp"]}")
        print(f"Best Validation Accuracy: {self.best_fitness_overall:.4f}")

        return self.best_individual_overall, self.best_fitness_overall, self.fitness_history
    
if __name__ == "__main__":
    print("Testing GeneticAlgorithmHPO class...")

    # Use a smaller GA config for quick testing
    test_ga_config = {
        "population_size": 4,
        "num_generations": 2,
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
        "elitism_count": 1,
        "tournament_size": 2,
    }

    original_fitness_epochs = FITNESS_TRACKING_CONFIG["num_epochs"]
    FITNESS_TRACKING_CONFIG["num_epochs"] = 1
    print(f"Temporarily setting num_epocs for fitness to : {FITNESS_TRACKING_CONFIG["num_epochs"]}")

    ga_hpo = GeneticAlgorithmHPO(
        hp_space=HYPERPARAMETER_SPACE,
        ga_config=test_ga_config,       # use test_ga_config
        fitness_function=train_and_evaluate_individual
    )
    best_ind, best_fit, fit_history = ga_hpo.run()

    if best_ind:
        print("\n--- Test Run SUmmary ---")
        print(f"Best Individual's HPs from test run: {best_ind["hp"]}")
        print(f"Best Fitness from test run: {best_fit:.4f}")
        print(f"Fitness History (best per gen): {fit_history}")
    else:
        print("GA run failed or produced no results.")
    
    FITNESS_TRACKING_CONFIG["num_epochs"] = original_fitness_epochs
    print(f"Restored num_epochs for fitness to: {FITNESS_TRACKING_CONFIG["num_epochs"]}")



