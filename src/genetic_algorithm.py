# --- START OF FILE genetic_algorithm.py ---

"""This Script will contain the logic for the Genetic Algorithm."""

import random
import copy
import numpy as np
import os
import torch
from tqdm import tqdm

CHECKPOINT_FILENAME = "ga_checkpoint.pth"

class GeneticAlgorithmHPO():
    def __init__(self, hp_space, ga_config, fitness_function, results_dir):
        self.hp_space = hp_space
        self.ga_config = ga_config
        self.fitness_function = fitness_function
        self.results_dir = results_dir
        self.checkpoint_path = os.path.join(self.results_dir, CHECKPOINT_FILENAME)

        # GA parameters
        self.population_size = self.ga_config["population_size"]
        self.num_generations = self.ga_config["num_generations"]
        self.mutation_rate = self.ga_config["mutation_rate"]
        self.crossover_rate = self.ga_config["crossover_rate"]
        self.elitism_count = self.ga_config["elitism_count"]
        self.tournament_size = self.ga_config["tournament_size"]

        # State tracking
        self.population = []
        self.fitness_history = []
        self.best_individual_overall = None
        self.best_fitness_overall = -1.0
        self.start_generation = 0

    def _initialize_population(self):
        """Initializes the population with random individuals."""
        from .hpo_config import sample_hyperparameters
        self.population = []
        for _ in range(self.population_size):
            individual_hp = sample_hyperparameters(self.hp_space)
            self.population.append({"hp": individual_hp, "fitness": -1.0})
        print(f"Initialized population with {len(self.population)} individuals.")

    def _evaluate_population(self, generation_num):
        """Evaluates the fitness of the entire population."""
        print(f"\n--- Evaluating Population for Generation {generation_num + 1} ---")
        # generation_num is 0-indexed, so we pass `generation_num + 1` for display
        for i, individual in enumerate(tqdm(self.population, desc=f"Gen {generation_num + 1} Eval")):
            if individual["fitness"] == -1.0: # Only evaluate if fitness is not calculated
                fitness_score = self.fitness_function(individual["hp"], generation_num + 1, i + 1)
                individual["fitness"] = fitness_score

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
        """Performs dictionary-based crossover."""
        child1_hp = copy.deepcopy(parent1_hp)
        child2_hp = copy.deepcopy(parent2_hp)

        if random.random() < self.crossover_rate:
            keys = list(self.hp_space.keys())
            num_keys_to_swap = random.randint(1, max(1, len(keys) // 2))
            keys_to_swap = random.sample(keys, num_keys_to_swap)

            for key in keys_to_swap:
                child1_hp[key], child2_hp[key] = parent2_hp[key], parent1_hp[key]

        return child1_hp, child2_hp

    def _mutate(self, individual_hp):
        """Mutates an individual's hyperparameters."""
        def _get_random_value(param_name, spec):
            if spec["type"] == "choices":
                return random.choice(spec["values"])
            elif spec["type"] == "range":
                low, high = spec["bounds"]
                step = spec.get("step")
                if isinstance(low, int) and isinstance(high, int) and (step is None or isinstance(step, int)):
                    if step: return random.choice(np.arange(low, high + step, step))
                    else: return random.randint(low, high)
                else:
                    if step:
                        num_steps = int(round((high - low) / step))
                        val = low + random.randint(0, num_steps) * step
                        return round(val, len(str(step).split(".")[-1]) if "." in str(step) else 0)
                    else: return random.uniform(low, high)
            elif spec["type"] == "log_range":
                log_low, log_high = spec["bounds"]
                return 10 ** random.uniform(log_low, log_high)
            return None

        mutated_hp = copy.deepcopy(individual_hp)
        for param_name, spec in self.hp_space.items():
            if random.random() < self.mutation_rate:
                mutated_hp[param_name] = _get_random_value(param_name, spec)

        if 'batch_size' in mutated_hp:
             mutated_hp['batch_size'] = int(mutated_hp['batch_size'])
        return mutated_hp

    def _save_checkpoint(self, generation_num):
        """Saves the current state of the GA."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        checkpoint = {
            'population': self.population,
            'best_individual_overall': self.best_individual_overall,
            'best_fitness_overall': self.best_fitness_overall,
            'fitness_history': self.fitness_history,
            'start_generation': generation_num + 1, # Next generation to run
            'random_state_python': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_seed': torch.get_rng_state()
        }
        try:
            torch.save(checkpoint, self.checkpoint_path)
            print(f"Checkpoint saved to {self.checkpoint_path} after generation {generation_num + 1}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def _load_checkpoint(self):
        """Loads the GA state from a checkpoint."""
        if os.path.exists(self.checkpoint_path):
            try:
                print(f"Loading checkpoint from {self.checkpoint_path}")
                # Load to CPU first to avoid GPU memory issues with old checkpoints
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
                
                self.population = checkpoint['population']
                self.best_individual_overall = checkpoint['best_individual_overall']
                self.best_fitness_overall = checkpoint['best_fitness_overall']
                self.fitness_history = checkpoint['fitness_history']
                self.start_generation = checkpoint['start_generation']

                random.setstate(checkpoint['random_state_python'])
                np.random.set_state(checkpoint['numpy_random_state'])
                torch.set_rng_state(checkpoint['torch_random_seed'])

                print(f"Checkpoint loaded. Resuming from generation {self.start_generation + 1}.")
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting fresh.")
                return False
        return False

    def run(self):
        """Runs the genetic algorithm for HPO."""
        if not self._load_checkpoint():
            print("Starting Genetic Algorithm from scratch...")
            self._initialize_population()
            # The first generation is generation_num=0.
            self._evaluate_population(generation_num=0)

            # Initialize overall best from the first population
            if not self.population:
                print("ERROR: Population is empty after initial evaluation. Exiting.")
                return None, -1.0, []
            
            self.best_individual_overall = copy.deepcopy(self.population[0])
            self.best_fitness_overall = self.population[0]["fitness"]
            self.fitness_history.append(self.best_fitness_overall)
            
            print(f"Initial Best Fitness (Gen 1): {self.best_fitness_overall:.4f}")
            self._save_checkpoint(generation_num=0)
            self.start_generation = 1 

        for gen_num in range(self.start_generation, self.num_generations):
            print(f"\n===== Starting Generation {gen_num + 1}/{self.num_generations} =====")

            parents = self._selection()
            next_population = []

            # Elitism
            if self.elitism_count > 0:
                elites = copy.deepcopy(self.population[:self.elitism_count])
                next_population.extend(elites)

            # Crossover and Mutation
            num_offspring_needed = self.population_size - len(next_population)
            offspring = []
            while len(offspring) < num_offspring_needed:
                if len(parents) < 2:
                    p1_container, p2_container = random.choice(parents), random.choice(parents)
                else:
                    p1_container, p2_container = random.sample(parents, 2)
                
                child1_hp, child2_hp = self._crossover(p1_container["hp"], p2_container["hp"])
                
                child1_hp = self._mutate(child1_hp)
                child2_hp = self._mutate(child2_hp)

                offspring.append({"hp": child1_hp, "fitness": -1.0})
                if len(offspring) < num_offspring_needed:
                    offspring.append({"hp": child2_hp, "fitness": -1.0})

            next_population.extend(offspring)
            self.population = next_population[:self.population_size]

            # Evaluate the new generation
            self._evaluate_population(generation_num=gen_num)

            # Update overall best and history
            current_best_in_gen = self.population[0]
            if current_best_in_gen["fitness"] > self.best_fitness_overall:
                self.best_fitness_overall = current_best_in_gen["fitness"]
                self.best_individual_overall = copy.deepcopy(current_best_in_gen)
                print(f"** New overall best fitness found: {self.best_fitness_overall:.4f} **")

            # Update fitness history for this generation's best
            if len(self.fitness_history) > gen_num:
                self.fitness_history[gen_num] = self.population[0]['fitness']
            else:
                self.fitness_history.append(self.population[0]['fitness'])

            print(f"Generation {gen_num + 1} - Best Fitness: {self.population[0]['fitness']:.4f}")
            print(f"Overall Best Fitness So Far: {self.best_fitness_overall:.4f}")
            
            self._save_checkpoint(generation_num=gen_num)

        print("\n--- Genetic Algorithm HPO Finished ---")
        if self.best_individual_overall:
            print(f"Best hyperparameters found: {self.best_individual_overall['hp']}")
            print(f"Best validation accuracy: {self.best_fitness_overall:.4f}")
        else:
            print("No best individual found.")

        return self.best_individual_overall, self.best_fitness_overall, self.fitness_history

if __name__ == '__main__':
    import sys
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, PROJECT_ROOT)

    from src.hpo_config import HYPERPARAMETER_SPACE, FINAL_TRAINING_CONFIG, FITNESS_TRACKING_CONFIG
    from src.train_evaluate import train_and_evaluate_individual

    print("Testing GeneticAlgorithmHPO class with checkpointing...")

    # A small config for quick testing
    test_ga_config = {
        "population_size": 4,
        "num_generations": 3,
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
        "elitism_count": 1,
        "tournament_size": 2,
    }

    test_results_dir = os.path.join(FINAL_TRAINING_CONFIG["results_dir"], "ga_test_checkpoint")
    os.makedirs(test_results_dir, exist_ok=True)

    old_checkpoint_path = os.path.join(test_results_dir, CHECKPOINT_FILENAME)
    if os.path.exists(old_checkpoint_path):
        print(f"Deleting old test checkpoint: {old_checkpoint_path}")
        os.remove(old_checkpoint_path)

    original_fitness_epochs = FITNESS_TRACKING_CONFIG["num_epochs"]
    FITNESS_TRACKING_CONFIG["num_epochs"] = 1
    print(f"Temporarily setting num_epochs for fitness to: {FITNESS_TRACKING_CONFIG['num_epochs']}")

    print("\n--- RUN 1: Start GA from scratch ---")
    ga_hpo_run1 = GeneticAlgorithmHPO(
        hp_space=HYPERPARAMETER_SPACE,
        ga_config=test_ga_config,
        fitness_function=train_and_evaluate_individual,
        results_dir=test_results_dir
    )
    best_ind_run1, best_fit_run1, fit_hist_run1 = ga_hpo_run1.run()

    if best_ind_run1:
        print("\n--- Test Run 1 Summary ---")
        print(f"Best HP: {best_ind_run1['hp']}")
        print(f"Best Fitness: {best_fit_run1:.4f}")
        print(f"Fitness History: {fit_hist_run1}")

    print("\n--- RUN 2: Create new GA object (should load from checkpoint) ---")
    ga_hpo_run2 = GeneticAlgorithmHPO(
        hp_space=HYPERPARAMETER_SPACE,
        ga_config=test_ga_config,
        fitness_function=train_and_evaluate_individual,
        results_dir=test_results_dir
    )
    best_ind_run2, best_fit_run2, fit_hist_run2 = ga_hpo_run2.run()

    if best_ind_run2:
        print("\n--- Test Run 2 (Resume Attempt) Summary ---")
        print(f"Best HP: {best_ind_run2['hp']}")
        print(f"Best Fitness: {best_fit_run2:.4f}")
        print(f"Fitness History: {fit_hist_run2}")
        assert len(fit_hist_run1) == len(fit_hist_run2), "History length should be the same on resume!"

    # Restore the original config value
    FITNESS_TRACKING_CONFIG["num_epochs"] = original_fitness_epochs
    print(f"\nRestored num_epochs for fitness to: {FITNESS_TRACKING_CONFIG['num_epochs']}")