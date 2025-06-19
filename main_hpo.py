"""This is the main script for Hyperparameter Optimisation

This script will:
1. Imporrt necessary configurations and classes.
2. Initialise and run the GeneticAlgorithmHPO.
3. Save the results (best HPs, fitness history).
4. Optionally, train a final model using the best HPs found and evaluate it on the test set."""

import torch
import time
import os

from src.data_loader import IMAGE_SIZE, DATASET_MEAN, DATASET_STD
from src.hpo_config import HYPERPARAMETER_SPACE, GA_CONFIG, FITNESS_TRACKING_CONFIG, FINAL_TRAINING_CONFIG
from src.genetic_algorithm import GeneticAlgorithmHPO
from src.train_evaluate import train_and_evaluate_individual        # Fitness function
from src.utils import save_ga_results, plot_fitness_history, train_final_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Starting Hyperparameter Optimisation using Genetic Algorith, on {DEVICE}")
    print("Ensure all configuration in hpo_config.py are set as desired for the full run.")

    # 1. Setup GA
    # GA_CONFIG population_size and num_generations will determine runtime.
    for k, v in GA_CONFIG.items():
        print(f"  {k}: {v}")
    print("\n--- Fitness Evaluation (during GA) Configuration ---")
    for k, v in FITNESS_TRACKING_CONFIG.items():
        print(f"  {k}: {v}")
    
    ga_hpo_runner = GeneticAlgorithmHPO(
        hp_space=HYPERPARAMETER_SPACE,
        ga_config=GA_CONFIG,
        fitness_function=train_and_evaluate_individual      # This is the costly function
    )

    # 2. Run GA 
    start_time = time.time()
    best_individual, best_fitness, fitness_history = ga_hpo_runner.run()
    end_time = time.time()

    ga_duration_minutes = (end_time - start_time) / 60
    print(f"\nGA HPO run completed in {ga_duration_minutes:.2f} minutes.")

    if best_individual is None:
        print("GA did nt find  avalid individual. Existing...")
        return 
    
    # 3. Save GA results
    print("\n--- Saving GA HPO Results ---")
    
    results_dir = FINAL_TRAINING_CONFIG["results_dir"]
    save_ga_results(
        best_individual_hp=best_individual["hp"],       # pass the HP dict directly
        best_fitness=best_fitness,
        fitness_history=fitness_history,            # plot_fitness_history is called within save_ga_results
        config=FINAL_TRAINING_CONFIG,           # For log name etc
        results_dir=results_dir
    )

    # 4. Train Final Model with Best Hyperparameters
    print("\n--- Proceeding to Final Model Training ---")
    print(f"\n--- Final Model Training Configuration ---")
    for k, v in FINAL_TRAINING_CONFIG.items():
        print(f"  {k}: {v}")

    final_val_acc, final_test_acc = train_final_model(
        best_hp=best_individual["hp"],      # Pass the HP dict
        final_config=FINAL_TRAINING_CONFIG,
        device=DEVICE
    )

    print("\n--- HPO Project Summary ---")
    print(f"Best Hyperparameters found by GA: {best_individual["hp"]}")
    print(f"Best Validation Accuracy from GA (short Training): {best_fitness:.4f}")
    print(f"Final model Validation Accuracy (longer training): {final_test_acc:.4f}")
    print(f"Final model test accuracy: {final_test_acc:.4f}")
    print(f"All results, logs, and plots are saved in {FINAL_TRAINING_CONFIG["results_dir"]}")

if __name__ == "__main__":
    # Creating a results directory if it dosen't exist
    if not os.path.exists(FINAL_TRAINING_CONFIG["results_dir"]):
        os.makedirs(FINAL_TRAINING_CONFIG["results_dir"])
        print(f"Created results directory: {FINAL_TRAINING_CONFIG["results_dir"]}")

    # Create plots subdirectory if it dosen't exists
    plots_dir = os.path.join(FINAL_TRAINING_CONFIG["results_dir"], "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")
    
    main()

