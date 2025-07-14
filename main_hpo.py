import torch
import time
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.hpo_config import HYPERPARAMETER_SPACE, GA_CONFIG, FITNESS_TRACKING_CONFIG, FINAL_TRAINING_CONFIG
from src.genetic_algorithm import GeneticAlgorithmHPO
from src.train_evaluate import train_and_evaluate_individual
from src.utils import save_ga_results, train_final_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Starting Hyperparameter Optimization using Genetic Algorithm on {DEVICE}")
    print("Ensure all configurations in hpo_config.py are set as desired.")

    results_dir = FINAL_TRAINING_CONFIG["results_dir"]
    plots_dir = os.path.join(results_dir, "plots")
    data_dir = FINAL_TRAINING_CONFIG["data_dir"]

    if not os.path.exists(data_dir):
        print(f"FATAL: Data directory not found at '{data_dir}'")
        print("Please check the 'data_dir' path in 'src/hpo_config.py'")
        sys.exit(1)

    os.makedirs(plots_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")


    print("\n--- GA Configuration ---")
    for k, v in GA_CONFIG.items():
        print(f"  {k}: {v}")
    print("\n--- Fitness Evaluation (during GA) Configuration ---")
    for k, v in FITNESS_TRACKING_CONFIG.items():
        print(f"  {k}: {v}")

    ga_hpo_runner = GeneticAlgorithmHPO(
        hp_space=HYPERPARAMETER_SPACE,
        ga_config=GA_CONFIG,
        fitness_function=train_and_evaluate_individual,
        results_dir=results_dir
    )

    print("\n--- Starting/Resuming GA HPO Run ---")
    start_time = time.time()
    best_individual, best_fitness, fitness_history = ga_hpo_runner.run()
    end_time = time.time()

    ga_duration_minutes = (end_time - start_time) / 60
    print(f"\nGA HPO run (or continuation) completed in {ga_duration_minutes:.2f} minutes.")

    if best_individual is None:
        print("GA did not find a valid best individual. Exiting.")
        return

    print("\n--- Saving GA HPO Results ---")
    save_ga_results(
        best_individual_hp=best_individual['hp'],
        best_fitness=best_fitness,
        fitness_history=fitness_history,
        config=FINAL_TRAINING_CONFIG,
        results_dir=results_dir
    )

    # Train Final Model with Best Hyperparameters 
    print("\n--- Proceeding to Final Model Training ---")
    print("\n--- Final Model Training Configuration ---")
    for k, v in FINAL_TRAINING_CONFIG.items():
        print(f"  {k}: {v}")

    final_val_acc, final_test_acc = train_final_model(
        best_hp=best_individual['hp'],
        final_config=FINAL_TRAINING_CONFIG,
        device=DEVICE
    )

    print("\n--- HPO Project Summary ---")
    print(f"Best hyperparameters found by GA: {best_individual['hp']}")
    print(f"Best validation accuracy from GA (short training): {best_fitness:.4f}")
    if final_val_acc is not None and final_val_acc > 0:
        print(f"Final model validation accuracy (longer training): {final_val_acc:.4f}")
        print(f"Final model test accuracy: {final_test_acc:.4f}")
    else:
        print("Final model training was not completed or did not report a valid accuracy.")
    print(f"All results, logs, and plots are saved in: {FINAL_TRAINING_CONFIG['results_dir']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Exiting gracefully.")
        from src.hpo_config import FINAL_TRAINING_CONFIG
        import os
        checkpoint_path = os.path.join(FINAL_TRAINING_CONFIG['results_dir'], "ga_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            print("A checkpoint file was found. You should be able to resume the training.")
        else:
            print("No checkpoint file found. The next run will start from scratch.")