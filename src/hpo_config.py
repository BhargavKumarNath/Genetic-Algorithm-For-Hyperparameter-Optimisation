# This file will define the search space for hyperparameters and GA settings
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR_RELATIVE = "data/blood_cell_images/"
RESULTS_DIR_RELATIVE = "results/"

# Data Constants
DATASET_MEAN = [0.8742497563362122, 0.7491484880447388, 0.7213816046714783]
DATASET_STD = [0.15781086683273315, 0.1840810924768448, 0.07911773025989532]
IMAGE_SIZE = (224, 224)
RANDOM_SEED = 46

HYPERPARAMETER_SPACE = {
    "lr": {"type": "log_range", "bounds": [-4, -2]},    
    "batch_size": {"type": "choices", "values": [16, 32, 64]},
    "optimizer": {"type": "choices", "values": ["adam", "sgd", "rmsprop"]},

    # Concolutional Layers
    "num_conv_blocks": {"type": "choices", "values": [2, 3, 4]},    # Number of Conv -> Act -> Pool -> Dropout blocks
    "conv_filters_start": {"type": "choices", "values": [16, 32]},  # Num filters in the first conv layer
    # Filters will double in subsequent blocks
    "kernel_size": {"type": "choices", "values": [3, 5]},
    "conv_activation": {"type": "choices", "values": ["relu", "leaky_relu", "elu"]},
    "conv_dropout_rate": {"type": "range", "bounds": [0.0, 0.4], "step": 0.1},      # Dropout after pooling

    # Fully Connected Layers
    "num_fc_layers": {"type": "choices", "values": [1, 2]},     # Number of FC layers after conv blocks
    "fc_neurons_start": {"type": "choices", "values": [128, 256, 512]},  # Neurons in the first FC layer
    # Neurons will have in subsequent FC layers if num_fc_layers > 1
    "fc_activation": {"type": "choices", "values":["relu", "leaky_relu", "elu"]},
    "fc_dropout_rate": {"type": "range", "bounds": [0.0, 0.5], "step": 0.1},
}


# --- GENETIC ALGORITHM PARAMETERS ---
GA_CONFIG = {
    "population_size": 10,      # Number of individuals in the population - 10
    "num_generations": 10,      # Number of generations to evolve - 10
    "mutation_rate": 0.15,      # Probability of mutating a gene
    "crossover_rate": 0.7,      # Probability of performing crossover
    "elitism_count": 2,         # number of best individuals to carry over to the next generation
    "tournament_size": 3,       # Size of the tournament for selectiob
}


# Training Parameters for Fitness Evaluation
# Each individual (hyperparameter set) will be trained for a short duration.

FITNESS_TRACKING_CONFIG = {
    "num_epochs": 10,        # Number of epochs to train CNN for fitness evaluation. We keep this low to spee dup HPO - 10
    "patience": 3,          # Early stopping patience during ditness evaluation
    "data_dir": os.path.join(PROJECT_ROOT, DATA_DIR_RELATIVE),     # data dir
    "num_classes": 8,       # from EDA
    "image_size": IMAGE_SIZE,   # From data_loader
    "dataset_mean": DATASET_MEAN,   # From data_loader
    "dataset_std": DATASET_STD, # From data_loader
}


# FINAL MODEL TRAINING PARAMETERS (after HPO)
# Once the best hyperparameters are found, train the final model for longer
FINAL_TRAINING_CONFIG = {
    "num_epochs": 50,   # 50
    "patience": 10,
    "data_dir": FITNESS_TRACKING_CONFIG["data_dir"],
    "num_classes": FITNESS_TRACKING_CONFIG["num_classes"],
    "image_size": FITNESS_TRACKING_CONFIG["image_size"],
    "dataset_mean": FITNESS_TRACKING_CONFIG["dataset_mean"],
    "dataset_std": FITNESS_TRACKING_CONFIG["dataset_std"],
    "results_dir": os.path.join(PROJECT_ROOT, RESULTS_DIR_RELATIVE),
    "best_model_name": "best_cnn_model.pth",
    "ga_log_name": "ga_hpo_log.csv"
}


# Helper function to sample a random individual
def sample_hyperparameters(space):
    """Samples random set of hyperparameters from the defined space"""
    individual = {}
    for param, spec in space.items():
        if spec["type"] == "choices":
            individual[param] = np.random.choice(spec["values"])
        elif spec["type"] == "range":
            low, high = spec["bounds"]
            step = spec.get("step")
            if isinstance(low, int) and isinstance(high, int) and (step is None or isinstance(step, int)):
                # integer range
                if step:
                    val = np.random.choice(np.arange(low, high + step, step))
                else:
                    val = np.random.randint(low, high + 1)
            else:
                # Float range
                if step:
                    num_steps = int(round((high - low) / step))
                    val = low + np.random.randint(0, num_steps + 1) * step
                    val = round(val, len(str(step).split(".")[-1]) if "." in str(step) else 0)  # preserve precision of step
                else:
                    # Continous step
                    val = np.random.uniform(low, high)
            individual[param] = val
        elif spec["type"] == "log_range":
            # For log uniform sampling (eg. learning rate)
            log_low, log_high = spec["bounds"]
            val = 10 ** np.random.uniform(log_low, log_high)
            individual[param] = val
    return individual

if __name__ == "__main__":
    print("--- Project Configuration ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {FINAL_TRAINING_CONFIG["data_dir"]}")
    print(f"Results Directory: {FINAL_TRAINING_CONFIG["results_dir"]}")

    
    print("--- Hyperparameter Space ---")
    for k, v in HYPERPARAMETER_SPACE.items():
        print(f"{k}: {v}")

    print(f"\n--- GA Configuration ---")
    for k, v in GA_CONFIG.items():
        print(f"{k}: {v}")
    
    print("\n--- Fitness Training Confugiration ---")
    for k, v in FITNESS_TRACKING_CONFIG.items():
        print(f"{k}: {v}")
    
    print("\n --- Example Sampled Individual ---")
    example_individual = sample_hyperparameters(HYPERPARAMETER_SPACE)
    for k, v in example_individual.items():
        print(f"{k}: {v}")
    
    print(f"\nExample learning rate: {example_individual["lr"]:.6f}")
    print(f"Example conv_dropout_rate: {example_individual["conv_dropout_rate"]:.1f}")
    print(f"Example fc_dropout_rate: {example_individual["fc_dropout_rate"]:.1f}")


