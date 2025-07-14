import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import torch.nn as nn
from tqdm import tqdm

def plot_fitness_history(fitness_history, save_path=None):
    """Plots the best fitness per generation"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, marker="o")
    plt.title("GA HPO: Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Validation Accuracy")
    plt.grid(True)
    plt.xticks(range(1, len(fitness_history) + 1))
    if save_path:
        plt.savefig(save_path)
        print(f"Fitness plot saved to {save_path}")
    plt.show()

def save_ga_results(best_individual_hp, best_fitness, fitness_history, config, results_dir):
    """Saves the best hyperparameter, fitness, history, and config to files"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save best HPs and fitness
    with open(os.path.join(results_dir, "best_hyperparameters.txt"), "w") as f:
        f.write(f"Best Validation Accuracy: {best_fitness:.4f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_individual_hp.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6f}\n")
            else:
                f.write(f"  {key}: {value}\n")
    print(f"Best hyperparameters saved to {os.path.join(results_dir, "best_hyperparameters.txt")}")

    # Save fitness history to csv
    history_df = pd.DataFrame({"generation": range(1, len(fitness_history) + 1), "best_fitness": fitness_history})
    history_path = os.path.join(results_dir, config["ga_log_name"])     # use from FINAL_TRAINING_CONFIG
    history_df.to_csv(history_path, index=False)
    print(f"Fitness history saved to {history_path}")

    # Plot the fitness history
    plot_path = os.path.join(results_dir, "plots", "fitness_history.png")
    plot_dir = os.path.dirname(plot_path)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_fitness_history(fitness_history, save_path=plot_path)

def train_final_model(best_hp, final_config, device):
    """Trains the final model using the best hyperparameters found by Genetic Algorithm"""
    from .data_loader import get_dataloaders
    from .cnn_model import DynamicCNN
    from .train_evaluate import get_optimizer, evaluate_model

    print("\n--- Training Final Model with Best Hyperparameters ---")
    print(f"Best HPs: {best_hp}")

    best_val_accuracy = 0.0
    test_accuracy = -1.0

    # 1. DataLoaders
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        data_dir=final_config["data_dir"],
        batch_size=int(best_hp["batch_size"])
    )
    print(f"Data loaded for final training. Num classes: {num_classes}, Classes: {class_names}")

    # 2. Model
    model = DynamicCNN(
        input_channels=3,
        image_size=final_config["image_size"],
        num_classes=num_classes,
        hp=best_hp
    ).to(device)
    print("Final model created")
    print(model)        # show the model architecture

    # 3. optimizer and loss
    optimizer = get_optimizer(model.parameters(), best_hp["optimizer"], best_hp["lr"])
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    epochs_no_improve = 0

    print(f"Starting final training for {final_config["num_epochs"]} epochs...")
    for epoch in range(final_config["num_epochs"]):
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Final Epoch {epoch+1}/{final_config["num_epochs"]} Trn", leave=False)

        for inputs, labels in train_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_progress_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0

            # Save the best mode;
            model_save_path = os.path.join(final_config["results_dir"], final_config["best_model_name"])
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation accuracy imporved. Model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= final_config["patience"]:
            print(f"Early stopping triggered after {epoch + 1} epochs during final training")
            break

    print("\n--- Final Model Trianing Finished ---")
    print(f"Best Validation accuracy during final training: {best_val_accuracy:4f}")

    # 5. Load best model and evaluate on Test Set
    print("\nLoading best model for test set evaluation...")
    best_model_path = os.path.join(final_config["results_dir"], final_config["best_model_name"])
    if os.path.exists(best_model_path):
        # Recreate model structure and load state_dict
        final_model_for_test = DynamicCNN(
            input_channels=3,
            image_size=final_config["image_size"],
            num_classes=num_classes,
            hp=best_hp
        ).to(device)
        final_model_for_test.load_state_dict(torch.load(best_model_path, map_location=device))

        test_loss, test_accuracy = evaluate_model(final_model_for_test, test_loader, criterion)
        print(f"Test Set Evaluation Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

        # Save test accuracy to results
        with open(os.path.join(final_config["results_dir"], "test_set_results.txt"), "w") as f:
            f.write(f"Test Set Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test Set Loss: {test_loss:.4f}\n")
    else:
        print(f"Could not final best model at {best_model_path} for test set evaluation.")

    return best_val_accuracy, test_accuracy if "test_accuracy" in locals() else -1.0
