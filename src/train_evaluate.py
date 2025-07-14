"""
This is the train function script it will do the following:
1. Takes a set of hyperparameters (individual_hp) and other training configurations.
2. Sets up the data loaders (using data_loader.py and the batch_size from individual_hp).
3. Builds the DynamicCNN model (using cnn_model.py and individual_hp).
4. Defines an optimizer (based on optimizer and lr from individual_hp).
5. Trains the model for a fixed, relatively small number of epochs (e.g., FITNESS_TRACKING_CONFIG["num_epochs"]).
6. Evaluates the model on the validation set.
7. Returns the validation accuracy (or another metric) as the fitness score.
8. Includes basic early stopping to save time if a model performs poorly or stops improving
"""
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from .cnn_model import DynamicCNN
from .data_loader import get_dataloaders
from .hpo_config import FITNESS_TRACKING_CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} for train_evaluate")

def get_optimizer(model_parameters, optimizer_name, learning_rate):
    """Returns an optimizer instance based on its name"""
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def evaluate_model(model, dataloader, criterion):
    """Evaluates the model on the given dataloader""" 
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

def train_and_evaluate_individual(individual_hp, generation_num, individual_num):
    print(f"\n--- Gen {generation_num}, Ind {individual_num} ---")
    print(f"Training with HP: {individual_hp}")

    try:
        # 1. Get DataLoaders
        current_batch_size = int(individual_hp["batch_size"])
        train_loader, val_loader, _, num_classes, _ = get_dataloaders(
            data_dir=FITNESS_TRACKING_CONFIG["data_dir"],
            batch_size=current_batch_size
        )

        if num_classes != FITNESS_TRACKING_CONFIG["num_classes"]:
            print(f"Warning: num_classes from data_loader ({num_classes}) differs from FITNESS_TRACKING_CONFIG ({FITNESS_TRACKING_CONFIG["num_classes"]}). Using {num_classes}.")
        
        # 2. Build the model
        model = DynamicCNN(
            input_channels=3, 
            image_size=FITNESS_TRACKING_CONFIG["image_size"],
            num_classes=num_classes,
            hp=individual_hp
        ).to(DEVICE)

        # 3. Optimizer and Loss Function
        optimizer = get_optimizer(model.parameters(), individual_hp["optimizer"],
                                  individual_hp["lr"])
        criterion = nn.CrossEntropyLoss()

        # 4. Training Loop
        best_val_accuracy = 0.0
        epochs_no_improve = 0

        for epoch in range(FITNESS_TRACKING_CONFIG["num_epochs"]):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{FITNESS_TRACKING_CONFIG["num_epochs"]} Trn", leave=False)
            
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                progress_bar.set_postfix(loss=loss.item())
            epoch_loss = running_loss / len(train_loader.dataset)

            # Validation
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
            print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= FITNESS_TRACKING_CONFIG["patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        print(f"Gen {generation_num}, Ind {individual_num}: Finished training. Best Val Acc: {best_val_accuracy:.4f}")
        return best_val_accuracy
    
    except Exception as e:
        print(f"Error during training/evaluation for Gen {generation_num}, Ind {individual_num} with HP {individual_hp}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    from hpo_config import sample_hyperparameters, HYPERPARAMETER_SPACE, FITNESS_TRACKING_CONFIG

    print("Testing train_and_evaluate_individual function...")
    
    # 1. Sample random hyperparameters for testing
    test_hp = sample_hyperparameters(HYPERPARAMETER_SPACE)

    # Modify some HPs for a quicker test
    test_hp['batch_size'] = 16 
    test_hp['num_conv_blocks'] = 2
    test_hp['conv_filters_start'] = 16
    test_hp['num_fc_layers'] = 1
    test_hp['fc_neurons_start'] = 64


    if "batch_size" not in test_hp: test_hp["batch_size"] = 32
    if "optimizer" not in test_hp: test_hp["optimizer"] = "adam"
    if "lr" not in test_hp: test_hp["lr"] = 0.001

    print("\nUsing Test Hyperparameters:")
    for k, v in test_hp.items():
        print(f" {k}: {v}")

    fitness = train_and_evaluate_individual(test_hp, generation_num=0, individual_num=0)
    print(f"\nTest Completed. Fitness (Validation Accuracy): {fitness:.4f}")

    test_hp_problem = {
        "lr": 0.1, "batch_size": 128, "optimizer": "sgd",
        "num_conv_blocks": 1, "conv_filters_start": 8, "kernel_size": 7,
        "conv_activation": "relu", "conv_dropout_rate": 0.5,
        "num_fc_layers": 1, "fc_neurons_start": 32,
        "fc_activation": "relu", "fc_dropout_rate": 0.7
    }
    print("\n--- Testing with potentially problematic HPs ---")
    fitness_problem = train_and_evaluate_individual(test_hp_problem, generation_num=0, individual_num=1)
    print(f"\nTest with problematic HPs completed. Fitness: {fitness_problem:.4f}")
