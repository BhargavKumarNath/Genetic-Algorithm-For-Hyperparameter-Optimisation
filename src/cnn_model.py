# This script will define a function or class that constructs a CNN based on the hyperparameters provided by GA
import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def get_activation_fn(activation_name_str):
    """Returns an activation function module based on its name"""
    if activation_name_str == "relu":
        return nn.ReLU()
    elif activation_name_str == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_name_str == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name_str}")

class DynamicCNN(nn.Module):
    def __init__(self, input_channels, image_size, num_classes, hp):
        """
        Initialise a dynamic CNN based on hyperparameters

        Args:
            input_channels (int): Number of input channels (eg. 3 for RGB)
            image_size (tuple): (height, width) of the input images.
            num_classes (int): Number of output classes.
            hp (dict): Dictionary of hyperparameters.
        """
        super(DynamicCNN, self).__init__()
        self.hp = hp
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes

        self.conv_layers = self._create_conv_layers()

        self.flattened_size = self._get_flattened_size()
        self.fc_layers = self._create_fc_layers()

    def _create_conv_layers(self):
        layers = OrderedDict()
        in_channels = self.input_channels
        current_filters = self.hp["conv_filters_start"]

        for i in range(self.hp["num_conv_blocks"]):
            layers[f"conv{i+1}"] = nn.Conv2d(
                in_channels,
                current_filters,
                kernel_size=self.hp["kernel_size"],
                padding="same"
            )
            layers[f"bn{i+1}"] = nn.BatchNorm2d(current_filters)
            layers[f"act{i+1}"] = get_activation_fn(self.hp["conv_activation"])
            layers[f"pool{i+1}"] = nn.MaxPool2d(kernel_size=2, stride=2)
            if self.hp["conv_dropout_rate"] > 0:
                layers[f"drop{i+1}"] = nn.Dropout2d(self.hp["conv_dropout_rate"])
            
            in_channels = current_filters
            current_filters = int(current_filters * 2)
        return nn.Sequential(layers)
    
    def _get_flattened_size(self):
        # Create a dummy input tensor
        # use img size from hpo_config 
        dummy_input = torch.randn(1, self.input_channels, self.image_size[0], self.image_size[1])
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
        
        return int(torch.flatten(output, 1).size(1))
    
    def _create_fc_layers(self):
        layers = OrderedDict()
        in_features = self.flattened_size
        current_neurons = self.hp["fc_neurons_start"]

        for i in range(self.hp["num_fc_layers"]):
            layers[f"fc{i+1}"] = nn.Linear(in_features, current_neurons)
            layers[f"fc_act{i+1}"] = get_activation_fn(self.hp["fc_activation"])
            if self.hp["fc_dropout_rate"] > 0:
                layers[f"fc_drop{i+1}"] = nn.Dropout(self.hp["fc_dropout_rate"])

            in_features = current_neurons
            if self.hp["num_fc_layers"] > 1 and i < self.hp["num_fc_layers"] - 1:   
                current_neurons = max(current_neurons // 2, self.num_classes * 2)
        layers["output"] = nn.Linear(in_features, self.num_classes)

        return nn.Sequential(layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = self.fc_layers(x)
        return x
    

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from src.hpo_config import sample_hyperparameters, HYPERPARAMETER_SPACE, FITNESS_TRACKING_CONFIG

    # Example usege:
    print("Testing DynamicCNN generation...")

    # 1. Sample Random Hyperparameters
    example_hp = sample_hyperparameters(HYPERPARAMETER_SPACE)
    print("\nSampled Hyperparameters:")
    for k, v in example_hp.items():
        print(f" {k}: {v}")
    
    # 2. Define model parameters
    input_channels = 3
    image_h, image_w = FITNESS_TRACKING_CONFIG["image_size"]
    num_classes = FITNESS_TRACKING_CONFIG["num_classes"]

    # 3. Create the model
    try:
        model = DynamicCNN(input_channels, (image_h, image_w), num_classes, example_hp)
        print(f"\nmodel Created Successfully!")
        print(model)

        # 4. Test Forward Pass
        batch_size_test = example_hp.get("batch_size", 32)
        dummy_batch = torch.randn(batch_size_test, input_channels, image_h, image_w)

        print(f"\nTesting forward pass with batch shape: {dummy_batch.shape}...")
        output = model(dummy_batch)
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size_test, num_classes), "output shape mismatch!"
        print("Forward pass successful.")
    
    except Exception as e:
        print(f"\nError during model creation or forward pass: {e}")
        import traceback 
        traceback.print_exc()
    
    print("\n--- Another Test with different HPs ---")
    # Example with min conv block and min FC layers
    specific_hp = {
        "lr": 0.001, "batch_size":32, "optimizer": "adam",
        "num_conv_blocks": 2, "conv_filters_start": 16, "kernel_size": 3,
        "conv_activation": "relu", "conv_dropout_rate":0.1, 
        "num_fc_layers":1, "fc_neurons_start": 128,
        "fc_activation": "relu", "fc_dropout_rate": 0.2
    }
    print("\nSpecific Hyperparameters: ")
    for k, v in specific_hp.items():
        print(f" {k}: {v}")
    
    try:
        model_specific = DynamicCNN(input_channels, (image_h, image_w), num_classes, specific_hp)
        print(f"\nModel (specific HP) created successfully!")

        dummy_batch_specific = torch.randn(specific_hp["batch_size"], input_channels, image_h, image_w)
        output_specific = model_specific(dummy_batch_specific)
        assert output_specific.shape == (specific_hp["batch_size"], num_classes), "output shape mismatch (specific HP)!"
        print("Forward pass (specific HP) successful.")
    
    except Exception as e:
        print(f"\nError during model creation or forward pass (specific HP): {e}")
        import traceback
        traceback.print_exc()



