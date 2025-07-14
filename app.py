import streamlit as st
import torch
import json
from torchvision import transforms
from PIL import Image
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from cnn_model import DynamicCNN
from hpo_config import IMAGE_SIZE, DATASET_MEAN, DATASET_STD

# Page config
st.set_page_config(
    page_title="Blood Cell Classifier | Evolved with Genetic Algorithm",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add state management
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Asset Loading
@st.cache_resource
def load_assets():
    """Loads the trained model, hyperparameters, and class names."""
    results_dir = "results"
    model_path = os.path.join(results_dir, "best_cnn_model.pth")
    hp_path = os.path.join(results_dir, "best_hyperparameters.txt")
    class_names_path = os.path.join(results_dir, "class_names.json")

    with open(hp_path, 'r') as f:
        lines = f.readlines()
    best_hp = {}
    for line in lines[3:]:  
        key, value = line.strip().split(': ')
        key = key.strip()
        if key in ['lr', 'conv_dropout_rate', 'fc_dropout_rate']:
            best_hp[key] = float(value)
        elif key in ['batch_size', 'num_conv_blocks', 'conv_filters_start', 'kernel_size', 'num_fc_layers', 'fc_neurons_start']:
            best_hp[key] = int(value)
        else:
            best_hp[key] = value

    # Load Class Names
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicCNN(
        input_channels=3,
        image_size=IMAGE_SIZE,
        num_classes=len(class_names),
        hp=best_hp
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    return model, class_names, device

try:
    model, class_names, device = load_assets()
    st.session_state.model_loaded = True
except FileNotFoundError as e:
    st.error(f"Error loading model assets: {e}")
    st.info("Please ensure that 'best_cnn_model.pth', 'best_hyperparameters.txt', and 'class_names.json' are in the 'results' directory.")
    st.stop()


# --- UI Layout ---

# Sidebar
with st.sidebar:
    st.title("🧬 Evolved Neural Network")
    st.markdown("### Blood Cell Classifier")
    st.write("This application uses a Convolutional Neural Network (CNN) to classify images of blood cells.")
    st.write("The model's architecture and hyperparameters were not chosen manually but were **discovered automatically using a Genetic Algorithm**.")
    
    st.subheader("Project Highlights")
    st.success("Test Set Accuracy: **97.15%**")
    st.info("Architecture found by GA: **4 Conv Blocks, 1 FC Layer**")

    st.markdown("---")
    st.write("Created by **Bhargav Kumar Nath**") 
    st.write("[View on GitHub](https://github.com/BhargavKumarNath/Genetic-Algorithm-For-Hyperparameter-Optimisation)") 

# Main content area
st.title("🔬 Interactive Blood Cell Classifier")
st.write("Upload an image of a blood cell, and the AI model will predict its type.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        # To read the file and display it
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_container_width=True)

    with col2:
        st.subheader("Prediction")
        with st.spinner("Classifying..."):
            # Define the same transformations as used for validation/testing
            transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
            ])
            
            # Preprocess the image
            img_t = transform(image)
            batch_t = torch.unsqueeze(img_t, 0).to(device) # Create a batch

            # Get model prediction
            with torch.no_grad():
                output = model(batch_t)
            
            # Get probabilities and the predicted class
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = class_names[predicted_idx.item()]
            
            # Display the result
            st.success(f"**Predicted Type:** {predicted_class.replace('_', ' ').title()}")
            st.info(f"**Confidence:** {confidence.item()*100:.2f}%")

            # Display confidence scores for all classes
            st.subheader("Confidence Scores")
            # Create a dictionary for easier plotting
            confidence_scores = {class_names[i]: prob.item() for i, prob in enumerate(probabilities)}
            st.bar_chart(confidence_scores)

st.markdown("---")
st.header("How It Works: A Look Under the Hood")
st.write("""
This isn't just a standard image classifier. It's the result of a sophisticated Hyperparameter Optimization (HPO) process.

1.  **The Challenge:** Manually designing a deep learning model is hard. Which optimizer should I use? How many layers? What learning rate? The number of combinations is astronomical.

2.  **The Solution: Genetic Algorithm (GA):** We let the computer do the work.
    *   A "population" of random models is created.
    *   Each model is trained for a short period, and its "fitness" (accuracy) is measured.
    *   The best models "breed" to create a new generation, combining their successful traits (hyperparameters).
    *   Random "mutations" introduce new ideas.
    *   After many generations, the population "evolves" towards a highly optimal solution.

3.  **The Result:** The model you're interacting with is the champion of this evolutionary process, fully trained to achieve high accuracy on this specific task. This approach showcases a powerful method for automating model design and achieving state-of-the-art results.
""")