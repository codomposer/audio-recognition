"""
Display module for audio classification model.
This module provides functionality to load a trained model and run inferences on spectrogram images.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time
import random
from model import EnhancedAudioCNN, train_model, eval_model


def load_model(model_path="audio_classification_model_augmented.pth", device=None):
    """
    Load a trained model from the specified path.
    
    Args:
        model_path (str): Path to the saved model file
        device (torch.device): Device to load the model on (CPU or CUDA)
        
    Returns:
        model: Loaded PyTorch model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    model = EnhancedAudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    return model, device


def run_inference(model, inference_dir, device=None):
    """
    Run inferences on spectrogram images in the specified directory.
    
    Args:
        model: Trained PyTorch model
        inference_dir (str): Directory containing spectrogram images
        device (torch.device): Device to run inference on
        
    Returns:
        dict: Dictionary mapping filenames to predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Run inferences on test spectrogram images
    print("\nRunning inferences on test spectrogram images:")
    model.eval()
    
    results = {}
    
    for spectrogram in os.listdir(inference_dir):
        if not spectrogram.endswith(".jpg"):
            continue
        try:
            # Load the spectrogram image
            img_path = os.path.join(inference_dir, spectrogram)
            image = Image.open(img_path).convert("RGB")
            # Apply transforms
            img_tensor = transform(image)
            # For CNN, we can use the image tensor directly
            # Add batch dimension
            features = img_tensor.unsqueeze(0).to(device)
            # Make prediction
            pred = model(features)
            if pred[0, 0] < 0.5:
                label = "cat"
            else:
                label = "dog"
            print(f"for {spectrogram}, the prediction is {label}.")
            results[spectrogram] = label
        except Exception as e:
            print(f"Error processing {spectrogram}: {e}")
            continue
    
    return results

def main():
    """
    Main function to demonstrate the usage of the module.
    """
    # Load the model
    model, device = load_model()
    
    # Define the directory containing test images
    # test_dir = "img_dataset/test/cat"
    test_dir = "img_dataset/inferences"
    
    # Check if the directory exists, otherwise use a default path
    if not os.path.exists(test_dir):
        print(f"Warning: {test_dir} does not exist. Please provide a valid directory.")
        return
    
    # Run inferences
    results = run_inference(model, test_dir, device)
    
    # Print summary
    cat_count = sum(1 for label in results.values() if label == "cat")
    dog_count = sum(1 for label in results.values() if label == "dog")
    
    print("\nSummary:")
    print(f"Total images processed: {len(results)}")
    print(f"Cat predictions: {cat_count}")
    print(f"Dog predictions: {dog_count}")
    
if __name__ == "__main__":
    main()
