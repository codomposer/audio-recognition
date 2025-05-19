#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to classify cat sounds using the trained model.
Input: WAV file
Output: True if the sound is from a cat, False otherwise

This script is specifically designed to identify cat sounds in audio files.
It doesn't matter if the other sound is a dog or something else - it only checks
if it's a cat sound or not.
"""

import os
import sys
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set up basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to path to import from model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import EnhancedAudioCNN

def create_spectrogram(audio_file_path, temp_dir='./temp'):
    """
    Create a spectrogram image from an audio file.
    
    Args:
        audio_file_path (str): Path to the audio file
        temp_dir (str): Directory to save temporary spectrogram image
        
    Returns:
        str: Path to the created spectrogram image
    """
    logging.info(f"Creating spectrogram from {audio_file_path}")
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load audio file
    x, sr = librosa.load(audio_file_path)
    
    # Create spectrogram
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    # Create figure and save spectrogram
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, y_axis='hz')
    plt.ylabel('')
    plt.axis('off')
    
    # Generate output filename
    file_name = os.path.basename(audio_file_path).replace('.wav', '')
    output_path = os.path.join(temp_dir, f"{file_name}.jpg")
    
    # Save spectrogram image
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return output_path

def load_model(model_path):
    """
    Load the trained model.
    
    Args:
        model_path (str): Path to the saved model weights
        
    Returns:
        model: Loaded PyTorch model
    """
    logging.info(f"Loading model from {model_path}")
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = EnhancedAudioCNN()
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set model to evaluation mode
    model.eval()
    
    return model.to(device)

def classify_audio(model, spectrogram_path):
    """
    Classify an audio file using the trained model.
    
    Args:
        model: Loaded PyTorch model
        spectrogram_path (str): Path to the spectrogram image
        
    Returns:
        bool: True if the sound is from a cat, False otherwise
        float: Confidence score (0-1)
    """
    logging.info(f"Classifying audio using spectrogram at {spectrogram_path}")
    # Define image transformations (same as used during training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform the spectrogram image
    image = Image.open(spectrogram_path).convert("RGB")
    img_tensor = transform(image)
    
    # Add batch dimension
    features = img_tensor.unsqueeze(0)
    
    # Get device
    device = next(model.parameters()).device
    features = features.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(features)
    
    # Extract prediction value (0-1)
    pred_value = prediction.item()
    
    # In this model, 0 = cat, 1 = dog
    # So we need to return True if prediction is < 0.5
    is_cat = pred_value < 0.5
    confidence = 1 - pred_value if is_cat else pred_value
    
    return is_cat, confidence

def main(audio_file_path, model_path=None):
    """
    Main function to classify a cat sound.
    
    Args:
        audio_file_path (str): Path to the audio file
        model_path (str, optional): Path to the saved model weights
        
    Returns:
        bool: True if the sound is from a cat, False otherwise
    """
    logging.info(f"Starting classification of: {audio_file_path}")
    logging.info(f"Using model: {model_path}")
    # Set default model path if not provided
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'audio_classification_model_augmented.pth'
        )
    logging.info(f"Final model path: {model_path}")
    
    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        logging.error(f"Audio file '{audio_file_path}' not found.")
        return False
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_path}' not found.")
        return False
    
    try:
        # Create spectrogram
        print("Creating spectrogram...")
        spectrogram_path = create_spectrogram(audio_file_path)
        logging.info(f"Spectrogram created at: {spectrogram_path}")
        
        # Load model
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully")
        logging.info("Model loaded successfully")
        
        # Classify audio
        print("Classifying audio...")
        is_cat, confidence = classify_audio(model, spectrogram_path)
        print("Classification complete")
        logging.info("Classification complete")
        
        # Print result
        result = "CAT" if is_cat else "NOT CAT"
        print(f"Classification result: {result} (confidence: {confidence:.2f})")
        logging.info(f"Classification result: {result} (confidence: {confidence:.2f})")
        
        # Clean up temporary files
        if os.path.exists(spectrogram_path):
            os.remove(spectrogram_path)
            logging.info(f"Removed temporary file: {spectrogram_path}")
        
        return is_cat
        
    except Exception as e:
        error_msg = f"Error during classification: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Cat Sound Classifier")
    print("-" * 50)
    logging.info("Script started")
    logging.info(f"Arguments: {sys.argv}")
    # Check if audio file path is provided
    if len(sys.argv) < 2:
        print("Usage: python classify_cat_sound.py <audio_file_path> [model_path]")
        sys.exit(1)
    
    # Get audio file path
    audio_file_path = sys.argv[1]
    
    # Get model path if provided
    model_path = None
    if len(sys.argv) >= 3:
        model_path = sys.argv[2]
    
    # Run classification
    result = main(audio_file_path, model_path)
    
    # Exit with appropriate code (0 for cat, 1 for not cat)
    print("-" * 50)
    print(f"Final result: {'CAT' if result else 'NOT CAT'}")
    logging.info(f"Final result: {'CAT' if result else 'NOT CAT'}")
    sys.exit(0 if result else 1)
