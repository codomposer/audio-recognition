#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to classify cat sounds using the trained model.
Input: WAV file
Output: True if the sound is from a cat, False otherwise
"""

import os
import sys
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import EnhancedAudioCNN

def create_spectrogram(audio_file_path, temp_dir='./temp'):
    """Create a spectrogram image from an audio file."""
    print(f"Creating spectrogram for {audio_file_path}")
    
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
    
    print(f"Spectrogram created at {output_path}")
    return output_path

def classify(audio_file_path, model_path):
    """Classify an audio file as cat or not cat."""
    try:
        print(f"Starting classification of {audio_file_path}")
        print(f"Using model {model_path}")
        
        # Check if files exist
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file {audio_file_path} not found")
            return False
            
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found")
            return False
        
        # Create spectrogram
        spectrogram_path = create_spectrogram(audio_file_path)
        
        # Load model
        print("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnhancedAudioCNN()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model = model.to(device)
        print(f"Model loaded successfully (using {device})")
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and transform the spectrogram image
        print("Processing spectrogram...")
        image = Image.open(spectrogram_path).convert("RGB")
        img_tensor = transform(image)
        features = img_tensor.unsqueeze(0).to(device)
        
        # Make prediction
        print("Making prediction...")
        with torch.no_grad():
            prediction = model(features)
        
        # Extract prediction value (0-1)
        pred_value = prediction.item()
        is_cat = pred_value < 0.5  # 0 = cat, 1 = dog
        confidence = 1 - pred_value if is_cat else pred_value
        
        # Print result
        result = "CAT" if is_cat else "NOT CAT"
        print(f"Classification result: {result} (confidence: {confidence:.2f})")
        
        # Clean up
        if os.path.exists(spectrogram_path):
            os.remove(spectrogram_path)
            
        return is_cat
        
    except Exception as e:
        import traceback
        print(f"Error during classification: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Cat Sound Classifier")
    print("-" * 50)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python cat_classifier.py <audio_file_path> [model_path]")
        sys.exit(1)
    
    # Get audio file path
    audio_file_path = sys.argv[1]
    
    # Get model path if provided, otherwise use default
    model_path = sys.argv[2] if len(sys.argv) >= 3 else "../audio_classification_model_augmented.pth"
    
    # Run classification
    result = classify(audio_file_path, model_path)
    
    # Print final result
    print("-" * 50)
    print(f"Final result: {'CAT' if result else 'NOT CAT'}")
    
    # Exit with appropriate code
    sys.exit(0 if result else 1)
