#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to classify cat sounds using the trained model.
This script focuses on the core functionality with minimal dependencies.
"""

import os
import sys
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Add parent directory to path to import from model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import EnhancedAudioCNN

def classify_cat_sound(audio_file_path, model_path):
    """
    Classify if an audio file contains a cat sound.
    
    Args:
        audio_file_path (str): Path to the audio file
        model_path (str): Path to the model weights
        
    Returns:
        bool: True if the sound is from a cat, False otherwise
    """
    # Step 1: Load the audio file
    x, sr = librosa.load(audio_file_path)
    
    # Step 2: Create spectrogram
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    # Step 3: Save spectrogram as temporary image
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok=True)
    file_name = os.path.basename(audio_file_path).replace('.wav', '')
    spectrogram_path = os.path.join(temp_dir, f"{file_name}.jpg")
    
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, y_axis='hz')
    plt.axis('off')
    plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Step 4: Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedAudioCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    
    # Step 5: Prepare the image for the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(spectrogram_path).convert("RGB")
    img_tensor = transform(image)
    features = img_tensor.unsqueeze(0).to(device)
    
    # Step 6: Make prediction
    with torch.no_grad():
        prediction = model(features)
    
    # Step 7: Process result (0 = cat, 1 = dog)
    pred_value = prediction.item()
    is_cat = pred_value < 0.5
    confidence = 1 - pred_value if is_cat else pred_value
    
    # Clean up
    if os.path.exists(spectrogram_path):
        os.remove(spectrogram_path)
    
    print(f"Classification result: {'CAT' if is_cat else 'NOT CAT'} (confidence: {confidence:.2f})")
    return is_cat

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python simple_cat_classifier.py <audio_file_path> [model_path]")
        sys.exit(1)
    
    # Get audio file path
    audio_file_path = sys.argv[1]
    
    # Get model path if provided
    model_path = sys.argv[2] if len(sys.argv) >= 3 else "../audio_classification_model_augmented.pth"
    
    # Run classification
    result = classify_cat_sound(audio_file_path, model_path)
    
    # Exit with appropriate code (0 for cat, 1 for not cat)
    sys.exit(0 if result else 1)
