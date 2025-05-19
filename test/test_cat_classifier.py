"""
Simple script to test the cat sound classifier.
"""

import os
import sys
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import EnhancedAudioCNN

# Disable matplotlib interactive mode
import matplotlib
matplotlib.use('Agg')

def test_classifier():
    """Test the cat sound classifier with a sample audio file."""
    print("Starting cat sound classifier test...")
    
    # Define paths
    audio_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cats_dogs", "train", "dog", "dog_barking_0.wav")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_classification_model_augmented.pth")
    
    # Check if files exist
    print(f"Checking if audio file exists: {os.path.exists(audio_file)}")
    print(f"Checking if model file exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    try:
        # Step 1: Load the audio file
        print("Loading audio file...")
        x, sr = librosa.load(audio_file)
        print(f"Audio loaded: length={len(x)}, sample rate={sr}")
        
        # Step 2: Create spectrogram
        print("Creating spectrogram...")
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        print(f"Spectrogram created: shape={Xdb.shape}")
        
        # Step 3: Save spectrogram as temporary image
        print("Saving spectrogram as image...")
        temp_dir = './temp'
        os.makedirs(temp_dir, exist_ok=True)
        file_name = os.path.basename(audio_file).replace('.wav', '')
        spectrogram_path = os.path.join(temp_dir, f"{file_name}.jpg")
        
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, y_axis='hz')
        plt.axis('off')
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Spectrogram saved to {spectrogram_path}")
        
        # Step 4: Load the model
        print("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = EnhancedAudioCNN()
        print("Model initialized")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded")
        
        model.eval()
        model = model.to(device)
        print("Model ready for inference")
        
        # Step 5: Prepare the image for the model
        print("Processing spectrogram for model input...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(spectrogram_path).convert("RGB")
        print(f"Image loaded: size={image.size}, mode={image.mode}")
        
        img_tensor = transform(image)
        print(f"Image transformed: shape={img_tensor.shape}")
        
        features = img_tensor.unsqueeze(0).to(device)
        print(f"Features ready for model: shape={features.shape}")
        
        # Step 6: Make prediction
        print("Making prediction...")
        with torch.no_grad():
            prediction = model(features)
        
        # Step 7: Process result (0 = cat, 1 = dog)
        pred_value = prediction.item()
        print(f"Raw prediction value: {pred_value}")
        
        # Model was trained with 0=cat, 1=dog
        # So values closer to 0 indicate cat, values closer to 1 indicate dog
        is_cat = pred_value < 0.5
        confidence = 1 - pred_value if is_cat else pred_value
        
        # Print result
        result = "CAT" if is_cat else "NOT CAT"
        print(f"Classification result: {result} (confidence: {confidence:.2f})")
        
        # Clean up
        if os.path.exists(spectrogram_path):
            os.remove(spectrogram_path)
            print(f"Temporary spectrogram file removed")
        
        print("Classification test completed successfully")
        return is_cat
        
    except Exception as e:
        import traceback
        print(f"Error during classification: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_classifier()
    print(f"\nFinal result: {'CAT' if result else 'NOT CAT'}")
    print("Test completed.")
