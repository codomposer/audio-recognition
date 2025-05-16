import os
import sys
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import EnhancedAudioCNN

# Set up logging to a file
log_file = open('classification_log.txt', 'w')

def log(message):
    """Write message to log file and print to console"""
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

def create_spectrogram(audio_file_path, temp_dir='./temp'):
    """Create a spectrogram image from an audio file."""
    log(f"Creating spectrogram for {audio_file_path}")
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load audio file
    x, sr = librosa.load(audio_file_path)
    log(f"Audio loaded: length={len(x)}, sample rate={sr}")
    
    # Create spectrogram
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    log(f"Spectrogram created: shape={Xdb.shape}")
    
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
    
    log(f"Spectrogram saved to {output_path}")
    return output_path

def classify_cat_sound():
    """Main function to test the cat sound classifier"""
    try:
        # Define paths
        audio_file = "../data/cats_dogs/train/cat/cat_1.wav"
        model_path = "../audio_classification_model_augmented.pth"
        
        log("Starting cat sound classification test")
        log(f"Python version: {sys.version}")
        log(f"Current working directory: {os.getcwd()}")
        
        # Check if files exist
        log(f"Audio file exists: {os.path.exists(audio_file)}")
        log(f"Model file exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(audio_file):
            log(f"ERROR: Audio file {audio_file} not found")
            return
            
        if not os.path.exists(model_path):
            log(f"ERROR: Model file {model_path} not found")
            return
        
        # Create spectrogram
        spectrogram_path = create_spectrogram(audio_file)
        
        # Load model
        log("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log(f"Using device: {device}")
        
        model = EnhancedAudioCNN()
        log("Model initialized")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        log("Model weights loaded")
        
        model.eval()
        model = model.to(device)
        log("Model ready for inference")
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and transform the spectrogram image
        log("Processing spectrogram...")
        image = Image.open(spectrogram_path).convert("RGB")
        log(f"Image loaded: size={image.size}, mode={image.mode}")
        
        img_tensor = transform(image)
        log(f"Image transformed: shape={img_tensor.shape}")
        
        features = img_tensor.unsqueeze(0).to(device)
        log(f"Features ready for model: shape={features.shape}")
        
        # Make prediction
        log("Making prediction...")
        with torch.no_grad():
            prediction = model(features)
        
        # Extract prediction value (0-1)
        pred_value = prediction.item()
        log(f"Raw prediction value: {pred_value}")
        
        is_cat = pred_value < 0.5  # 0 = cat, 1 = dog
        confidence = 1 - pred_value if is_cat else pred_value
        
        # Print result
        result = "CAT" if is_cat else "NOT CAT"
        log(f"Classification result: {result} (confidence: {confidence:.2f})")
        
        # Clean up
        if os.path.exists(spectrogram_path):
            os.remove(spectrogram_path)
            log(f"Temporary spectrogram file removed")
        
        log("Classification test completed successfully")
        
    except Exception as e:
        log(f"ERROR during classification: {str(e)}")
        log("Traceback:")
        traceback.print_exc(file=log_file)

if __name__ == "__main__":
    classify_cat_sound()
    log_file.close()
    print("Test completed. See classification_log.txt for details.")
