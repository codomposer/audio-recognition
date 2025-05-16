import os
import sys
import traceback

# Print Python version and working directory
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    # Check if files exist
    audio_file = "../data/cats_dogs/train/cat/cat_1.wav"
    model_path = "../audio_classification_model_augmented.pth"
    
    print(f"Audio file exists: {os.path.exists(audio_file)}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    # Try to import the module
    print("Importing classify_cat_sound module...")
    # Add the parent directory to sys.path so Python can find the server module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from server.classify_cat_sound import main as classify_cat_sound
    
    # Run classification
    print(f"Running classification on {audio_file} with model {model_path}")
    result = classify_cat_sound(audio_file, model_path)
    
    # Print result
    print(f"Classification result: {'CAT' if result else 'NOT CAT'}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("Traceback:")
    traceback.print_exc()
