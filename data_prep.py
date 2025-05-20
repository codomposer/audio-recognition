#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preparation script for audio classification.
Converts audio files to spectrograms and organizes them into train/test folders.
"""

# Import Libraries
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms

def create_spectogram(audio_file_name, source_path, save_path):
    """
    Create a spectrogram image from an audio file and save it as a jpg file.
    
    Args:
        audio_file_name (str): Name of the audio file
        source_path (str): Path to the directory containing the audio file
        save_path (str): Path to save the spectrogram image
    """
    x, sr = librosa.load(source_path + audio_file_name)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, y_axis='hz')
    plt.ylabel('')
    plt.axis('off')
    file_name = audio_file_name.replace('.wav', '')
    plt.savefig(save_path + file_name + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

def create_directory_structure():
    """Create the necessary directory structure for the dataset."""
    os.makedirs('img_dataset/train/cat', exist_ok=True)
    os.makedirs('img_dataset/train/dog', exist_ok=True)
    os.makedirs('img_dataset/test/cat', exist_ok=True)
    os.makedirs('img_dataset/test/dog', exist_ok=True)
    os.makedirs('img_dataset/inferences', exist_ok=True)

def limit_files(file_list, limit):
    """Limit the number of files to process.
    
    Args:
        file_list (list): List of files to limit
        limit (int): Maximum number of files to return
        
    Returns:
        list: Limited list of files
    """
    if len(file_list) <= limit:
        return file_list
    return file_list[:limit]

def process_audio_files(train_limit=100, test_limit=13):
    """Process audio files and convert them to spectrograms.
    
    Args:
        train_limit (int): Number of samples to use for training from each class
        test_limit (int): Number of samples to use for testing from each class
    """
    # Process training data with limits
    CAT_TRAIN = './data/cats_dogs/train/cat/'
    cat_train_files = limit_files(os.listdir(CAT_TRAIN), train_limit)
    print(f"Processing {len(cat_train_files)} cat training samples")
    for sound in cat_train_files:
        create_spectogram(sound, CAT_TRAIN, './img_dataset/train/cat/')
    
    DOG_TRAIN = './data/cats_dogs/train/dog/'
    dog_train_files = limit_files(os.listdir(DOG_TRAIN), train_limit)
    print(f"Processing {len(dog_train_files)} dog training samples")
    for sound in dog_train_files:
        create_spectogram(sound, DOG_TRAIN, './img_dataset/train/dog/')
    
    # Process test data with limits
    DOG_TEST = './data/cats_dogs/test/test/'
    dog_test_files = limit_files(os.listdir(DOG_TEST), test_limit)
    print(f"Processing {len(dog_test_files)} dog testing samples")
    for sound in dog_test_files:
        create_spectogram(sound, DOG_TEST, './img_dataset/test/dog/')
    
    CAT_TEST = './data/cats_dogs/test/cats/'
    cat_test_files = limit_files(os.listdir(CAT_TEST), test_limit)
    print(f"Processing {len(cat_test_files)} cat testing samples")
    for sound in cat_test_files:
        create_spectogram(sound, CAT_TEST, './img_dataset/test/cat/')
    
    # Process inference data if it exists
    INFERENCES = './data/cats_dogs/inferences/'
    if os.path.exists(INFERENCES):
        for sound in os.listdir(INFERENCES):
            create_spectogram(sound, INFERENCES, './img_dataset/inferences/')

def create_metadata():
    """Create metadata CSV files for the dataset.
    
    Only includes files that actually exist in the image directories.
    """
    # Initialize empty lists for each dataset
    test_images = []
    test_locations = []
    test_targets = []
    
    train_images = []
    train_locations = []
    train_targets = []
    
    # Process test data
    for class_name in ['cat', 'dog']:
        img_dir = f'./img_dataset/test/{class_name}/'
        if os.path.exists(img_dir):
            for img in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img)
                if os.path.isfile(img_path):
                    test_images.append(img)
                    test_locations.append(img_path)
                    test_targets.append(class_name)
    
    # Process train data
    for class_name in ['cat', 'dog']:
        img_dir = f'./img_dataset/train/{class_name}/'
        if os.path.exists(img_dir):
            for img in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img)
                if os.path.isfile(img_path):
                    train_images.append(img)
                    train_locations.append(img_path)
                    train_targets.append(class_name)
    
    # Create dataframes for test and train sets
    test_set = pd.DataFrame({
        'image_name': test_images,
        'image_location': test_locations,
        'target': test_targets
    })
    
    train_set = pd.DataFrame({
        'image_name': train_images,
        'image_location': train_locations,
        'target': train_targets
    })
    
    # Print dataset statistics
    print(f"Training set: {len(train_set)} samples")
    print(f"  - Cat samples: {len(train_set[train_set['target'] == 'cat'])}")
    print(f"  - Dog samples: {len(train_set[train_set['target'] == 'dog'])}")
    
    print(f"Test set: {len(test_set)} samples")
    print(f"  - Cat samples: {len(test_set[test_set['target'] == 'cat'])}")
    print(f"  - Dog samples: {len(test_set[test_set['target'] == 'dog'])}")
    
    # Save metadata to CSV files
    test_set.to_csv('./img_dataset/test/test.csv', index=False)
    train_set.to_csv('./img_dataset/train/train.csv', index=False)

def unzip_data():
    """Unzip the data archive if it hasn't been extracted yet."""
    if not os.path.exists('./data/cats_dogs'):
        print("Extracting data archive...")
        os.system('unzip ./data/archive.zip -d ./data/')
    else:
        print("Data already extracted.")

def main():
    """Main function to run the data preparation process."""
    print("Starting data preparation...")
    
    # Step 1: Unzip data if needed
    # unzip_data()
    
    # Step 2: Create directory structure
    create_directory_structure()
    
    # Step 3: Process audio files and create spectrograms
    print("Converting audio files to spectrograms...")
    # Use 100 samples for training and 13 for testing from each class
    process_audio_files(train_limit=100, test_limit=13)
    
    # Step 4: Create metadata CSV files
    print("Creating metadata files...")
    create_metadata()
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
