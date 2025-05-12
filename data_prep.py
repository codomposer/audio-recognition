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

def process_audio_files():
    """Process all audio files and convert them to spectrograms."""
    # Process training data
    CAT_TRAIN = './data/cats_dogs/train/cat/'
    for sound in os.listdir(CAT_TRAIN):
        create_spectogram(sound, CAT_TRAIN, './img_dataset/train/cat/')
    
    DOG_TRAIN = './data/cats_dogs/train/dog/'
    for sound in os.listdir(DOG_TRAIN):
        create_spectogram(sound, DOG_TRAIN, './img_dataset/train/dog/')
    
    # Process test data
    DOG_TEST = './data/cats_dogs/test/test/'
    for sound in os.listdir(DOG_TEST):
        create_spectogram(sound, DOG_TEST, './img_dataset/test/dog/')
    
    CAT_TEST = './data/cats_dogs/test/cats/'
    for sound in os.listdir(CAT_TEST):
        create_spectogram(sound, CAT_TEST, './img_dataset/test/cat/')
    
    # Process inference data if it exists
    INFERENCES = './data/cats_dogs/inferences/'
    if os.path.exists(INFERENCES):
        for sound in os.listdir(INFERENCES):
            create_spectogram(sound, INFERENCES, './img_dataset/inferences/')

def create_metadata():
    """Create metadata CSV files for the dataset."""
    image_names_ls = []
    file_location = []
    
    for i in ['test', 'train']:
        for j in ['cat', 'dog']:
            image_names_ls.append([img for img in os.listdir(f'./img_dataset/{i}/{j}/')])
            file_location.append([f'./img_dataset/{i}/{j}/{img}' for img in os.listdir(f'./img_dataset/{i}/{j}/')])
    
    # Create dataframes for test and train sets
    test_set = pd.DataFrame({
        'image_name': image_names_ls[0] + image_names_ls[1], 
        'image_location': file_location[0] + file_location[1], 
        'target': len(file_location[0])*['cat'] + len(file_location[1])*['dog']
    })
    
    train_set = pd.DataFrame({
        'image_name': image_names_ls[2] + image_names_ls[3], 
        'image_location': file_location[2] + file_location[3], 
        'target': len(file_location[2])*['cat'] + len(file_location[3])*['dog']
    })
    
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
    process_audio_files()
    
    # Step 4: Create metadata CSV files
    print("Creating metadata files...")
    create_metadata()
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
