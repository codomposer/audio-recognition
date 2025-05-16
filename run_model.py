#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the LSTM model on the audio classification dataset.
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
from model import LSTM, train_model, eval_model

# Define dataset class
class SpectrogramSequenceDataset(Dataset):
    def __init__(self, csv_file, input_size=40, max_len=100, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with spectrogram image paths and labels.
            input_size (int): Number of features to extract from each column of the spectrogram
            max_len (int): Maximum length of time steps (will pad/truncate)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.data_frame = pd.read_csv(csv_file)
        self.input_size = input_size
        self.max_len = max_len
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the spectrogram image path
        img_path = self.data_frame.iloc[idx, 1]  # image_location column
        
        # Load the spectrogram image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Convert the image tensor to a sequence for LSTM
        # image shape: [3, H, W] where H is height and W is width
        # We'll convert this to a sequence of feature vectors by treating each column as a time step
        
        # First, convert to numpy and take the mean across color channels
        img_np = image.numpy().mean(axis=0)  # Now shape is [H, W]
        
        # Resize along the height dimension to match our input_size
        # This will give us a consistent number of features per time step
        if img_np.shape[0] != self.input_size:
            # Resize by averaging or interpolating along height dimension
            resized = np.zeros((self.input_size, img_np.shape[1]))
            for i in range(self.input_size):
                # Simple linear interpolation
                orig_i = i * img_np.shape[0] / self.input_size
                floor_i = int(np.floor(orig_i))
                ceil_i = min(floor_i + 1, img_np.shape[0] - 1)
                alpha = orig_i - floor_i
                resized[i] = img_np[floor_i] * (1 - alpha) + img_np[ceil_i] * alpha
            img_np = resized
        
        # Now transpose to get [W, H] where W is time steps and H is features
        sequence = img_np.T  # Now shape is [W, input_size]
        
        # Pad or truncate to max_len
        if sequence.shape[0] < self.max_len:
            # Pad with zeros
            pad_width = self.max_len - sequence.shape[0]
            sequence = np.pad(sequence, ((0, pad_width), (0, 0)), mode='constant')
        else:
            # Truncate to max_len
            sequence = sequence[:self.max_len, :]
        
        # Convert to tensor
        features = torch.FloatTensor(sequence)
        
        # Normalize the features
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Convert target to numeric (0 for cat, 1 for dog)
        target = 0 if self.data_frame.iloc[idx, 2] == 'cat' else 1
        
        return features, target

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    # First shuffle the data with a fixed random seed for reproducibility
    train_csv = pd.read_csv('./img_dataset/train/train.csv')
    test_csv = pd.read_csv('./img_dataset/test/test.csv')
    
    train_csv = train_csv.sample(frac=1, random_state=42)
    test_csv = test_csv.sample(frac=1, random_state=42)
    
    # Save shuffled CSVs if needed
    train_csv.to_csv('./img_dataset/train/train.csv', index=False)
    test_csv.to_csv('./img_dataset/test/test.csv', index=False)
    
    # Define transformations for the spectrogram images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # LSTM parameters
    input_size = 40  # Number of features per time step
    max_len = 100  # Maximum sequence length
    
    train_dataset = SpectrogramSequenceDataset(
        csv_file='./img_dataset/train/train.csv',
        input_size=input_size,
        max_len=max_len,
        transform=transform
    )
    
    test_dataset = SpectrogramSequenceDataset(
        csv_file='./img_dataset/test/test.csv',
        input_size=input_size,
        max_len=max_len,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = LSTM(input_size=input_size, hidden_size=128, num_layers=2, bidirectional=True)
    model = model.to(device)
    print(model)
    
    # Train the model
    print("Starting model training...")
    best_model, training_stats = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=60,
        lr=0.0001
    )
    
    # Evaluate the final model
    final_accuracy, final_loss = eval_model(best_model, test_loader)
    print(f"Final model accuracy: {final_accuracy:.4f}")
    print(f"Final model loss: {final_loss:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(training_stats['epoch_nums'], training_stats['training_loss'], label='Training Loss')
    plt.plot(training_stats['epoch_nums'], training_stats['validation_loss'], label='Validation Loss')
    plt.title('Loss Curve using Binary Cross Entropy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    print("Loss curve saved to 'loss_curve.png'")
    
    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(training_stats['epoch_nums'], training_stats['validation_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy pct')
    plt.savefig('accuracy_curve.png')
    print("Accuracy curve saved to 'accuracy_curve.png'")
    
    # Save the model
    torch.save(best_model.state_dict(), 'audio_classification_model.pth')
    print("Model saved to 'audio_classification_model.pth'")
    
    # Run inferences on test spectrogram images
    print("\nRunning inferences on test spectrogram images:")
    model.eval()
    inference_dir = "./img_dataset/inferences"
    for spectrogram in os.listdir(inference_dir):
        if not spectrogram.endswith('.png'):
            continue
            
        try:
            # Load the spectrogram image
            img_path = os.path.join(inference_dir, spectrogram)
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            img_tensor = transform(image)
            
            # Convert the image tensor to a sequence for LSTM
            img_np = img_tensor.numpy().mean(axis=0)  # Average across color channels
            
            # Resize along the height dimension to match our input_size
            if img_np.shape[0] != input_size:
                resized = np.zeros((input_size, img_np.shape[1]))
                for i in range(input_size):
                    orig_i = i * img_np.shape[0] / input_size
                    floor_i = int(np.floor(orig_i))
                    ceil_i = min(floor_i + 1, img_np.shape[0] - 1)
                    alpha = orig_i - floor_i
                    resized[i] = img_np[floor_i] * (1 - alpha) + img_np[ceil_i] * alpha
                img_np = resized
            
            # Transpose to get time steps as the first dimension
            sequence = img_np.T  # Now shape is [W, input_size]
            
            # Pad or truncate to max_len
            if sequence.shape[0] < max_len:
                pad_width = max_len - sequence.shape[0]
                sequence = np.pad(sequence, ((0, pad_width), (0, 0)), mode='constant')
            else:
                sequence = sequence[:max_len, :]
            
            # Normalize
            sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
            
            # Convert to tensor and add batch dimension
            features = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            # Make prediction
            pred = model(features)
            if pred[0,0] < .5:
                label = 'cat'
            else:
                label = 'dog'
            print(f"for {spectrogram}, the prediction is {label}.")
        except Exception as e:
            print(f"Error processing {spectrogram}: {e}")
            continue

if __name__ == "__main__":
    main()
