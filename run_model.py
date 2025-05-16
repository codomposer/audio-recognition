#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the CNN model on the audio classification dataset.
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
from model import CNN, train_model, eval_model

# Define dataset class
class AudioSpectrogramDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.data_frame.iloc[idx, 1]  # image_location column
        image = Image.open(img_path).convert('RGB')
        
        # Convert target to numeric (0 for cat, 1 for dog)
        target = 0 if self.data_frame.iloc[idx, 2] == 'cat' else 1
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    # First shuffle the data with a fixed random seed for reproducibility
    train_csv = pd.read_csv('./img_dataset/train/train.csv')
    test_csv = pd.read_csv('./img_dataset/test/test.csv')
    
    train_csv = train_csv.sample(frac=1, random_state=42)
    test_csv = test_csv.sample(frac=1, random_state=42)
    
    # Save shuffled CSVs if needed
    train_csv.to_csv('./img_dataset/train/train.csv', index=False)
    test_csv.to_csv('./img_dataset/test/test.csv', index=False)
    
    train_dataset = AudioSpectrogramDataset(
        csv_file='./img_dataset/train/train.csv',
        transform=transform
    )
    
    test_dataset = AudioSpectrogramDataset(
        csv_file='./img_dataset/test/test.csv',
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
    model = CNN()
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
    
    # Run inferences on test images
    print("\nRunning inferences on test images:")
    model.eval()
    for spectogram in os.listdir("./img_dataset/inferences"):
        try:
            image = Image.open(f"./img_dataset/inferences/{spectogram}")
            
            # unsqueeze converts transform.(image) from 3d (3,256,256) to 4d (1,3,256,256).
            # you can think of '1' as a batch of 1 image since the model was trained with mini-batches.
            ts = transform(image).unsqueeze(0).to(device)
     
            pred = model(ts)
            if pred[0,0] < .5:
                label = 'cat'
            else:
                label = 'dog'
            print(f"for {spectogram}, the prediction is {label}.")
        except Exception as e:
            print(f"Error processing {spectogram}: {e}")
            continue

if __name__ == "__main__":
    main()
