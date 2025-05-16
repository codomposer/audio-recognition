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
import random
from model import LSTM, train_model, eval_model


# Define dataset class
class SpectrogramSequenceDataset(Dataset):
    def __init__(
        self, csv_file, input_size=40, max_len=100, transform=None, augment=False
    ):
        """
        Args:
            csv_file (string): Path to the csv file with spectrogram image paths and labels.
            input_size (int): Number of features to extract from each column of the spectrogram
            max_len (int): Maximum length of time steps (will pad/truncate)
            transform (callable, optional): Optional transform to be applied on the image
            augment (bool): Whether to apply data augmentation
        """
        self.data_frame = pd.read_csv(csv_file)
        self.input_size = input_size
        self.max_len = max_len
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the spectrogram image path
        img_path = self.data_frame.iloc[idx, 1]  # image_location column

        # Load the spectrogram image
        image = Image.open(img_path).convert("RGB")

        # Apply data augmentation if enabled (only for training)
        if self.augment:
            # Apply random augmentations
            augmented_image = self.apply_augmentation(image)
            image = augmented_image

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
            sequence = np.pad(sequence, ((0, pad_width), (0, 0)), mode="constant")
        else:
            # Truncate to max_len
            sequence = sequence[: self.max_len, :]

        # Convert to tensor
        features = torch.FloatTensor(sequence)

        # Normalize the features
        features = (features - features.mean()) / (features.std() + 1e-8)

        # Convert target to numeric (0 for cat, 1 for dog)
        target = 0 if self.data_frame.iloc[idx, 2] == "cat" else 1

        return features, target

    def apply_augmentation(self, image):
        """
        Apply various augmentation techniques to the spectrogram image

        Args:
            image (PIL.Image): Input spectrogram image

        Returns:
            PIL.Image: Augmented image
        """
        # Create a list of possible augmentations
        augmentations = [
            # Time shifting (horizontal shift)
            lambda img: transforms.functional.affine(
                img,
                angle=0,
                translate=(int(img.width * 0.1 * (random.random() - 0.5)), 0),
                scale=1.0,
                shear=0,
            ),
            # Frequency shifting (vertical shift)
            lambda img: transforms.functional.affine(
                img,
                angle=0,
                translate=(0, int(img.height * 0.1 * (random.random() - 0.5))),
                scale=1.0,
                shear=0,
            ),
            # Time stretching/compression (horizontal scaling)
            lambda img: transforms.functional.affine(
                img,
                angle=0,
                translate=(0, 0),
                scale=1.0 + 0.1 * (random.random() - 0.5),
                shear=0,
            ),
            # Frequency masking (horizontal bars)
            lambda img: self.frequency_masking(img),
            # Time masking (vertical bars)
            lambda img: self.time_masking(img),
            # Small random rotation
            lambda img: transforms.functional.rotate(img, angle=random.uniform(-5, 5)),
            # Slight brightness/contrast adjustment
            lambda img: transforms.functional.adjust_brightness(
                img, brightness_factor=random.uniform(0.8, 1.2)
            ),
            lambda img: transforms.functional.adjust_contrast(
                img, contrast_factor=random.uniform(0.8, 1.2)
            ),
        ]

        # Randomly select 2-3 augmentations to apply
        num_augmentations = random.randint(1, 3)
        selected_augmentations = random.sample(augmentations, num_augmentations)

        # Apply the selected augmentations
        augmented_image = image
        for augmentation in selected_augmentations:
            augmented_image = augmentation(augmented_image)

        return augmented_image

    def frequency_masking(self, image):
        """
        Apply frequency masking (horizontal bars) to the spectrogram
        """
        img_np = np.array(image)
        height, width = img_np.shape[0], img_np.shape[1]

        # Apply 1-2 frequency masks
        num_masks = random.randint(1, 2)
        for _ in range(num_masks):
            f_height = random.randint(
                1, int(height * 0.15)
            )  # Mask up to 15% of frequency bins
            f_start = random.randint(0, height - f_height)

            # Create the mask
            mask_value = random.randint(0, 100)  # Random dark value
            img_np[f_start : f_start + f_height, :, :] = mask_value

        return Image.fromarray(img_np)

    def time_masking(self, image):
        """
        Apply time masking (vertical bars) to the spectrogram
        """
        img_np = np.array(image)
        height, width = img_np.shape[0], img_np.shape[1]

        # Apply 1-2 time masks
        num_masks = random.randint(1, 2)
        for _ in range(num_masks):
            t_width = random.randint(
                1, int(width * 0.15)
            )  # Mask up to 15% of time steps
            t_start = random.randint(0, width - t_width)

            # Create the mask
            mask_value = random.randint(0, 100)  # Random dark value
            img_np[:, t_start : t_start + t_width, :] = mask_value

        return Image.fromarray(img_np)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    # First shuffle the data with a fixed random seed for reproducibility
    train_csv = pd.read_csv("./img_dataset/train/train.csv")
    test_csv = pd.read_csv("./img_dataset/test/test.csv")

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_csv = train_csv.sample(frac=1, random_state=42)
    test_csv = test_csv.sample(frac=1, random_state=42)

    # Save shuffled CSVs if needed
    train_csv.to_csv("./img_dataset/train/train.csv", index=False)
    test_csv.to_csv("./img_dataset/test/test.csv", index=False)

    # Create transforms
    base_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Define dataset parameters
    input_size = 40  # Number of features per time step
    max_len = 100  # Maximum number of time steps

    # Create training dataset with augmentation
    train_dataset = SpectrogramSequenceDataset(
        csv_file="./img_dataset/train/train.csv",
        input_size=input_size,
        max_len=max_len,
        transform=base_transform,
        augment=True,  # Enable augmentation for training
    )

    # Create test dataset without augmentation
    test_dataset = SpectrogramSequenceDataset(
        csv_file="./img_dataset/test/test.csv",
        input_size=input_size,
        max_len=max_len,
        transform=base_transform,
        augment=False,  # No augmentation for testing
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model with improved parameters
    model = LSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.3,  # Add dropout for regularization
    )
    model = model.to(device)
    print(model)

    # Train the model with improved parameters
    print("Starting model training with data augmentation...")
    best_model, training_stats = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=60,
        lr=0.001,
        weight_decay=1e-5,  # L2 regularization
    )

    # Evaluate the final model
    final_accuracy, final_loss = eval_model(best_model, test_loader)
    print(f"Final model accuracy: {final_accuracy:.4f}")
    print(f"Final model loss: {final_loss:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(
        training_stats["epoch_nums"],
        training_stats["training_loss"],
        label="Training Loss",
    )
    plt.plot(
        training_stats["epoch_nums"],
        training_stats["validation_loss"],
        label="Validation Loss",
    )
    plt.title("Loss Curve with Data Augmentation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("loss_curve_augmented.png")
    print("Loss curve saved to 'loss_curve_augmented.png'")

    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(
        training_stats["epoch_nums"],
        training_stats["validation_acc"],
        label="Validation Accuracy",
    )
    plt.title("Validation Accuracy with Data Augmentation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.0])
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig("accuracy_curve_augmented.png")
    print("Accuracy curve saved to 'accuracy_curve_augmented.png'")

    # Save the model
    torch.save(best_model.state_dict(), "audio_classification_model_augmented.pth")
    print("Model saved to 'audio_classification_model_augmented.pth'")

    # Save training stats for comparison
    import json

    with open("training_stats_augmented.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        stats_to_save = {
            "epoch_nums": training_stats["epoch_nums"],
            "training_loss": [float(x) for x in training_stats["training_loss"]],
            "validation_loss": [float(x) for x in training_stats["validation_loss"]],
            "validation_acc": [float(x) for x in training_stats["validation_acc"]],
            "best_accuracy": float(training_stats["best_accuracy"]),
            "training_time": float(training_stats["training_time"]),
        }
        json.dump(stats_to_save, f, indent=4)
    print("Training stats saved to 'training_stats_augmented.json'")

    # Run inferences on test spectrogram images
    print("\nRunning inferences on test spectrogram images:")
    model.eval()
    inference_dir = "./img_dataset/inferences"
    for spectrogram in os.listdir(inference_dir):
        if not spectrogram.endswith(".png"):
            continue

        try:
            # Load the spectrogram image
            img_path = os.path.join(inference_dir, spectrogram)
            image = Image.open(img_path).convert("RGB")

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
                sequence = np.pad(sequence, ((0, pad_width), (0, 0)), mode="constant")
            else:
                sequence = sequence[:max_len, :]

            # Normalize
            sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)

            # Convert to tensor and add batch dimension
            features = torch.FloatTensor(sequence).unsqueeze(0).to(device)

            # Make prediction
            pred = model(features)
            if pred[0, 0] < 0.5:
                label = "cat"
            else:
                label = "dog"
            print(f"for {spectrogram}, the prediction is {label}.")
        except Exception as e:
            print(f"Error processing {spectrogram}: {e}")
            continue


if __name__ == "__main__":
    main()
