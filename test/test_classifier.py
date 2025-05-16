#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the cat sound classifier.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.classify_cat_sound import main as classify_cat_sound

def test_classifier():
    # Define paths
    audio_file = "../data/cats_dogs/train/cat/cat_1.wav"
    model_path = "../audio_classification_model_augmented.pth"
    
    # Check if files exist
    print(f"Audio file exists: {os.path.exists(audio_file)}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    # Run classification
    print(f"Running classification on {audio_file} with model {model_path}")
    result = classify_cat_sound(audio_file, model_path)
    
    # Print result
    print(f"Classification result: {'CAT' if result else 'NOT CAT'}")
    
    return result

if __name__ == "__main__":
    test_classifier()
