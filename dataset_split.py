import pandas as pd
import numpy as np
import os

# Set random seed for reproducible results
SEED = 42
np.random.seed(SEED)

# Create output directory if it doesn't exist
os.makedirs('dataset', exist_ok=True)

# Read the complete dataset
print("Reading the complete dataset...")
df = pd.read_csv('music_scores_with_track_count.csv')
print(f"Dataset loaded successfully with {len(df)} samples")

# Check if the dataset has enough samples
if len(df) < 1100:
    raise ValueError(f"Dataset only has {len(df)} samples, which is insufficient to create 1000 training samples and 100 test samples")

# Randomly select 1000 unique samples for training set
print("Selecting training samples...")
train_indices = np.random.choice(df.index, size=1000, replace=False)
train_set = df.loc[train_indices]

# Randomly select 100 unique samples for test set from remaining data
print("Selecting test samples...")
remaining_indices = df.index.difference(train_indices)
test_indices = np.random.choice(remaining_indices, size=100, replace=False)
test_set = df.loc[test_indices]

# Save the datasets
print("Saving datasets...")
train_set.to_csv('dataset/mini_train.csv', index=False)
test_set.to_csv('dataset/mini_test.csv', index=False)

# Print summary information
print("\n" + "="*50)
print("DATASET CREATION COMPLETED SUCCESSFULLY")
print("="*50)
print(f"Training set size: {len(train_set)} samples")
print(f"Test set size: {len(test_set)} samples")
print(f"Total unique samples: {len(train_set) + len(test_set)}")
print(f"Training and test sets overlap: {len(set(train_indices) & set(test_indices)) > 0}")
print(f"Training set saved to: dataset/mini_train.csv")
print(f"Test set saved to: dataset/mini_test.csv")
print("="*50)