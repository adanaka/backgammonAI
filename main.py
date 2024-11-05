import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from model import Model

# Set up argument parsing with argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='If true, test against a random strategy.')
parser.add_argument('--play', action='store_true', help='If true, play against a trained TD-Gammon strategy.')
parser.add_argument('--restore', action='store_true', help='If true, restore a checkpoint before training.')
args = parser.parse_args()

model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = 'checkpoints/final_2000'

# Create directories if they do not exist
os.makedirs(model_path, exist_ok=True)
os.makedirs(summary_path, exist_ok=True)
# Ensure checkpoint path directory creation
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

if __name__ == '__main__':
    # Create a TensorFlow 2.x Model instance
    model = Model(model_path, summary_path, checkpoint_path, restore=args.restore)

    # Define ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path + '_checkpoint.weights.h5'),
        save_weights_only=True,  # Set to True if you only want to save weights
        save_best_only=True,  # Save only the best model
        verbose=1
    )

    # Perform actions based on command-line arguments
    if args.play:
        model.play(100)
    else:
        model.train()
