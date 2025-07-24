# Training Data

This folder is for organizing training data and utilities for improving the gong detection model.

## Structure

- `samples/` - Training audio samples (positive and negative examples)
- `scripts/` - Data processing and training scripts
- `models/` - Trained model checkpoints and configurations

## Usage

When you're ready to train the gong detector:

1. Add your training samples to the `samples/` folder
2. Use the scripts in `scripts/` to process and prepare the data
3. Train your model and save checkpoints to `models/`

## File Naming Convention

- Positive samples: `gong_positive_*.wav`
- Negative samples: `gong_negative_*.wav`
- Background noise: `background_*.wav`
