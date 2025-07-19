import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Step 1: Load the model from TensorFlow Hub
print("Loading YAMNet...")
model = hub.load("https://tfhub.dev/google/yamnet/1")
print("✅ Model loaded successfully!")

# Step 2: Create a dummy waveform (1 second of silence at 16kHz)
print("Generating dummy audio...")
dummy_waveform = np.zeros(16000, dtype=np.float32)  # 1 second of silence

# Step 3: Run the dummy audio through YAMNet
print("Running inference...")
scores, embeddings, spectrogram = model(dummy_waveform)

# Step 4: Convert TensorFlow tensor to NumPy array
scores_np = scores.numpy()

# Step 5: Get class index with highest confidence
top_class_index = np.argmax(np.mean(scores_np, axis=0))

# Step 6: Load class names from the YAMNet CSV
import pandas as pd
csv_url = "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv"
class_names = pd.read_csv(csv_url)['display_name'].to_list()

# Step 7: Print result
print(f"Top class: {class_names[top_class_index]}")
