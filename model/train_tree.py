# TODO
# 1. FRONTEND
# 2. ViT STRUCTURE WRITEUP
# 3. TREE WRITEUP






import random
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import os
import joblib

from sklearn.tree import DecisionTreeClassifier

from utils.dataset import init_train_val_loaders
from utils.inference import TorchInference
from architectures.unet import BaseUNet

# Fix random seed for determinism
SEED = 42

random.seed(SEED)                  # Python built-in random
np.random.seed(SEED)               # NumPy
torch.manual_seed(SEED)            # PyTorch (CPU)
torch.cuda.manual_seed(SEED)       # PyTorch (single GPU)
torch.cuda.manual_seed_all(SEED)   # PyTorch (all GPUs)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SPLIT = 0.8
BATCH_SIZE = 16
MODEL_NAME = "base_tree"
PATCH_SIZE = 10

# Directories
X_PATH = "/kaggle/processed-tiles/train_X.npy"
Y_PATH = "/kaggle/processed-tiles/train_Y.npy"
STATS_PATH = "/kaggle/processed-tiles/train_stats.json"
META_PATH = "/kaggle/processed-tiles/train_meta.jsonl"
SAVE_DIR = "/kaggle/working/Flood-Impact-Evaluator/model/output"

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Define dataset variables
train_loader, val_loader, loss_weight = init_train_val_loaders(X_path=X_PATH, 
                                                               y_path=Y_PATH, 
                                                               batch_size=BATCH_SIZE, 
                                                               valid_split=(1-TRAIN_SPLIT), 
                                                               seed=SEED,
                                                               save_dir=SAVE_DIR)

# Set model
tree_model = DecisionTreeClassifier(random_state=SEED, max_depth=20)

# Variables to store processed train and val data
X_train = []
y_train = []
X_val = []
y_val = []

progress_bar = tqdm(train_loader, desc=f"Processing training data")
for i, (inputs, labels, *_) in enumerate(progress_bar):
    inputs = torch.nn.functional.avg_pool2d(inputs, kernel_size=10, stride=10)
    X_train.append(inputs)
    y_train.append(labels)

progress_bar = tqdm(val_loader, desc=f"Processing validation data")
for i, (inputs, labels, *_) in enumerate(progress_bar):
    inputs = torch.nn.functional.avg_pool2d(inputs, kernel_size=10, stride=10)
    X_val.append(inputs)
    y_val.append(labels)

X_train = torch.cat(X_train, dim=0).permute(0, 2, 3, 1).reshape(-1, 5).numpy()
y_train = torch.cat(y_train, dim=0).reshape(-1).numpy()
X_val = torch.cat(X_val, dim=0).permute(0, 2, 3, 1).reshape(-1, 5).numpy()
y_val = torch.cat(y_val, dim=0).reshape(-1).numpy()

print(f"Training: {X_train.shape}")
print(f"Validation: {X_val.shape}")

print("Training Decision Tree...")
tree_model.fit(X_train, y_train)

# Evaluate
train_pred = tree_model.predict(X_train)
train_prob = tree_model.predict_proba(X_train)[:, 1]

val_pred = tree_model.predict(X_val)
val_prob = tree_model.predict_proba(X_val)[:, 1]

# We can define an inference class and pass in a dummy torch model to help us with evaluation metrics
unet_model = BaseUNet(5, 1)
inference = TorchInference(model=unet_model, 
                           device=DEVICE, 
                           model_name=MODEL_NAME, 
                           test_loader=val_loader, 
                           save_path=SAVE_DIR)

print("Evaluation on training dataset:")
inference.calculate_metrics(train_pred, train_prob, y_train)

print("Evaluation on validation dataset:")
inference.calculate_metrics(val_pred, val_prob, y_val)

# Save everything
inference.save_metrics()

# Save the trained model (trainer.save_model())
model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pkl")
joblib.dump(tree_model, model_path)
print(f"Model saved to {model_path}")

print("Training completed and all files saved!")
