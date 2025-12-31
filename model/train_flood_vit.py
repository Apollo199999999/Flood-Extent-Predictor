# This is in the form of a Python script, rather than a Jupyter notebook, so we can run this on Kaggle and use their GPUs

# Imports
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import init_train_val_loaders
from utils.trainer import TorchTrainer
from utils.inference import TorchInference
from architectures.transformers import FloodViT

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
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_ACCUM_STEPS = 4
MODEL_NAME = "flood_vit"

# Directories
X_PATH = "/kaggle/processed-tiles/train_X.npy"
Y_PATH = "/kaggle/processed-tiles/train_Y.npy"
STATS_PATH = "/kaggle/processed-tiles/train_stats.json"
META_PATH = "/kaggle/processed-tiles/train_meta.jsonl"
SAVE_DIR = "/kaggle/working/Flood-Impact-Evaluator/model/output"

# X_PATH = "E:\\train_X.npy"
# Y_PATH = "E:\\train_Y.npy"
# STATS_PATH = "E:\\train_stats.json"
# META_PATH = "E:\\train_meta.jsonl"
# SAVE_DIR = "E:\\experiments"

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Get dataloaders
train_loader, val_loader, loss_weight = init_train_val_loaders(X_path=X_PATH, 
                                                               y_path=Y_PATH, 
                                                               batch_size=BATCH_SIZE, 
                                                               valid_split=(1-TRAIN_SPLIT), 
                                                               seed=SEED,
                                                               save_dir=SAVE_DIR)

# Retrieve model
vit_model = FloodViT(embed_dim=256, 
                     depth=4, 
                     num_heads=4, 
                     mlp_dim=768,
                     dropout=0.1)

# Define optimizer, lr scheduler, loss fn
optimizer_vit = optim.AdamW(vit_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler_vit = optim.lr_scheduler.CosineAnnealingLR(optimizer_vit, T_max=NUM_EPOCHS)

# Use weighted CE loss for segmentation
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([loss_weight]).to(DEVICE))

# Train!
trainer = TorchTrainer(
    model=vit_model,
    device=DEVICE,
    model_name=MODEL_NAME,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer_vit,
    scheduler=scheduler_vit,
    num_epochs=NUM_EPOCHS,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    save_path=SAVE_DIR
)

trainer.train_and_validate_model()

# Define an inference class for to test on valid set
inference = TorchInference(model=trainer.get_model(), 
                           device=DEVICE, 
                           model_name=MODEL_NAME, 
                           test_loader=val_loader, 
                           save_path=SAVE_DIR)

# Test on metrics using val_loader
model_preds, model_probs, model_labels = inference.test_model()
inference.calculate_metrics(model_preds, model_probs, model_labels)

# Save everything
trainer.save_model()
trainer.save_losses()
inference.save_metrics()
