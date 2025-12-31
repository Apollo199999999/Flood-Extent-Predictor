# Use this script to ensure your model produces the correct output before pushing to GitHub/Kaggle

# Imports
import random
import numpy as np
import torch
import torch.nn as nn

from architectures.transformers import FloodViT
from architectures.unet import BaseUNet

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Retrieve model
model = FloodViT(embed_dim=256, 
                     depth=4, 
                     num_heads=4, 
                     mlp_dim=768,
                     dropout=0.1)
# model = BaseUNet(5, 1)
print(model)

# Test data shape
test_data = torch.randn(2, 5, 500, 500, device=DEVICE)
print(test_data.shape)

test_output = model(test_data)
print(test_output.shape)
