import torch
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import os, json
import torch.nn.functional as F

class TorchFloodDataset(Dataset):
    """
    Returns:
      X: float32 tensor [5, 500, 500]  
      y: int64   tensor [50, 50]     
    """
    def __init__(self, X, y, transform=None, do_augment=True):
        self.X = X
        self.Y = y
        self.N = self.X.shape[0]

        # Create a water mask to mask out water bodies, used in inference
        # Will be "1" if water bodies are not present, "0" if there are water bodies
        # The mask should have the same size as label 
        
        # In land cover, 210 means water body
        self.water_mask = self.X[:, 3] == 210 # N, 500, 500
        self.water_mask = torch.from_numpy(self.water_mask.astype(np.float32))
        self.water_mask = F.avg_pool2d(self.water_mask, kernel_size=10, stride=10) # N, 50, 50
        self.water_mask = self.water_mask < 0.9

        # Just for checking
        assert self.Y.shape[0] == self.N

        self.transform = transform

        # Add some basic data augmentation to make models more robust
        self.do_augment = do_augment
        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip()
        ])
        
    def __len__(self): 
        return self.N

    def __getitem__(self, i):
        x = self.X[i]  
        y = self.Y[i]  
        mask = self.water_mask[i] 

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = prepare_data(x)

        if self.transform is not None:
            x = self.transform(x)

        # Apply data augmentation
        if self.do_augment:
            x, y = self.augment(x, y)

        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64)), mask.int()


"""
Function to perform feature engineering, standardisation, etc.
X is a single data sample of size (5, 500, 500)
"""
def prepare_data(X):
    out = np.copy(X)

    # Transform DEM and distance to water in a log scale to reduce the magnitude of values
    # Transform categotical land use values to be of lower magnitude
    # Transform slope to be bounded between 0 and 1
    # Signed log for DEM (since DEM can be negative)
    out[0] = np.sign(X[0]) * np.log(1 + np.abs(X[0]))
    # Regular log for distance to water
    out[4] = np.log(1 + X[4])

    # Next, we need to transform the categorical lc values to be of a lower magnitude as well
    out[3] = X[3] / 100

    # Finally, we need to map slope values to be between 0 and 1
    out[1] = X[1] / 90

    return out
    

"""
Function to initialise train and val dataloaders
"""
def init_train_val_loaders(X_path, y_path, batch_size, valid_split, seed, save_dir):
    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    # Get indices for train samples and test samples
    N = X.shape[0]               
    idx = np.arange(N)           
    train_idx, test_idx = train_test_split(idx, test_size=valid_split, random_state=seed, shuffle=True)   

    # From the samples used for training, calculate mean and std
    sums = np.zeros(X.shape[1], dtype=np.float64)
    sumsq = np.zeros(X.shape[1], dtype=np.float64)
    counts = np.zeros(X.shape[1], dtype=np.int64)

    print("Calculating mean and std...")
    for i in tqdm(train_idx):
        # Calculate processed features on the fly to save memory
        sample = prepare_data(X[i])

        for c in range(X.shape[1]):
            feature = sample[c]
            finite = np.isfinite(feature)

            # No NaN values
            if finite.any():
                vals = feature[finite].astype(np.float64)
                sums[c]  += vals.sum()
                sumsq[c] += (vals * vals).sum()
                counts[c] += vals.size

    means = (sums / np.maximum(counts, 1)).tolist()
    variances = (sumsq / np.maximum(counts, 1) - np.square(sums / np.maximum(counts, 1)))
    stds = np.sqrt(np.maximum(variances, 0)).tolist()

    # Dump the mean and std to a json file
    with open(os.path.join(save_dir, f"proc_data_mean.json"), "w") as f:
        json.dump(means, f)
    with open(os.path.join(save_dir, f"proc_data_std.json"), "w") as f:
        json.dump(stds, f)

    # Apply standardisation
    transform = v2.Normalize(mean=means, std=stds)

    # Build dataset
    dataset = TorchFloodDataset(X=X, y=y, transform=transform)
    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, test_idx)

    # Build dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # Calculate loss weight
    # Calculate the class imbalance so we can perform weighted cross entropy loss
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    loss_weight = n_neg / n_pos

    return train_loader, val_loader, loss_weight

