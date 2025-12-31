import torch
import numpy as np
import os, csv, json
from tqdm import tqdm
from skimage import measure
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, jaccard_score

# This class deals with running inference on trained models - 
# running predictions, stitching tiles back together, cleaning up segmentation masks, recording evaluation metrics etc.
class TorchInference():
    def __init__(self, 
                 model, 
                 device, 
                 model_name=None,
                 test_loader=None, 
                 save_path=None):
        
        super().__init__()

        # Get params
        self.model = model.to(device)
        self.model_name = model_name
        self.test_loader = test_loader
        self.device = device
        self.save_path = save_path

        # Arrays to save stats
        self.test_metrics = {
            "accuracy": 0,
            "f1": 0,
            "precision": 0,
            "recall": 0,
            "roc": 0,
            "iou": 0
        }


    # We know that the test dataset has features stored in 500x500 tiles with 50x50 labels, \
    # so we need to reconstruct everything back into a single image for visualisation purposes.
    # From a numpy array, and the metadata, reconstruct the full image
    def reconstruct_tiles(self, array, meta_path):
        # array is of shape [N, H, W]
        H_tile = array.shape[-2]
        W_tile = array.shape[-1]

        # Obtain the number of rows and columns from reading the first line of metadata
        with open(meta_path, "r") as f:
            meta = json.loads(f.readline())
            # Metadata assumes tile sizes of 500
            nrows = meta["Hc"] // 500 
            ncols = meta["Wc"] // 500 

        # Dimensions of image
        H_img = H_tile * nrows
        W_img = W_tile * ncols

        # Create an empty numpy array to store this
        img = np.zeros((H_img, W_img))

        with open(meta_path, "r") as f:
            i = 0
            for line in f:
                meta = json.loads(line)

                # Metadata assumes tile sizes of 500, but this is not necessarily the case (label)
                row = meta["r"] // (500 // H_tile)
                col = meta["c"] // (500 // W_tile)

                img[row:row+H_tile, col:col+W_tile] = array[i]
                i += 1
        
        return img


    # Testing function
    # filter_water - whether to mask out water pixels in model predictions and label
    # per_event_threshold - whether to automatically calculate a probabilty threshold to determine flood extent based on model probibilities. 
    # If false, uses a default threshold of 0.5
    def test_model(self, filter_water=False, per_event_threshold=False):
        if self.test_loader is None:
            raise NotImplementedError()
        
        self.model.eval()

        # Used to store the model's output probabilties for each tile, preserving shape
        model_probs = []
        # Since we might be filtering out the true labels with water mask
        model_labels = []
        
        with torch.no_grad():
            for inputs, labels, water_mask in tqdm(self.test_loader, desc="Testing"):
                inputs, labels, water_mask = inputs.to(self.device), labels.to(self.device), water_mask.to(self.device)
                outputs = self.model(inputs)

                if filter_water == True:
                    probs = torch.sigmoid(outputs) * water_mask
                    labels = labels * water_mask
                else:
                    probs = torch.sigmoid(outputs)

                model_probs.append(probs.cpu())
                model_labels.append(labels.cpu())

        model_probs = torch.cat(model_probs, dim=0).cpu().numpy() # Shape: N, 25, 25
        model_labels = torch.cat(model_labels, dim=0).cpu().numpy() # Shape: N, 25, 25
        
        if per_event_threshold == True:
            # Simple threshold calculation
            threshold = model_probs.min() + (np.percentile(model_probs, 95) - model_probs.min()) * 0.80
        else:
            threshold = 0.5

        print("Threshold: " + str(threshold))

        # Calculate predicted flood areas using the threshold
        model_preds = (model_probs > threshold).astype(int)

        # Return model predictions and probabilities that can be used for plotting later, as well as masked labels
        return model_preds, model_probs, model_labels

    

    # Calculates evaluation metrics using binarised model predictions, model probabilities, and ground truth labels
    # model_preds, model_probs, and model_labels can be of any shape
    def calculate_metrics(self, model_preds, model_probs, model_labels):
        # Calculate metrics
        accuracy = 100 * (model_preds == model_labels).sum() / model_labels.size
        f1 = f1_score(model_labels.flatten(), model_preds.flatten())
        prec = precision_score(model_labels.flatten(), model_preds.flatten())
        recall = recall_score(model_labels.flatten(), model_preds.flatten())
        roc_auc = roc_auc_score(model_labels.flatten(), model_probs.flatten())
        iou = jaccard_score(model_labels.flatten(), model_preds.flatten())

        print(f'Accuracy: {accuracy:.2f} %')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {prec:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')
        print(f'IOU: {iou:.4f}')

        self.test_metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": prec,
            "recall": recall,
            "roc": roc_auc,
            "iou": iou
        }
    

    # Save testing metrics
    def save_metrics(self):
        if self.save_path is None or self.model_name is None:
            raise NotImplementedError()
        
        with open(os.path.join(self.save_path, f"{self.model_name}_metrics.json"), "w") as f:
            json.dump(self.test_metrics, f)


    # Function that runs inference on an entire test event, stitches tiles together, and calculates metrics
    def infer_full_test_event(self, meta_path, filter_water=False, per_event_threshold=False):
        if self.test_loader == None:
            return NotImplementedError()
        
        # Get the model predictions
        # Each array is of size N, H, W
        model_preds, model_probs, model_labels = self.test_model(filter_water, per_event_threshold)
        return self.post_process_predictions(model_preds, model_probs, model_labels, meta_path)

    # Stitch tiles together, clean up mask, and calculate metrics
    def post_process_predictions(self, model_preds, model_probs, model_labels, meta_path):
        # Next, stitch the tiles together
        model_preds = self.reconstruct_tiles(model_preds, meta_path)
        model_probs = self.reconstruct_tiles(model_probs, meta_path)
        model_labels = self.reconstruct_tiles(model_labels, meta_path)

        # Segmentation clean up
        # First, only keep areas with a minimum specified area, regardless of probbility 
        labelled = measure.label(model_preds, connectivity=2)
        props = measure.regionprops(labelled)
        keep_labels = {p.label for p in props if p.area >= 500}
        model_preds = np.isin(labelled, list(keep_labels))

        # Next, we keep only large detected areas that are connected together
        labelled = measure.label(model_preds, connectivity=2)
        props = measure.regionprops(labelled)
        keep_labels = {p.label for p in props if p.area >= 3000}
        area_mask = (np.isin(labelled, list(keep_labels)) != 0)

        # Keep areas that the model is very confident in, so we can still allow small areas to be kept
        probs_mask = model_probs > 0.9

        # Mask out small areas only if the model isn't very confident in it
        model_preds = model_preds * (probs_mask | area_mask)

        # Calculate metrics
        self.calculate_metrics(model_preds, model_probs, model_labels)
        self.save_metrics()

        # Return arrays for potential plotting later on
        return model_preds, model_probs, model_labels
    