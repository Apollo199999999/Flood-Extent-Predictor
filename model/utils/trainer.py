import torch
import numpy as np
import os, csv, json
from tqdm import tqdm
from pathlib import Path

# This class deals with training, validation, and logging train/valid losses
class TorchTrainer():
    def __init__(self, 
                 model, 
                 device, 
                 model_name=None,
                 train_loader=None, 
                 val_loader=None, 
                 criterion=None, 
                 optimizer=None, 
                 scheduler=None, 
                 num_epochs=None, 
                 grad_accum_steps=1,
                 save_path=None):
        
        super().__init__()

        # Get params
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        self.save_path = save_path

        # Arrays to save stats
        self.train_losses = []
        self.val_losses = []
        

    # Training function
    def train_and_validate_model(self):
        if self.train_loader is None or self.val_loader is None or self.criterion is None or self.optimizer is None or self.num_epochs is None:
            raise NotImplementedError()
        
        if self.save_path is None or self.model_name is None:
            raise NotImplementedError()
        
        best_loss = 999

        for epoch in range(self.num_epochs):
            # Train model
            self.model.train()
            self.optimizer.zero_grad()

            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for i, (inputs, labels, *_) in enumerate(progress_bar):
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Show loss and lr for current batch
                lr = self.optimizer.param_groups[0]['lr']
                running_loss += loss.item() * inputs.size(0)
                progress_bar.set_postfix(loss=loss.item(), lr=lr)

                # Implement gradient accumulation to simulate a larger batch size
                loss = loss / self.grad_accum_steps 
                loss.backward()

                # Wait for several backward steps
                if (i + 1) % self.grad_accum_steps == 0 or i + 1 == len(self.train_loader):      
                    # Now we can do an optimizer step       
                    self.optimizer.step()                        
                    self.optimizer.zero_grad()

            # Step the LR scheduler at the end of each epoch
            if self.scheduler:
                self.scheduler.step()

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

            # Validate model and print loss
            val_loss = self.validate_model()
            self.val_losses.append(val_loss)

            # If val loss is lowest, save model
            if val_loss < best_loss:
                print("Best validation loss! Saving model...")
                best_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{self.model_name}_best_val_loss.pt"))


    # Validate model and print val loss
    def validate_model(self):
        if self.val_loader is None or self.criterion is None:
            raise NotImplementedError()
        
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels, *_ in tqdm(self.val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            val_loss = running_loss / len(self.val_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}")

            return val_loss
        
    
    def save_model(self):
        if self.save_path is None or self.model_name is None:
            raise NotImplementedError()
        
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{self.model_name}.pt"))
    
    # If load_bet is true, will attempt to load model weights with lowest validation loss
    def get_model(self, load_best=True):
        if load_best == False:
            return self.model
        
        # Check if best model weights exist
        weights_path = os.path.join(self.save_path, f"{self.model_name}_best_val_loss.pt")
        
        if Path(weights_path).is_file():
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))
            return self.model
        else:
            print("Unable to load best model weights, returning current model instead...")
            return self.model

    
    # Save train loss, val loss
    def save_losses(self):
        if self.save_path is None or self.model_name is None:
            raise NotImplementedError()
        
        with open(os.path.join(self.save_path, f"{self.model_name}_losses.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss"])
            for i, (tr, va) in enumerate(zip(self.train_losses, self.val_losses), start=1):
                w.writerow([i, tr, va])        
           