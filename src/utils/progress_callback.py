
""" 
progress_callback.py
Utility callback for displaying training progress with a progress bar.

Brendan Dileo, August 2025
"""

from transformers import TrainerCallback
from tqdm import tqdm

class ProgressCallback(TrainerCallback):
    
    def on_train_begin(self, args, state, control, **kwargs):
        """ Initialize progress bar at the start of training """
        self.epochs = int(args.num_train_epochs)
        print(f"Training for {self.epochs} epochs...")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """ Initialize progress bar at the start of each epoch """
        self.pbar = tqdm(total=state.max_steps, desc=f"Epoch {state.epoch:.0f}/{args.num_train_epochs}")
        
    def on_step_end(self, args, state, control, **kwargs):
        """ Update progress bar at the end of each step """
        self.pbar.update(1)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """ Close progress bar at the end of each epoch """
        self.pbar.close()