from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn

class TrainingLogger():
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def log_training_step(self, loss: float, step: int, lr: float):
        self.writer.add_scalar("Train/Loss", loss, step)
        self.writer.add_scalar("Train/Learning_Rate", lr, step)
        
    def log_weight_histograms(self, model: nn.Module, step: int):
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param.data.cpu().numpy(), step)
            
    def log_validation_step(self, val_loss: float, step: int):
        self.writer.add_scalar("Validation/Loss", val_loss, step)
        
    # TODO: Add attention maps logging
    
    def close(self):
        self.writer.close()