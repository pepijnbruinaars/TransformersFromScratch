from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn

class TrainingLogger():
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def log_training_step(self, loss: float, step: int, lr: float):
        """Log training loss and learning rate."""
        self.writer.add_scalar("Train/Loss", loss, step)
        self.writer.add_scalar("Train/Learning_Rate", lr, step)
    
    def log_gradient_norm(self, grad_norm: float, step: int):
        """Log gradient norm."""
        self.writer.add_scalar("Train/Grad_Norm", grad_norm, step)
        
    def log_weight_histograms(self, model: nn.Module, step: int):
        """Log weight histograms for all parameters."""
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param.data.cpu().numpy(), step)
            
    def log_validation_step(self, val_loss: float, step: int):
        """Log validation loss."""
        self.writer.add_scalar("Validation/Loss", val_loss, step)
        
    def flush(self):
        """Flush pending events to TensorBoard."""
        self.writer.flush()    
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
