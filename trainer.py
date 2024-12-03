import torch
import torch.nn as nn
from torch.optim import AdamW
import logging
from pathlib import Path
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer:
    def __init__(self, model: torch.nn.Module, config: Any, checkpoint_dir: Path) -> None:
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._initialize_training_components()
        self.writer = SummaryWriter(log_dir="runs")
        
    def _initialize_training_components(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def save_checkpoint(self, epoch: int, loss: float) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
        
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        self.logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
        
    def train(self, data_processor: Any) -> None:
        self.model.train()
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Starting epoch {epoch}")
            epoch_loss = self._train_epoch(data_processor, epoch)  # Pass epoch to method
            
            # Log metrics
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(epoch, epoch_loss)
                
    def _train_epoch(self, data_processor: Any, epoch: int) -> float:  # Add epoch parameter
        self.model.train()
        train_dataloader = data_processor.create_dataloader("train.pt")
        
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward pass
            logits = self.model(input_ids)  # Shape: [batch_size, seq_len, vocab_size]
            
            # Reshape for cross entropy
            B, S, V = logits.shape
            logits = logits.view(B * S, V)
            labels = labels.view(-1)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if (step + 1) % self.config.logging_steps == 0:
                self.logger.info(f"Epoch {epoch}, Step {step+1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        self.logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        return avg_loss