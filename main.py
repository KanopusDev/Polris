import os
import torch
import logging
from pathlib import Path
from functools import partial
from typing import Dict
import yaml
import psutil
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from models.transformer import EnhancedCodeTransformer
from models.tokenizer import EnhancedCodeTokenizer
from data.data_loader import GitHubDataLoader, CodeDataLoader
from models.conversation_manager import ConversationManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.setup_memory_management()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )

    def setup_memory_management(self):
        """Configure memory settings for CPU training."""
        # Set memory fraction to use
        mem = psutil.virtual_memory()
        self.max_memory = int(mem.total * 0.8)  # Use 80% of available RAM
        
        # Enable memory efficient settings
        torch.backends.cpu.memory_efficient_fp16 = True
        torch.backends.cpu.memory_efficient_linear = True
        
        # Set number of threads based on available CPU cores
        torch.set_num_threads(psutil.cpu_count() // 2)  # Use half of available cores
        
    def create_dataloaders(self, code_loader: CodeDataLoader, tokenizer: EnhancedCodeTokenizer):
        """Create memory-efficient dataloaders."""
        try:
            # Load data from multiple sources
            all_code_data = code_loader.load_data(
                sources=['github', 'codeparrot', 'codenet']
            )
            
            # Subsample if needed
            if len(all_code_data) > self.config['max_samples']:
                all_code_data = np.random.choice(
                    all_code_data, 
                    size=self.config['max_samples'], 
                    replace=False
                ).tolist()
            
            # Create train/val split
            train_size = int(0.9 * len(all_code_data))
            train_data = all_code_data[:train_size]
            val_data = all_code_data[train_size:]
            
            # Create dataloaders
            train_loader = DataLoader(
                train_data,
                batch_size=self.config['batch_size'],
                sampler=RandomSampler(train_data),
                num_workers=2,
                pin_memory=False
            )
            
            val_loader = DataLoader(
                val_data,
                batch_size=self.config['batch_size'],
                num_workers=1,
                pin_memory=False
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            logging.error(f"Error creating dataloaders: {str(e)}")
            raise

    def train(self):
        """Main training loop with memory-efficient processing."""
        # Initialize components
        tokenizer = EnhancedCodeTokenizer(
            vocab_size=self.config['vocab_size']
        )
        
        model = EnhancedCodeTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        # Move model to CPU and enable memory efficient optimizations
        model = model.cpu()
        model = torch.jit.script(model)  # Use TorchScript for better CPU performance
        
        # Initialize data loading with multiple sources
        code_loader = CodeDataLoader(
            token=self.config['github_token'],
            languages=self.config['languages']
        )
        
        train_loader, val_loader = self.create_dataloaders(code_loader, tokenizer)
        
        # Initialize optimizer with CPU-specific settings
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Training loop
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(self.config['epochs']):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                
                # Process in smaller chunks if needed
                loss = self.process_batch(model, batch)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Clear memory
                del loss
                torch.cuda.empty_cache()
            
            # Validation
            val_loss = self.validate(model, val_loader)
            
            # Save checkpoint if improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(model, optimizer, epoch, val_loss)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= self.config['early_stopping_patience']:
                logging.info("Early stopping triggered")
                break
    
    def process_batch(self, model, batch):
        """Process batch with memory-efficient handling."""
        outputs = model(batch)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            batch.view(-1),
            reduction='mean'
        )
        return loss
    
    def validate(self, model, val_loader):
        """Validate model with memory-efficient processing."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self.process_batch(model, batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, model, optimizer, epoch, val_loss):
        """Save model checkpoint."""
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, f'checkpoints/model_epoch_{epoch}_loss_{val_loss:.4f}.pt')

if __name__ == "__main__":
    # Configuration for CPU training
    config = {
        'github_token': os.getenv('GITHUB_TOKEN'),  # Load from environment
        'languages': ['python'],
        'min_stars': 100,
        'max_samples': 10000,  # Limit dataset size for CPU training
        'vocab_size': 32000,
        'd_model': 256,  # Reduced model size for CPU
        'nhead': 4,
        'num_layers': 4,
        'dropout': 0.1,
        'batch_size': 8,  # Small batch size for CPU
        'learning_rate': 1e-4,
        'epochs': 10,
        'early_stopping_patience': 3
    }
    
    if not config['github_token']:
        raise ValueError("GitHub token not found in environment variables")
    
    trainer = ModelTrainer(config)
    trainer.train()