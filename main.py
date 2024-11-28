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
from data.data_loader import CodeDataLoader
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
        """Create memory-efficient dataloaders with better error handling."""
        try:
            # Load and process data
            all_code_data = code_loader.load_data(sources=['github'])
            
            if len(all_code_data) < 2:  # Need at least 2 samples for train/val split
                raise ValueError(f"Insufficient data: only {len(all_code_data)} samples found")
            
            # Process data in batches
            processed_data = []
            for code in all_code_data:
                try:
                    # Normalize and clean code
                    code = str(code).strip()
                    if len(code) < 10:
                        continue
                    
                    # Tokenize with length check
                    tokens = tokenizer.encode(code, add_special_tokens=True)
                    if len(tokens) > 1 and len(tokens) <= 2048:  # Add length limit
                        processed_data.append(tokens)
                        
                except Exception as e:
                    logging.warning(f"Failed to process code sample: {str(e)}")
                    continue
            
            if len(processed_data) < 2:
                raise ValueError(f"No valid processed data available. Only {len(processed_data)} samples after processing")
            
            # Create train/val split
            train_size = int(0.9 * len(processed_data))
            train_data = processed_data[:train_size]
            val_data = processed_data[train_size:]
            
            # Create tensor datasets
            train_dataset = torch.utils.data.TensorDataset(
                torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(x, dtype=torch.long) for x in train_data],
                    batch_first=True
                )
            )
            
            val_dataset = torch.utils.data.TensorDataset(
                torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(x, dtype=torch.long) for x in val_data],
                    batch_first=True
                )
            )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=min(self.config['batch_size'], len(train_dataset)),
                shuffle=True,
                num_workers=2,
                pin_memory=False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(self.config['batch_size'], len(val_dataset)),
                shuffle=False,
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
        code_loader = CodeDataLoader(
            token=self.config['github_token'],
            languages=self.config['languages']
        )
        
        # Load data first
        code_data = code_loader.load_data(sources=['github'])
        if not code_data:
            raise ValueError("No code data loaded from GitHub")
            
        # Initialize and train tokenizer
        tokenizer = EnhancedCodeTokenizer(
            vocab_size=self.config['vocab_size']
        )
        
        try:
            tokenizer.train(code_data)
        except Exception as e:
            logging.error(f"Failed to train tokenizer: {e}")
            raise
            
        model = EnhancedCodeTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        # Move model to CPU and enable memory efficient optimizations
        model = model.cpu()
        model = torch.jit.script(model)
        
        # Create dataloaders after tokenizer is trained
        train_loader, val_loader = self.create_dataloaders(code_loader, tokenizer)
        
        # Initialize optimizer
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
        # Unpack batch tensor
        inputs = batch[0]
        targets = inputs[:, 1:]  # Use shifted input as targets
        inputs = inputs[:, :-1]  # Remove last token from inputs
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_token_id
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
        'vocab_size': 3500,  # Reduced from 32000 to comply with SentencePiece limits
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