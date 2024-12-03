import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging
import os

class DataProcessor:
    def __init__(self, batch_size: int, max_seq_length: int) -> None:
        self.logger = logging.getLogger(__name__)
        self._validate_params(batch_size, max_seq_length)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    @staticmethod
    def _validate_params(batch_size: int, max_seq_length: int) -> None:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if max_seq_length <= 0:
            raise ValueError("Max sequence length must be positive")
    
    def create_dataloader(self, data_path: str) -> torch.utils.data.DataLoader:
        path = self.data_dir / data_path
        if not path.suffix:
            path = path.with_suffix('.pt')
            
        if not path.exists():
            self.logger.info(f"Creating dummy data file: {path}")
            self._create_dummy_data(path)
            
        try:
            dataset = MemoryEfficientDataset(
                path,
                max_seq_length=self.max_seq_length
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=False,  # Changed to False for CPU-only
                num_workers=4,
                drop_last=True,
                collate_fn=self._collate_fn
            )
        except Exception as e:
            self.logger.error(f"Failed to create dataloader: {str(e)}")
            raise
    
    def _collate_fn(self, batch):
        """Custom collate function with shape validation"""
        input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
        labels = torch.cat([item['labels'] for item in batch], dim=0)
        
        # Ensure shapes match model expectations
        if input_ids.shape[1] > self.max_seq_length:
            input_ids = input_ids[:, :self.max_seq_length]
        if labels.shape[1] > self.max_seq_length:
            labels = labels[:, :self.max_seq_length]
            
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def _create_dummy_data(self, path: Path) -> None:
        """Create dummy data for testing"""
        dummy_data = np.random.randn(
            self.max_seq_length * 1000  # Create 1000 sequences
        ).astype(np.float32)
        
        # Save as memory-mapped file
        fp = np.memmap(
            path,
            dtype='float32',
            mode='w+',
            shape=dummy_data.shape
        )
        fp[:] = dummy_data[:]
        fp.flush()

class MemoryEfficientDataset(Dataset):
    def __init__(self, data_path, max_seq_length):
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        # Memory-mapped file handling
        self.data = np.memmap(
            data_path,
            dtype='float32',
            mode='r'
        )
        
    def __len__(self):
        return len(self.data) // self.max_seq_length
        
    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_length
        end_idx = start_idx + self.max_seq_length
        
        # Load chunk of data
        chunk = self.data[start_idx:end_idx].copy()
        
        # Ensure proper shapes and padding if needed
        input_data = chunk[:-1]
        label_data = chunk[1:]
        
        # Pad or truncate to match model's hidden size
        input_tensor = torch.tensor(input_data[:self.max_seq_length-1])
        label_tensor = torch.tensor(label_data[:self.max_seq_length-1])
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        label_tensor = label_tensor.unsqueeze(0).long()  # Convert to long for CrossEntropyLoss
        
        return {
            'input_ids': input_tensor,
            'labels': label_tensor
        }