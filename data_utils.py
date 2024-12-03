import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

class DataProcessor:
    def __init__(self, batch_size: int, max_seq_length: int) -> None:
        self.logger = logging.getLogger(__name__)
        self._validate_params(batch_size, max_seq_length)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        
    @staticmethod
    def _validate_params(batch_size: int, max_seq_length: int) -> None:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if max_seq_length <= 0:
            raise ValueError("Max sequence length must be positive")
    
    def create_dataloader(self, data_path: str) -> torch.utils.data.DataLoader:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        try:
            dataset = MemoryEfficientDataset(
                path,
                max_seq_length=self.max_seq_length
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
                drop_last=True
            )
        except Exception as e:
            self.logger.error(f"Failed to create dataloader: {str(e)}")
            raise

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
        return {
            'input_ids': torch.tensor(chunk[:-1]),
            'labels': torch.tensor(chunk[1:])
        }