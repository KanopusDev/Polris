import logging
import warnings
import sys
from pathlib import Path
from typing import Optional
import torch
from model import CPUOptimizedTransformer
from data_utils import DataProcessor
from config import TrainingConfig
from trainer import Trainer

def setup_logging(log_file: Optional[str] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file or "training.log")
        ]
    )

def setup_environment():
    """Configure PyTorch environment and silence warnings"""
    # Must set threads before any other PyTorch operations
    torch.set_num_threads(4)  # Or number of CPU cores
    torch.set_num_interop_threads(1)
    
    # Configure PyTorch backend settings
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)
    
    # Enable CPU optimizations
    torch.backends.cpu.preferred_memory_format = torch.channels_last
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    
    # Silence warnings
    warnings.filterwarnings("ignore", category=UserWarning)

def main() -> None:
    # Must call setup_environment before any other operations
    setup_environment()
    
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Load and validate config
        config = TrainingConfig()
        config.validate()
        
        # Initialize model with error handling
        try:
            model = CPUOptimizedTransformer(
                layers=config.model.layers,
                hidden_size=config.model.hidden_size,
                quantization_level=config.model.quantization,
                vocab_size=config.model.vocab_size
            )
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
            
        # Initialize training components
        data_processor = DataProcessor(
            batch_size=config.batch_size,
            max_seq_length=config.max_seq_length,
            vocab_size=config.model.vocab_size
        )
        
        trainer = Trainer(
            model=model,
            config=config,
            checkpoint_dir=Path("checkpoints")
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train(data_processor)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
