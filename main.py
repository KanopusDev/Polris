import logging
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

def main() -> None:
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
                quantization_level=config.model.quantization
            )
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
            
        # Initialize training components
        data_processor = DataProcessor(
            batch_size=config.training.batch_size,
            max_seq_length=config.training.max_seq_length
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
