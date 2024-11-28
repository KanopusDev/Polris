import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
import wandb
from pathlib import Path
from data.data_loader import CodeDataLoader  # Update import here too
from models.transformer import CodeTransformer
from models.tokenizer import CodeTokenizer
import os
from dotenv import load_dotenv

load_dotenv()

class CodeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens)

class Trainer:
    def __init__(self, model, tokenizer, device='cpu', learning_rate=3e-4):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
    def train(self, train_dataloader, val_dataloader, epochs=10):
        num_training_steps = epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_training_steps * 0.1,
            num_training_steps=num_training_steps
        )
        
        wandb.init(project="Polris")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_dataloader:
                output = self.model(batch['input_ids'].to(self.device))
                loss = CrossEntropyLoss()(
                    output.view(-1, output.size(-1)),
                    batch['labels'].to(self.device).view(-1)
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
            
            val_loss = self._validate(val_dataloader)
            wandb.log({
                'train_loss': total_loss / len(train_dataloader),
                'val_loss': val_loss
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

    def _validate(self, val_dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                output = self.model(batch['input_ids'].to(self.device))
                loss = CrossEntropyLoss()(
                    output.view(-1, output.size(-1)),
                    batch['labels'].to(self.device).view(-1)
                )
                total_loss += loss.item()
        return total_loss / len(val_dataloader)

    def _save_checkpoint(self, epoch, val_loss):
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, f'checkpoints/model_epoch_{epoch}_loss_{val_loss:.4f}.pt')

if __name__ == "__main__":
    # Initialize data loader
    code_loader = CodeDataLoader(token=os.getenv('GITHUB_TOKEN'))
    code_data = code_loader.load_data()
    
    # Initialize model, tokenizer, and trainer
    tokenizer = CodeTokenizer()
    tokenizer.fit(code_data)
    
    model = CodeTransformer(vocab_size=tokenizer.vocab_size)
    
    dataset = CodeDataset(code_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    trainer = Trainer(model, tokenizer, device='cpu')
    trainer.train(dataloader, dataloader)