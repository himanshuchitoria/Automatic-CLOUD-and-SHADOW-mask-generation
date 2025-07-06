# src/training/trainer.py

import os
import time
import torch
from torch.utils.data import DataLoader
from src.models.attention_unet import AttentionUNet
from src.models.losses import get_loss_function
from src.utils.evaluation import compute_metrics
from src.utils.logger import get_logger

class Trainer:
    def __init__(self, config, train_dataset, val_dataset, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = AttentionUNet(
            in_channels=config['model']['input_channels'],
            out_channels=config['model']['num_classes']
        ).to(self.device)
        self.loss_fn = get_loss_function(config['model']['loss_function'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['model']['learning_rate']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=config['model']['batch_size'], shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config['model']['batch_size'], shuffle=False, num_workers=2
        )
        self.logger = get_logger(config['logging']['log_dir'])
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': []}

    def train(self):
        num_epochs = self.config['model']['epochs']
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for batch in self.train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(self.train_loader)
            self.history['train_loss'].append(avg_train_loss)

            # Validation
            val_loss, val_metrics = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(val_metrics)
            self.scheduler.step(val_loss)

            # Logging
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val IoU: {val_metrics['iou']:.4f}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

        self.logger.info("Training complete.")

    def validate(self):
        self.model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
        avg_val_loss = val_loss / len(self.val_loader)
        metrics = compute_metrics(all_preds, all_labels, num_classes=self.config['model']['num_classes'])
        return avg_val_loss, metrics

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint_dir = self.config['output']['submission_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        filename = f"model_epoch_{epoch+1}.pth"
        if is_best:
            filename = "best_model.pth"
        path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'history': self.history
        }, path)
        self.logger.info(f"Saved checkpoint: {filename}")

# Example usage
if __name__ == "__main__":
    import yaml
    from src.data.data_loader import NRSCCloudShadowDataset

    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Prepare datasets (replace with actual paths)
    train_dataset = NRSCCloudShadowDataset(
        images_dir=config['data']['processed_dir'] + '/toa_reflectance/train',
        labels_dir=config['data']['labels_dir'] + '/manual'
    )
    val_dataset = NRSCCloudShadowDataset(
        images_dir=config['data']['processed_dir'] + '/toa_reflectance/val',
        labels_dir=config['data']['labels_dir'] + '/manual'
    )

    trainer = Trainer(config, train_dataset, val_dataset)
    trainer.train()
