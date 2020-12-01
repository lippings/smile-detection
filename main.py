from pathlib import Path

import yaml
from torch import nn, manual_seed
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix

from data_loader import get_data_loaders
from network import SmileClassifier
from tools import read_yaml

manual_seed(42)  # Reproducibility for testing


def main():
    config = read_yaml(Path('config/main_settings.yaml'))

    train_loader, val_loader, test_loader = get_data_loaders(
        Path(config['directories']['dataset_dir']),
        config['training']['batch_size'],
        config['training']['validation_split'],
        config['training']['test_split'],
        config['training']['shuffle']
    )

    model = SmileClassifier().float()

    device = config['training']['device']
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=config['directories']['model_dir'],
        filename='smile-rec-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min'
    )
    save_best = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=config['directories']['model_dir'],
        filename='smile-rec-best',
        save_top_k=1,
        mode='min'
    )

    trainer_kawrgs = {
        'callbacks': [checkpoint_callback, save_best],
        'max_epochs': config['training']['epochs']
    }

    if device == 'gpu':
        trainer_kawrgs['gpus'] = 1
    elif device == 'tpu':
        trainer_kawrgs['tpu_cores'] = 1

    trainer = pl.Trainer(**trainer_kawrgs)
    
    if config['workflow']['train']:
        trainer.fit(model, train_loader, val_loader)
    
    if config['workflow']['test']:
        trainer.test(model, test_loader, ckpt_path='models/smile-rec-best-v6.ckpt')


if __name__ == '__main__':
    main()
