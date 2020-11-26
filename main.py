from pathlib import Path

import yaml
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix

from data_loader import get_data_loaders
from network import SmileClassifier
from tools import read_yaml


def main():
    config = read_yaml(Path('config/main_settings.yaml'))

    train_loader, val_loader, test_loader = get_data_loaders(
        Path(config['directories']['dataset_dir']),
        config['training']['batch_size'],
        config['training']['validation_split'],
        config['training']['test_split']
    )

    model = SmileClassifier().float()

    device = config['training']['device']

    if device == 'gpu':
        trainer = pl.Trainer(gpus=1)
    elif device == 'tpu':
        trainer = pl.Trainer(tpu_cores=1)
    else:
        trainer = pl.Trainer()
    
    if config['workflow']['train']:
        trainer.fit(model, train_loader, val_loader)
    
    if config['workflow']['test']:
        for test_batch in test_loader:
            pass


if __name__ == '__main__':
    main()
