from numpy import round
from torch import nn
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl

from tools import read_yaml


class SmileClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        config = read_yaml('config/main_settings.yaml')

        loss = config['training']['loss'].lower()
        optimizer = config['training']['optimizer'].lower()
        kernel_size = config['network']['kernel_size']
        dilation = config['network']['dilation']
        learning_rate = float(config['training']['learning_rate'])

        losses = {
            'binary_crossentropy': F.binary_cross_entropy
        }

        optimizers = {
            'adam': lambda: optim.Adam(self.parameters(), lr=learning_rate)
        }

        self.loss_func = losses.get(loss, None)
        if self.loss_func is None:
            print(f'Unrecognized loss function {loss}. Currently recognized:\n{", ".join(losses.keys())}')
        
        self.opt_getter = optimizers.get(optimizer, None)
        if self.opt_getter is None:
            print(f'Unrecognized optimizer {optimizer}. Currently recognized:\n{", ".join(optimizers.keys())}')

        padding = ((kernel_size - 1) / 2) * dilation

        kernel_size = int(kernel_size)
        dilation = int(dilation)
        padding = int(padding)

        conv_kwargs = {
            'kernel_size': int(kernel_size),
            'dilation': int(dilation),
            'padding': padding if type(padding) == tuple else (padding, padding)
        }

        maxpool_kwargs = {
            'kernel_size': 3,
            'stride': 2,
            'padding': (1, 1)
        }

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, **conv_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(**maxpool_kwargs),
            nn.Conv2d(32, 32, **conv_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(**maxpool_kwargs),
            nn.Conv2d(32, 32, **conv_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(**maxpool_kwargs),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.val_accuracy = pl.metrics.Accuracy()
    
    def forward(self, x, transform_label=True):
        h = self.conv(x)
        y = self.fc(h).squeeze()

        if transform_label:
            y = round(y)
            y = ['No smile', 'Smile'][y]

        return y

    def configure_optimizers(self):
        return self.opt_getter()
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        h = self.conv(x)
        y_hat = self.fc(h).squeeze()

        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        y_hat = self.fc(self.conv(x)).squeeze()

        loss = self.loss_func(y_hat, y)

        return {'loss': loss, 'pred': y_hat, 'target': y}
    
    def validation_step_end(self, outputs):
        acc = self.val_accuracy(outputs['pred'], outputs['target'])
        loss = outputs['loss']

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
    def validation_epoch_end(self, outputs):
        acc = self.val_accuracy.compute()

        self.log('Validation accuracy', acc)
    
    def test_step(self, test_batch, batch_idx):
        predictions = self.forward(test_batch, transform_label=True)

        return predictions
