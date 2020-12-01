from numpy import round
from torch import nn
import pytorch_lightning as pl

from tools import read_yaml


class SmileClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        config = read_yaml('config/main_settings.yaml')

        kernel_size = config['network']['kernel_size']
        dilation = config['network']['dilation']

        batch_norm_kwargs = config['network']['batch_norm_params']

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
            # nn.Dropout(0.05),
            nn.BatchNorm2d(32, **batch_norm_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(**maxpool_kwargs),

            nn.Conv2d(32, 32, **conv_kwargs),
            # nn.Dropout(0.05),
            nn.BatchNorm2d(32, **batch_norm_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(**maxpool_kwargs),

            nn.Conv2d(32, 32, **conv_kwargs),
            # nn.Dropout(0.05),
            nn.BatchNorm2d(32, **batch_norm_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(**maxpool_kwargs),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.Dropout(0.25),
            nn.BatchNorm1d(128, **batch_norm_kwargs),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, transform_label=False):
        h = self.conv(x)
        y = self.fc(h).squeeze()

        if transform_label:
            y = round(y)
            y = ['No smile', 'Smile'][y]

        return y
