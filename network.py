from numpy import round
from torch import nn
from torchvision.models import mobilenet_v2, resnet50
from torchvision.transforms import Resize

from tools import read_yaml


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SmileClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        config = read_yaml('config/main_settings.yaml')

        batch_norm_kwargs = config['network']['batch_norm_params']

        activations = {
            'relu': nn.ReLU,
            'relu6': nn.ReLU6
        }

        act_func = activations[config['network']['activation'].lower()]()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.Dropout(0.1),
            nn.BatchNorm1d(128, **batch_norm_kwargs),
            act_func,
            # nn.Linear(1024, 128),
            # nn.BatchNorm1d(128, **batch_norm_kwargs),
            # nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class SmileNetworkBase(nn.Module):
    def __init__(self):
        super().__init__()

        config = read_yaml('config/main_settings.yaml')

        kernel_size = config['network']['kernel_size']
        dilation = config['network']['dilation']

        batch_norm_kwargs = config['network']['batch_norm_params']

        activations = {
            'relu': nn.ReLU,
            'relu6': nn.ReLU6
        }

        act_func = activations[config['network']['activation'].lower()]()

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
            'kernel_size': 2,
            'stride': 2,
            'padding': (0, 0)
        }

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, **conv_kwargs),
            # nn.Dropout(0.05),
            nn.BatchNorm2d(32, **batch_norm_kwargs),
            act_func,
            nn.MaxPool2d(**maxpool_kwargs),

            # nn.Conv2d(32, 32, **conv_kwargs),
            # nn.BatchNorm2d(32, **batch_norm_kwargs),
            # nn.ReLU(),

            nn.Conv2d(32, 32, **conv_kwargs),
            # nn.Dropout(0.05),
            nn.BatchNorm2d(32, **batch_norm_kwargs),
            act_func,
            nn.MaxPool2d(**maxpool_kwargs),

            # nn.Conv2d(32, 32, **conv_kwargs),
            # nn.BatchNorm2d(32, **batch_norm_kwargs),
            # nn.ReLU(),

            nn.Conv2d(32, 32, **conv_kwargs),
            # nn.Dropout(0.05),
            nn.BatchNorm2d(32, **batch_norm_kwargs),
            act_func,
            nn.MaxPool2d(**maxpool_kwargs),
        )

        self.cls = SmileClassifier(2048)
    
    def forward(self, x):
        h = self.conv(x)
        y = self.cls(h).squeeze()

        return y


class SmileNetworkPretrained(nn.Module):
    def __init__(self, pretrained_name, finetune=False):
        super().__init__()

        self.resizer = Resize((224, 224))  # Pretrained models expect at least 224 X 224 (but task specifies 64x64)

        pretrained_name = pretrained_name.lower()

        if pretrained_name == 'mobilenetv2':
            self.feat_ext = mobilenet_v2(pretrained=True)

            self.feat_ext.classifier = Identity()

            out_dim = 1280
        elif pretrained_name == 'resnet50':
            self.feat_ext = resnet50(True)

            self.feat_ext.fc = Identity()

            out_dim = 2048
        else:
            self.feat_ext = None

        if self.feat_ext is None:
            msg = f'Unrecognized pretrained model name {pretrained_name}.' \
                  f'\nCurrently known: ResNet50, MobileNetV2'
            raise AttributeError(msg)

        if not finetune:
            self._freeze_pretrained()

        self.cls = SmileClassifier(out_dim)

    def forward(self, x):
        res_x = self.resizer(x)

        h = self.feat_ext(res_x)
        y_hat = self.cls(h)

        return y_hat.squeeze()

    def _freeze_pretrained(self):
        for param in self.feat_ext.parameters():
            param.requires_grad = False
