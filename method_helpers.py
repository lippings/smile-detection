from pathlib import Path

import torch.optim
from torch.nn import functional as F

from network import SmileNetworkPretrained, SmileNetworkBase


def get_model_path(config):
    model_path = config['pretrained_path']
    model_folder = Path(config['directories']['model_dir'])
    name = config['method_name']

    if model_path is None:
        model_path = str(model_folder / f'model_{name}')

    if not Path(model_path).exists():
        raise AttributeError(f'Could not find model weights in {model_path}')

    return model_path


def get_model(config):
    pretrained_name = config['network'].get('pretrained_name', None)
    finetune = config['network']['finetune']

    if pretrained_name is None:
        return SmileNetworkBase()
    else:
        return SmileNetworkPretrained(pretrained_name, finetune)


def get_optimizer(params, config):
    opt_name = config['training']['optimizer'].lower()
    opt_params = config['training']['optimizer_params'].get(opt_name, None)

    if opt_params is None:
        opt_params = {}

    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW
    }

    opt_getter = optimizers.get(opt_name, None)
    if opt_getter is None:
        msg = f'Unrecognized optimizer name {opt_name}.' \
              f'\nCurrently known: {", ".join(optimizers.keys())}'

        raise AttributeError(msg)

    optimizer = opt_getter(params, **opt_params)

    sched_name = config['training']['scheduler']

    if sched_name is not None:
        sched_name = sched_name.lower()
        sched_params = config['training']['scheduler_params'].get(sched_name, None)

        if sched_params is None:
            sched_params = {}

        schedulers = {
            'exponential': torch.optim.lr_scheduler.ExponentialLR,
            'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
        }

        sched_getter = schedulers.get(sched_name, None)
        if sched_getter is None:
            msg = f'Unrecognized scheduler name {sched_name}.' \
                  f'\nCurrently known: {", ".join(schedulers.keys())}'

            raise AttributeError(msg)

        scheduler = sched_getter(optimizer, **sched_params)

        return optimizer, scheduler
    return optimizer, None


def get_loss_func(config):
    training_params = config['training']
    loss_name = training_params['loss'].lower()

    loss_functions = {
        'binary_crossentropy': F.binary_cross_entropy,
        'binary_crossentropy_logits': F.binary_cross_entropy_with_logits
    }

    loss_func = loss_functions.get(loss_name, None)
    if loss_func is None:
        msg = f'Unrecognized loss function name {loss_name}.' \
              f'\nCurrently known: {", ".join(loss_functions.keys())}'

        raise AttributeError(msg)

    return loss_func
