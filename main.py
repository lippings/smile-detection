from pathlib import Path

import yaml
from numpy import mean
from numpy import save as save_obj
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tqdm import trange

from data_loader import get_data_loaders
from network import SmileClassifier
from tools import read_yaml


class MetricEvaluator():
    def __init__(self, metric_name):
        self._metric_functions = {
            'accuracy': accuracy_score
        }

        metric_name = metric_name.lower()

        self._eval = self._metric_functions.get(metric_name.lower(), None)

        if self._eval is None:
            msg = f'Unknown metric {metric_name}. Currently implemented:\n\t{", ".join(self._metric_functions.keys())}'

            raise AttributeError(msg)

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor, round=True):
        pred = torch.round(pred)
        return self._eval(
            gt.detach().cpu(),
            pred.detach().cpu()
        )


def main():
    config = read_yaml(Path('config/main_settings.yaml'))

    train_loader, val_loader, test_loader = get_data_loaders(
        Path(config['directories']['dataset_dir']),
        config['training']['batch_size'],
        config['training']['validation_split'],
        config['training']['test_split'],
        config['training']['shuffle']
    )

    name = config['method_name']
    out_folder = Path(config['directories']['output_dir']) / name
    model_folder = Path(config['directories']['output_dir'])

    out_folder.mkdir(parents=True, exist_ok=True)
    model_folder.mkdir(parents=True, exist_ok=True)

    model = SmileClassifier().float()

    if config['workflow']['train']:
        # trainer.fit(model, train_loader, val_loader)

        # Giving up on pl
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = model.to(device)

        epochs = config['training']['epochs']
        lr = float(config['training']['learning_rate'])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = F.binary_cross_entropy
        accuracy_eval = MetricEvaluator('accuracy')

        losses = {
            'dev': [],
            'val': []
        }

        accuracies = {
            'dev': [],
            'val': []
        }

        best_acc = 0

        for epoch_id in range(epochs):
            print(f'Epoch {epoch_id}')

            # Train loop
            model.train()
            batch_losses = []
            batch_accs = []
            print('Training', flush=True)
            t = trange(len(train_loader))
            for _, batch in zip(t, train_loader):
                x, y = batch

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)

                loss = loss_func(y_hat, y)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                batch_losses.append(loss.item())

                acc = accuracy_eval(y_hat, y)
                batch_accs.append(acc)

                t.set_postfix_str(s=f'loss={loss.item():.4f}, acc={acc:.3f}', refresh=True)

            bloss = mean(batch_losses)
            bacc = mean(batch_accs)

            print(f'Training loss: {bloss} accuracy: {bacc}')

            losses['dev'].append(bloss)
            accuracies['dev'].append(bacc)

            model.eval()
            with torch.no_grad():
                batch_losses = []
                batch_accs = []
                print('Validation', end='', flush=True)
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    y_hat = model(x)

                    loss = loss_func(y_hat, y)
                    batch_losses.append(loss.item())

                    acc = accuracy_eval(y_hat, y)
                    batch_accs.append(acc)

                bloss = mean(batch_losses)
                bacc = mean(batch_accs)

                losses['val'].append(bloss)
                accuracies['val'].append(bacc)

                print(f' loss: {bloss} acc: {bacc}')

                if bacc > best_acc:
                    best_acc = bacc

                    torch.save(model.state_dict(), str(model_folder / f'model_{name}'))

    save_obj(str(out_folder / 'loss_history.npy'), losses)
    save_obj(str(out_folder / 'acc_history.npy'), accuracies)

    plt.figure()
    plt.title('Accuracy history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    plt.plot(accuracies['dev'])
    plt.plot(accuracies['val'])
    plt.legend(['Training', 'Validation'])
    plt.savefig(str(out_folder / f'acc_history.png'))
    
    if config['workflow']['test']:
        if config['load_pretrained']:
            if config['pretrained_path'] is None:
                model_path = str(model_folder / f'model_{name}')
            else:
                model_path = config['pretrained_path']

            if not Path(model_path).exists():
                raise AttributeError(f'Could not find model weights in {model_path}')

            model.load_state_dict(torch.load(model_path))

        pred = []
        gt = []
        print('Testing')
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                # TODO: Confusion matrix
                pass


if __name__ == '__main__':
    main()
