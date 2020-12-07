from pathlib import Path

from numpy import mean
from numpy import save as save_obj
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tqdm import trange
from torchsummary import summary
from mlxtend.plotting import plot_confusion_matrix

from data_loader import get_data_loaders
from tools import read_yaml, archive_settings
from method_helpers import get_model_path, get_model, get_loss_func, get_optimizer

torch.manual_seed(42)  # For testing


class MetricEvaluator:
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

    archive_settings(config)

    train_loader, val_loader, test_loader = get_data_loaders(
        Path(config['directories']['dataset_dir']),
        config['training']['batch_size'],
        config['training']['validation_split'],
        config['training']['test_split'],
        config['training']['shuffle']
    )

    name = config['method_name']
    out_folder = Path(config['directories']['output_dir']) / name
    model_folder = Path(config['directories']['model_dir'])

    out_folder.mkdir(parents=True, exist_ok=True)
    model_folder.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'gpu' else 'cpu'

    model = get_model(config).float()
    summary(model, (3, 64, 64), config['training']['batch_size'], device='cpu')

    if config['workflow']['train']:
        model = model.to(device)

        epochs = config['training']['epochs']
        optimizer, scheduler = get_optimizer(model.parameters(), config)
        loss_func = get_loss_func(config)
        accuracy_eval = MetricEvaluator('accuracy')

        losses = {
            'dev': [],
            'val': []
        }

        accuracies = {
            'dev': [],
            'val': []
        }

        best_loss = 1e6

        for epoch_id in range(1, epochs + 1):
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

                t.set_postfix_str(s=f'loss={mean(batch_losses):.4f}, acc={mean(batch_accs):.3f}', refresh=True)

            bloss = mean(batch_losses)
            bacc = mean(batch_accs)

            print(f'Training loss: {bloss} accuracy: {bacc}')

            losses['dev'].append(bloss)
            accuracies['dev'].append(bacc)

            # Validation loop
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

                if scheduler is not None:
                    if config['training']['scheduler'].lower() != 'plateau':
                        scheduler.step()
                    else:
                        scheduler.step(bloss)

                print(f' loss: {bloss} acc: {bacc}')

                if bloss < best_loss:
                    best_loss = bloss

                    torch.save(model.state_dict(), str(model_folder / f'model_{name}'))

        save_obj(str(out_folder / 'loss_history.npy'), losses)
        save_obj(str(out_folder / 'acc_history.npy'), accuracies)

        def save_history_plot(name, dev_vals, val_vals):
            plt.figure()
            plt.title(f'{name.capitalize()} history')
            plt.ylabel(name.capitalize())
            plt.xlabel('Epoch')
            plt.grid()
            plt.plot(dev_vals)
            plt.plot(val_vals)
            plt.legend(['Training', 'Validation'])
            plt.savefig(str(out_folder / f'{name}_history.png'))

        save_history_plot('accuracy', accuracies['dev'], accuracies['val'])
        save_history_plot('loss', losses['dev'], losses['val'])
    
    if config['workflow']['test']:
        if config['load_pretrained']:
            model_path = get_model_path(config)

            model.load_state_dict(torch.load(model_path))

        pred = []
        gt = []
        print('Testing')
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)

                y_hat = torch.round(model(x))

                y = y.detach().cpu().numpy().tolist()
                y_hat = y_hat.detach().cpu().numpy().tolist()

                try:
                    pred.extend(y_hat)
                    gt.extend(y)
                except TypeError:
                    # y_hat and y are floats
                    pred.append(y_hat)
                    gt.append(y)

            test_acc = accuracy_score(gt, pred)
            cm = confusion_matrix(gt, pred)

            print()
            print(f'Results for method {name}')
            print(f'Test accuracy: {test_acc}')
            print('Confusion_matrix')
            print(cm)

            plot_confusion_matrix(cm, class_names=['No smile', 'Smile'])

            plt.title('Confusion matrix, test accuracy: {test_acc:.2f}')
            plt.savefig(str(out_folder / f'{name}_confusion.png'))


if __name__ == '__main__':
    main()
