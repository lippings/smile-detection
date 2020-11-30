from pathlib import Path
from typing import Tuple
import csv

from torch import from_numpy
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from numpy import float32
import cv2


class GENKIDataset(Dataset):
    def __init__(self, dataset_dir: Path, transform=None):
        super().__init__()

        self._dset_root = dataset_dir.joinpath('GENKI-R2009a/Subsets/GENKI-4K')
        self._image_dir = dataset_dir / 'GENKI-R2009a'
        metadata_dir = self._image_dir / 'Subsets' / 'GENKI-4K'

        file_name_file = metadata_dir.joinpath('GENKI-4K_Images.txt')
        label_file = metadata_dir.joinpath('GENKI-4K_Labels.txt')

        self._files = [row[0] for row in self._read_csv(file_name_file)]
        self._labels = [row[0] for row in self._read_csv(label_file)]

        self.transform = transform
    
    def _read_csv(self, path: Path):
        with path.open('r') as f:
            reader = csv.reader(f, delimiter=' ')
            return [row for row in reader]
    
    def __getitem__(self, index):
        label = float32(self._labels[index])
        file_name = self._files[index]
        file_path = self._image_dir.joinpath('files', file_name)

        og_image = cv2.imread(str(file_path))
        res_image = cv2.resize(og_image, (64, 64), interpolation=cv2.INTER_AREA)

        tensor_image = from_numpy(res_image).permute(2, 0, 1)  # H, W, C -> C, H, W
        tensor_image = tensor_image.float()

        return tensor_image, label

    def __len__(self):
        return len(self._labels)


def get_data_loaders(dataset_dir: Path, batch_size: int, validation_split: float, test_split: float, shuffle: bool) \
        -> Tuple[DataLoader, DataLoader, DataLoader]:
    if test_split > 1:
        test_split /= 100.
    
    if validation_split > 1:
        validation_split /= 100.
    
    train_split = 1 - validation_split - test_split
    
    dataset = GENKIDataset(dataset_dir)
    split_lengths = [len(dataset) * split_size for split_size in [train_split, validation_split, test_split]]
    split_lengths = [int(i) for i in split_lengths]
    train_set, validation_set, test_set = random_split(dataset, split_lengths)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=1)

    return train_loader, validation_loader, test_loader
