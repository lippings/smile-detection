from pathlib import Path
from typing import Tuple
import csv

from torch import randn as randn_torch
from torch import clamp as clamp_torch
from numpy import clip as clamp_numpy
from numpy import ndarray as np_array
from numpy import uint8
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from numpy import float32
import cv2

DEBUG = False


def show_tensor(ten, title=''):
    cv2.imshow(title, ten.permute(1, 2, 0).numpy())


# From https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if type(tensor) == np_array:
            noise = randn_torch(tensor.shape) * self.std + self.mean
            noise = noise.numpy() * 255

            noised = tensor + noise
            noised = clamp_numpy(noised, 0, 255)
            noised = uint8(noised)
        else:
            noise = randn_torch(tensor.size()) * self.std + self.mean

            noised = tensor + noise
            noised = clamp_torch(noised, 0, 1)

        return noised

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class GENKIDataset(Dataset):
    def __init__(self, dataset_dir: Path, transform=None, train_transforms=(None, None)):
        super().__init__()

        self._dset_root = dataset_dir.joinpath('GENKI-R2009a/Subsets/GENKI-4K')
        self._image_dir = dataset_dir / 'GENKI-R2009a'
        metadata_dir = self._image_dir / 'Subsets' / 'GENKI-4K'

        file_name_file = metadata_dir.joinpath('GENKI-4K_Images.txt')
        label_file = metadata_dir.joinpath('GENKI-4K_Labels.txt')

        self._files = [row[0] for row in self._read_csv(file_name_file)]
        self._labels = [row[0] for row in self._read_csv(label_file)]

        self.transform = transform
        self.train_transforms = train_transforms

        self.training = False
    
    def _read_csv(self, path: Path):
        with path.open('r') as f:
            reader = csv.reader(f, delimiter=' ')
            return [row for row in reader]

    def train(self):
        self.training = True
    
    def __getitem__(self, index):
        label = float32(self._labels[index])
        file_name = self._files[index]
        file_path = self._image_dir.joinpath('files', file_name)

        og_image = cv2.imread(str(file_path))

        if DEBUG:
            cv2.imshow('Original', og_image)

        if self.training:
            if self.train_transforms[0]:
                og_image = self.train_transforms[0](og_image)

                if DEBUG:
                    cv2.imshow('Transform 1', og_image)

        res_image = cv2.resize(og_image, (64, 64), interpolation=cv2.INTER_AREA)
        tensor_image = transforms.ToTensor()(res_image)

        if self.training:
            if self.train_transforms[1]:
                tensor_image = self.train_transforms[1](tensor_image)

                if DEBUG:
                    show_tensor(tensor_image, 'Transform 2')

        if self.transform:
            tensor_image = self.transform(tensor_image)

            if DEBUG:
                show_tensor(tensor_image, 'Transform 3')

        return tensor_image, label

    def __len__(self):
        return len(self._labels)


def get_data_loaders(dataset_dir: Path, batch_size: int, validation_split: float, test_split: float, shuffle: bool) \
        -> Tuple[DataLoader, DataLoader, DataLoader]:
    if test_split > 1:
        test_split /= 100.
    
    if validation_split > 1:
        validation_split /= 100.
    
    dataset = GENKIDataset(
        dataset_dir,
        # Add augmentations and normalization
        transform=transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet values
        ]),
        # transform=None,
        train_transforms=(
            transforms.Compose([
                AddGaussianNoise(0, 0.05)
            ]),
            # None,
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.GaussianBlur(3, (0.1, 0.2)),
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.RandomRotation(20)
            ])
        )
    )
    train_split = 1 - validation_split - test_split

    split_lengths = [len(dataset) * split_size for split_size in [train_split, validation_split, test_split]]
    split_lengths = [int(i) for i in split_lengths]
    train_set, validation_set, test_set = random_split(dataset, split_lengths)

    train_set.dataset.train()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, validation_loader, test_loader
