"""Clean datasets, transforms, and loading of attacked-model folders (model.pt +
poisoned_test.npz; bn_defense/convert.py creates them from BackdoorBench attack folders)."""
import os
import shutil

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

DATASETS = {
    'cifar10': dict(num_classes=10, size=32, flip=True,
                    mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    'tiny': dict(num_classes=200, size=64, flip=False,
                 mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
}
TINY_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


def get_transform(dataset, train):
    """BackdoorBench transforms: resize, (random crop with padding 4 [+ flip on CIFAR-10]),
    to tensor, normalize."""
    d = DATASETS[dataset]
    t = [transforms.Resize((d['size'], d['size']))]
    if train:
        t.append(transforms.RandomCrop((d['size'], d['size']), padding=4))
        if d['flip']:
            t.append(transforms.RandomHorizontalFlip())
    t += [transforms.ToTensor(), transforms.Normalize(d['mean'], d['std'])]
    return transforms.Compose(t)


def _prepare_tiny(root):
    """Download Tiny ImageNet if missing and sort the validation images into class folders."""
    base = os.path.join(root, 'tiny-imagenet-200')
    if not os.path.isdir(os.path.join(base, 'train')):
        download_and_extract_archive(TINY_URL, root, filename='tiny-imagenet-200.zip', remove_finished=True)
    val = os.path.join(base, 'val')
    annotations = os.path.join(val, 'val_annotations.txt')
    if os.path.exists(annotations):
        with open(annotations) as f:
            for line in f:
                img, label = line.split()[:2]
                os.makedirs(os.path.join(val, label), exist_ok=True)
                src = os.path.join(val, 'images', img)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(val, label, img))
        shutil.rmtree(os.path.join(val, 'images'), ignore_errors=True)
        os.remove(annotations)
    return base


def clean_datasets(dataset, data_dir):
    """(train, test) clean datasets returning (PIL image, label); downloaded on first use."""
    root = os.path.join(data_dir, dataset)
    if dataset == 'cifar10':
        return CIFAR10(root, train=True, download=True), CIFAR10(root, train=False, download=True)
    if dataset == 'tiny':
        base = _prepare_tiny(root)
        return ImageFolder(os.path.join(base, 'train')), ImageFolder(os.path.join(base, 'val'))
    raise ValueError(f'unsupported dataset {dataset}')


class Transformed(Dataset):
    """(transform(image), label) view of a dataset."""

    def __init__(self, dataset, transform):
        self.dataset, self.transform = dataset, transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img, label = self.dataset[i][:2]
        return self.transform(img), label


class PoisonedTestSet(Dataset):
    """Poisoned test images of an attack: returns (image, attack target, true label)."""

    def __init__(self, images, target, label, transform):
        self.images, self.target, self.label, self.transform = images, target, label, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), self.target[i], self.label[i]


class AttackedModel:
    """An attacked-model folder: model.pt (backdoored weights) and poisoned_test.npz (the
    poisoned test images), plus the clean datasets of its dataset."""

    def __init__(self, model_dir, data_dir='data'):
        info = torch.load(os.path.join(model_dir, 'model.pt'), map_location='cpu')
        self.name = os.path.basename(os.path.normpath(model_dir))
        self.dataset, self.arch, self.num_classes = info['dataset'], info['arch'], info['num_classes']
        self.state_dict = info['state_dict']
        self.clean_train, self.clean_test = clean_datasets(self.dataset, data_dir)
        with np.load(os.path.join(model_dir, 'poisoned_test.npz')) as z:
            images, target, label = z['images'], z['target'], z['label']
        self.bd_test = PoisonedTestSet(images, target, label, get_transform(self.dataset, train=False))
