"""Clean datasets, transforms, and loading of attacked models saved in the BackdoorBench
format (<model folder>/attack_result.pt, poisoned test images in <model folder>/bd_test_dataset/)."""
import os
import shutil

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
    """Poisoned test images of an attack: returns (image, attack target, true label).
    Images are kept in the attack result or stored as files; a stored path that does not
    exist (e.g. after downloading the folder) is looked up inside the model folder."""

    def __init__(self, state, clean_test, transform, model_dir):
        container = state['bd_data_container']
        self.items = container['data_dict']
        self.on_disk = container['save_folder_path'] is not None
        self.index = state['original_index_array']
        self.poisoned = state['poison_indicator']
        self.clean_test, self.transform, self.model_dir = clean_test, transform, model_dir

    def __len__(self):
        return len(self.index)

    def _path(self, path):
        if os.path.exists(path):
            return path
        for sub in ('bd_test_dataset', 'bd_train_dataset'):
            if f'{sub}/' in path:
                return os.path.join(self.model_dir, sub, path.rsplit(f'{sub}/', 1)[1])
        return path

    def __getitem__(self, i):
        j = self.index[i]
        if self.poisoned[j] == 0:
            img, label = self.clean_test[j]
            target = label
        elif self.on_disk:
            item = self.items[j]
            with Image.open(self._path(item['path'])) as f:
                img = f.copy()
            target, label = item['other_info']
        else:
            img, target, label = self.items[j]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        return self.transform(img), target, label


class AttackedModel:
    """An attacked model folder: the backdoored weights, its clean datasets and the
    poisoned test set (BackdoorBench attack_result.pt)."""

    def __init__(self, model_dir, data_dir='data'):
        result = torch.load(os.path.join(model_dir, 'attack_result.pt'), map_location='cpu')
        self.name = os.path.basename(os.path.normpath(model_dir))
        self.dataset = result['clean_data']
        self.arch = result['model_name']
        self.num_classes = result['num_classes']
        self.state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in result['model'].items()}
        self.clean_train, self.clean_test = clean_datasets(self.dataset, data_dir)
        test_tf = get_transform(self.dataset, train=False)
        self.bd_test = PoisonedTestSet(result['bd_test'], self.clean_test, test_tf, model_dir)
