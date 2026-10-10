"""Convert a BackdoorBench attack folder (attack_result.pt + poisoned images stored as files)
into a self-contained attacked-model folder for this project:

    <out>/model.pt              architecture, dataset, number of classes, weights
    <out>/poisoned_test.npz     poisoned test images (uint8, N x H x W x 3) with the attack
                                target ('target') and the true label ('label') of each image

    python -m bn_defense.convert <backdoorbench folder> attacked_models/<name>

The poisoned images are copied exactly (no re-generation), so results do not change. Image
paths stored in attack_result.pt are used if they exist, otherwise they are looked up in
<backdoorbench folder>/bd_test_dataset/.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .data import clean_datasets


def _resolve(path, folder):
    if os.path.exists(path):
        return path
    if 'bd_test_dataset/' in path:
        return os.path.join(folder, 'bd_test_dataset', path.rsplit('bd_test_dataset/', 1)[1])
    raise FileNotFoundError(path)


def poisoned_test_set(result, folder, data_dir):
    """(images, target, label) of the BackdoorBench poisoned test set, in its order."""
    state = result['bd_test']
    items = state['bd_data_container']['data_dict']
    on_disk = state['bd_data_container']['save_folder_path'] is not None
    clean_test = None
    images, target, label = [], [], []
    for j in state['original_index_array']:
        if state['poison_indicator'][j] == 0:     # a clean image inside the poisoned test set
            if clean_test is None:
                clean_test = clean_datasets(result['clean_data'], data_dir)[1]
            img, y = clean_test[j]
            t = y
        elif on_disk:
            with Image.open(_resolve(items[j]['path'], folder)) as f:
                img = f.copy()
            t, y = items[j]['other_info']
        else:
            img, t, y = items[j]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        assert img.mode == 'RGB', f'unexpected image mode {img.mode}'
        images.append(np.asarray(img, dtype=np.uint8))
        target.append(int(t))
        label.append(int(y))
    return np.stack(images), np.array(target, dtype=np.int64), np.array(label, dtype=np.int64)


def convert(folder, out, data_dir='data'):
    result = torch.load(os.path.join(folder, 'attack_result.pt'), map_location='cpu')
    os.makedirs(out, exist_ok=True)
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in result['model'].items()}
    torch.save({'arch': result['model_name'], 'dataset': result['clean_data'], 'num_classes': result['num_classes'],
                'state_dict': state_dict}, os.path.join(out, 'model.pt'))
    images, target, label = poisoned_test_set(result, folder, data_dir)
    np.savez_compressed(os.path.join(out, 'poisoned_test.npz'), images=images, target=target, label=label)
    print(f'{out}: {result["model_name"]} on {result["clean_data"]}, {len(images)} poisoned test images')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', help='BackdoorBench attack folder (contains attack_result.pt)')
    p.add_argument('out', help='output folder, e.g. attacked_models/<name>')
    p.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      'data'))
    a = p.parse_args()
    convert(a.folder, a.out, a.data_dir)
