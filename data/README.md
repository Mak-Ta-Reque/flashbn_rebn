# data/: clean datasets

The clean datasets are downloaded here automatically on first use:

| Dataset | Folder | Source |
|---|---|---|
| CIFAR-10 | `data/cifar10/cifar-10-batches-py/` | torchvision download |
| Tiny ImageNet | `data/tiny/tiny-imagenet-200/{train,val}/` | http://cs231n.stanford.edu/tiny-imagenet-200.zip; the validation images are sorted into class folders automatically |

If you already have them, copy or symlink them to these paths, or pass
`--data_dir <folder>` to `python -m bn_defense.run`.
