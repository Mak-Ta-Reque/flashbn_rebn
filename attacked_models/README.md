# attacked_models/: put the attacked models here

Each attacked model is one folder with two files:

```
attacked_models/
├── cifar10_preactresnet18_combat_pipeline/
│   ├── model.pt              # backdoored weights + architecture, dataset, number of classes
│   └── poisoned_test.npz     # poisoned test images: images (N x H x W x 3, uint8), target, label
├── tiny_preactresnet18_blended_0_1/
│   ├── model.pt
│   └── poisoned_test.npz
└── ...
```

## Getting the models

**Option A: the paper's 17 models (recommended).** From the project folder:

```bash
# download from: LINK_TO_BE_ADDED
unzip attacked_models.zip -d .
ls attacked_models/*/model.pt        # 17 files
```

**Option B: Tiny ImageNet models from the BackdoorBench model zoo.**

1. Download the PreActResNet-18 Tiny ImageNet attack folders, each with `attack_result.pt` and
   `bd_test_dataset/`:
   https://cuhko365.sharepoint.com/:f:/s/SDSbackdoorbench/EmYD8BoPY8hAqNCV_Rb_zwsBFdqf88Yx01xi0V8tc4whvw?e=d7oJNc
2. Convert each folder:

```bash
python -m bn_defense.convert <downloaded folder> attacked_models/<folder name from the table>
```

The CIFAR-10 models are not in the zoo, so use Option A for them. A BackdoorBench folder you
trained yourself with its attack scripts is converted the same way. The poisoned test images
are copied, not regenerated.

## Folder names used by the notebook

| Dataset | Folder names |
|---|---|
| CIFAR-10 | `cifar10_preactresnet18_<attack>_pipeline` for `reattack_bb__from_badnet__def_nad`, `ctrl`, `ftrojann`, `bpp`, `poison_ink`, `diffusion_backdoor`, `adaptive_blend`, `combat` |
| Tiny ImageNet | `tiny_preactresnet18_<attack>` for `badnet_0_1`, `blended_0_1`, `lf_0_1`, `ssba_0_1`, `wanet_0_1`, `inputaware_0_1`, `bpp_0_1`, `trojannn_0_1`, `sig_0_005` |

The notebook skips any model that is missing.
