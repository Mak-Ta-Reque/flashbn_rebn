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

- `model.pt` holds the backdoored network.
- `poisoned_test.npz` holds every non-target test image with the trigger already applied.
  `target` is the label the attack wants and `label` is the true class; ASR and RA are
  computed from them.
- A model folder needs nothing else: no attack code, trigger generator or extra paths.

## Getting the paper's models

1. Download the 17 model folders: <!-- TODO: add the download link -->.
2. Unpack them here, keeping the folder names:

   | Dataset | Folder names |
   |---|---|
   | CIFAR-10 | `cifar10_preactresnet18_<attack>_pipeline` for `reattack_bb__from_badnet__def_nad`, `ctrl`, `ftrojann`, `bpp`, `poison_ink`, `diffusion_backdoor`, `adaptive_blend`, `combat` |
   | Tiny ImageNet | `tiny_preactresnet18_<attack>` for `badnet_0_1`, `blended_0_1`, `lf_0_1`, `ssba_0_1`, `wanet_0_1`, `inputaware_0_1`, `bpp_0_1`, `trojannn_0_1`, `sig_0_005` |

3. Check: `ls attacked_models/*/model.pt`. The notebook skips any model that is missing.

## Using a BackdoorBench model

A BackdoorBench attack folder holds `attack_result.pt` and `bd_test_dataset/`. It comes
from BackdoorBench's model zoo or from running one of its attack scripts. Convert it
once:

```bash
python -m bn_defense.convert <backdoorbench folder> attacked_models/<name>
```

- The poisoned test images are copied exactly, not regenerated, so results are the same as
  in BackdoorBench.
- Supported models: PreActResNet-18 on CIFAR-10 or Tiny ImageNet.
