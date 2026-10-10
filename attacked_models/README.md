# attacked_models/: put the downloaded attacked models here

Put one folder per attacked model, in the BackdoorBench format, directly inside this
folder:

```
attacked_models/
├── cifar10_preactresnet18_combat_pipeline/
│   ├── attack_result.pt        # required: backdoored weights + poisoned test set
│   └── bd_test_dataset/        # poisoned test images (if the attack stored them as files)
├── tiny_preactresnet18_blended_0_1/
│   ├── attack_result.pt
│   └── bd_test_dataset/
└── ...
```

## Getting the models

1. Download the attacked-model folders.
   - The paper's 17 models: <!-- TODO: add the download link -->.
   - Standard BackdoorBench models can also be used, for example from the BackdoorBench
     model zoo (https://github.com/SCLBD/BackdoorBench). Each download contains
     `attack_result.pt` and `bd_test_dataset/`.
2. Unpack each model folder into `attacked_models/`, keeping the folder name. Copying
   or symlinking works.
3. Check the files: `ls attacked_models/*/attack_result.pt`.

The notebook (`notebooks/paper_results.ipynb`) expects these 17 folder names. A missing
model is skipped with a message.

| Dataset | Folder names |
|---|---|
| CIFAR-10 | `cifar10_preactresnet18_<attack>_pipeline` for `reattack_bb__from_badnet__def_nad`, `ctrl`, `ftrojann`, `bpp`, `poison_ink`, `diffusion_backdoor`, `adaptive_blend`, `combat` |
| Tiny ImageNet | `tiny_preactresnet18_<attack>` for `badnet_0_1`, `blended_0_1`, `lf_0_1`, `ssba_0_1`, `wanet_0_1`, `inputaware_0_1`, `bpp_0_1`, `trojannn_0_1`, `sig_0_005` |

## Your own attacked models

Any BackdoorBench attack output folder works (`record/<save_folder_name>/` after
running a BackdoorBench attack script), as long as the model is a PreActResNet-18 on
CIFAR-10 or Tiny ImageNet. Run a defense on it with:

```bash
python -m bn_defense.run flashbn attacked_models/<folder>
```

Poisoned-image paths stored inside `attack_result.pt` that no longer exist are looked up
in `<folder>/bd_test_dataset/`, so folders can be moved freely.
