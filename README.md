# FlashBN and ReBN: backdoor removal by intervening only in BatchNorm

Two defenses that remove backdoors from a trained network by changing only its BatchNorm
layers (statistics, scale/shift, input clipping). Convolution and classifier weights are
never modified.

- **FlashBN** is training-free. It re-initialises BN once from 2,000 clean training
  images: exact clean statistics, a closed-form scale/shift, per-channel input clipping,
  and optionally per-sample statistic clipping and BN fine-pruning. It takes no gradient
  step, and the defended model scores each image on its own, in eval mode.
- **ReBN** is FlashBN followed by retraining of only the BN scale/shift on 1,000 clean
  images.

## Project layout

```
flashbn_rebn/
├── README.md
├── requirements.txt
├── bn_defense/               the code
│   ├── models.py             PreActResNet-18
│   ├── data.py               clean datasets + loading of attacked models
│   ├── flashbn.py            FlashBN (step 1)
│   ├── rebn.py               ReBN (step 1 + BN-only retraining)
│   ├── run.py                command line: python -m bn_defense.run
│   └── convert.py            BackdoorBench attack folder -> attacked-model folder
├── configs/                  defaults per dataset (cifar10.yaml, tiny.yaml)
├── notebooks/
│   └── paper_results.ipynb   runs every paper experiment and prints the tables
├── attacked_models/          PUT THE ATTACKED MODELS HERE: <model>/model.pt + poisoned_test.npz
├── data/                     clean datasets, downloaded automatically (see data/README.md)
└── results/                  outputs, one folder per model and run
```

## Quick start

```bash
pip install -r requirements.txt   # Python >= 3.8, PyTorch with CUDA

# 1. Attacked models: download them and unpack each model folder into attacked_models/,
#    e.g. attacked_models/tiny_preactresnet18_blended_0_1/{model.pt, poisoned_test.npz}
#    (folder names, and converting BackdoorBench models: attacked_models/README.md)

# 2. Run everything from the notebook (start Jupyter in this folder)
jupyter notebook notebooks/paper_results.ipynb

# or run a single defense from the command line
python -m bn_defense.run flashbn attacked_models/tiny_preactresnet18_blended_0_1 --fp_ratio 0.05
python -m bn_defense.run rebn attacked_models/tiny_preactresnet18_lf_0_1 \
    --init_mode confidence_recalib_opt --confidence_patience 2
```

- **Datasets:** CIFAR-10 and Tiny ImageNet download to `data/` on first use.
- **Settings:** every key in `configs/<dataset>.yaml` can be overridden with `--key value`.
- **Outputs:** each run writes to `results/<model>/<method>/`:
  - `log.txt`;
  - `settings.json`;
  - `metrics.json`: Clean Acc / ASR / RA after every stage, plus `final`;
  - `defended_model.pt`: the state dict plus the clipping bounds, which are part of the
    defense.

## Paper results (PreActResNet-18, 3 seeds, mean ± std; Clean Acc / ASR / RA %)

| Dataset | No defense | FlashBN | ReBN |
|---|---|---|---|
| CIFAR-10 (8 attacks) | 90.3 / 81.4 / 16.2 | 87.9 ± 0.1 / 4.9 ± 0.1 / 74.5 ± 0.2 | 89.3 ± 0.1 / 3.7 ± 0.2 / 72.0 ± 0.3 |
| Tiny ImageNet (9 attacks) | 56.4 / 97.5 / 1.2 | 53.3 ± 0.1 / 10.3 ± 0.2 / 33.1 ± 0.2 | 50.4 ± 0.1 / 0.6 ± 0.1 / 35.9 ± 0.5 |

The per-attack tables and the frozen per-attack settings are in
`notebooks/paper_results.ipynb`.

## Monitoring with TensorBoard

Every run writes TensorBoard events into its result folder:

| Chart | Content |
|---|---|
| `stages/` | Clean Acc / ASR / RA after each stage; step 0 is the undefended model |
| `retrain/` | ReBN only, per epoch: Clean Acc / ASR / RA (epoch 0 = after FlashBN), training loss, training accuracy, learning rate |
| `final/` | Final metrics |
| Text tab | Full settings and metrics |

```bash
tensorboard --logdir results --port 6006      # then open http://localhost:6006
```

- **Older TensorBoard (e.g. 2.7) with protobuf ≥ 4:** start it with
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. The code sets this itself for writing.
- **Remote machine:** open a tunnel first, `ssh -N -L 6006:localhost:6006 <user>@<server>`.
