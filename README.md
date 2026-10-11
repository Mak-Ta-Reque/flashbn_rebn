# FlashBN and ReBN

Code to run the FlashBN and ReBN backdoor defenses on attacked PreActResNet-18 models
(CIFAR-10, Tiny ImageNet) and reproduce the paper results.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate   # or a conda environment; Python >= 3.8
pip install -r requirements.txt                      # PyTorch with CUDA
```

Keep this environment activated for all steps below, so that `jupyter` and `python` come from it.

## 2. Add the attacked models

Download the attacked models and unpack each model folder into `attacked_models/`:

```
attacked_models/
├── cifar10_preactresnet18_combat_pipeline/
│   ├── model.pt
│   └── poisoned_test.npz
└── ...
```

- **Expected folder names:** listed in `attacked_models/README.md`. A missing model is
  skipped.
- **BackdoorBench attack folders** (`attack_result.pt` + `bd_test_dataset/`) can be converted
  first:

```bash
python -m bn_defense.convert <backdoorbench folder> attacked_models/<name>
```

## 3. Datasets

CIFAR-10 and Tiny ImageNet are downloaded into `data/` automatically on first use. To use
copies you already have, see `data/README.md`.

## 4. Reproduce the paper results

Open the notebook from the project folder and run all cells:

```bash
jupyter notebook notebooks/paper_results.ipynb
# or, without the Jupyter interface:
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/paper_results.ipynb
```

- **Output:** the notebook runs FlashBN and ReBN on every attacked model with 3 seeds, then
  prints the result tables (Clean Acc / ASR / RA, mean ± std).
- **Runtime:** a full run from scratch takes about 6–7 hours on 2 GPUs.
- **Finished runs** in `results/` are reused, so a re-run only computes what is missing.

Options in the notebook's first code cell:

| Option | Meaning |
|---|---|
| `GPUS`, `PARALLEL` | GPUs to use and how many runs at once |
| `SEEDS` | Seeds to run (default `[0, 1, 2]`) |
| `RUN = False` | Only build the tables from finished runs |
| `FORCE = True` | Recompute all runs |
| `FLASHBN_REFRESH` | `'adopted'` (paper), `'off'` or `'all'` |

## 5. Run one defense

```bash
python -m bn_defense.run flashbn attacked_models/<model folder>
python -m bn_defense.run rebn    attacked_models/<model folder>
```

- **Defaults:** read from `configs/cifar10.yaml` or `configs/tiny.yaml`, chosen by the
  model's dataset.
- **Overriding a setting:** any key in those files can be changed on the command line, e.g.
  `--epochs 50 --fp_ratio 0.1 --device cuda:1`.
- **Per-attack settings:** the paper's settings are listed in the notebook.

## 6. Outputs

Each run writes to `results/<model>/<method>/` (or `--out_dir`):

| File | Content |
|---|---|
| `metrics.json` | Clean Acc / ASR / RA after each stage, and `final` |
| `defended_model.pt` | The defended model |
| `settings.json` | The settings used |
| `log.txt` | The run log |

## 7. Watch runs in TensorBoard

```bash
tensorboard --logdir results --port 6006     # open http://localhost:6006
```

- **Remote server:** first run `ssh -N -L 6006:localhost:6006 <user>@<server>`.
- **TensorBoard 2.7 with protobuf 4 or newer:** start it with
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python tensorboard ...`.
