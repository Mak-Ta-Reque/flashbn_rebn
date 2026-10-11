# FlashBN and ReBN

Code for the NeurIPS 2026 paper **Backdoor Attacks Rerouted: BatchNorm as a Sink for
Adversarial Signals** ([OpenReview](https://openreview.net/forum?id=1YqAFIeJti)). It runs the
FlashBN and ReBN backdoor defenses on attacked PreActResNet-18 models (CIFAR-10, Tiny
ImageNet) and reproduces the paper results.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate   # or a conda environment; Python >= 3.8
pip install -r requirements.txt                      # PyTorch with CUDA
```

Keep this environment activated for all steps below, so that `jupyter` and `python` come from it.

## 2. Add the attacked models

Each attacked model is one folder in `attacked_models/`:

```
attacked_models/
├── cifar10_preactresnet18_combat_pipeline/
│   ├── model.pt
│   └── poisoned_test.npz
└── ...
```

**Option A: the paper's 17 models (recommended).** Download the archive and unpack it in the
project folder:

```bash
wget https://huggingface.co/abka03/flashbn_rebn-models/resolve/main/attacked_models.zip   # 1.7 GB, from huggingface.co/abka03/flashbn_rebn-models
sha256sum attacked_models.zip      # should print 2f81bddb7ecc79e2f0cfa2e70c147ea87606cacf56f8e5968d725433ebb5db1d
unzip attacked_models.zip -d .          # creates attacked_models/<17 model folders>
ls attacked_models/*/model.pt           # should list 17 files
```

**Option B: the Tiny ImageNet models from the BackdoorBench model zoo.**

1. Download the PreActResNet-18 Tiny ImageNet attack folders from the zoo:
   https://cuhko365.sharepoint.com/:f:/s/SDSbackdoorbench/EmYD8BoPY8hAqNCV_Rb_zwsBFdqf88Yx01xi0V8tc4whvw?e=d7oJNc
   (for example `tiny_preactresnet18_blended_0_1`).
2. Convert each folder into this format:

```bash
python -m bn_defense.convert <downloaded folder> attacked_models/tiny_preactresnet18_blended_0_1
```

The 8 CIFAR-10 models are not in the zoo; they are only in the Option A archive.

- **Expected folder names:** listed in `attacked_models/README.md`.
- **Missing models:** the notebook skips any model that isn't there.

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

## 7. Follow the runs live in TensorBoard

Every run writes TensorBoard updates into its results folder while it runs. To follow them:

1. Start TensorBoard in a second terminal, with the same environment activated, before or while
   the notebook runs:

   ```bash
   tensorboard --logdir results --port 6006 --reload_interval 30
   ```

2. Open http://localhost:6006. On a remote server, first open a tunnel from your own
   computer with `ssh -N -L 6006:localhost:6006 <user>@<server>`.
3. Leave the page open. New runs and new points appear on their own; TensorBoard re-reads
   `results/` every 30 seconds, and the reload button (top right) updates it at once.

What you can see:

| Section | Content |
|---|---|
| `stages/` | Clean Acc / ASR / RA after each defense stage; step 0 is the attacked model |
| `retrain/` | ReBN only, per epoch: Clean Acc / ASR / RA, training loss, training accuracy, learning rate |
| `final/` | The final Clean Acc / ASR / RA of each run |
| Text tab | The full settings and metrics of each run |

- **Choosing runs:** run names follow `<model>/<method>_s<seed>`. Type part of one into
  the run filter on the left, for example `tiny_`, `rebn_s0` or `blended`.
- **TensorBoard 2.7 with protobuf 4 or newer:** if it fails to start, use
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python tensorboard --logdir results --port 6006`.

## Citation

```bibtex
@inproceedings{anonymous2026backdoor,
  title={Backdoor Attacks Rerouted: BatchNorm as a Sink for Adversarial Signals},
  author={Anonymous},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=1YqAFIeJti}
}
```

This code builds on BackdoorBench, whose models and data formats it uses:

```bibtex
@inproceedings{backdoorbench,
  title={BackdoorBench: A Comprehensive Benchmark of Backdoor Learning},
  author={Wu, Baoyuan and Chen, Hongrui and Zhang, Mingda and Zhu, Zihao and Wei, Shaokui and Yuan, Danni and Shen, Chao},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free for non-commercial use with
attribution; see `LICENSE`. BackdoorBench components and the BackdoorBench model zoo are also
under CC BY-NC 4.0.
