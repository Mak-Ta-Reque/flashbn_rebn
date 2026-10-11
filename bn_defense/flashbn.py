"""FlashBN: training-free backdoor removal by re-initializing BatchNorm from a small clean
pool. Only BN statistics, BN scale/shift and BN-input clipping change; convolutions and
the classifier are never touched. FlashBN is step 1 of ReBN (rebn.py).

Step 1, from a clean pool of `pool_size` training images:
  init_mode
    hist_clip               exact clean BN running statistics and a closed-form scale/shift
                            (least-squares fit of each layer's original output) on the
                            confidently classified pool images; then every BN input is
                            clipped to its per-channel clean [clip_low_pct, clip_high_pct] range.
    pgd_hist_clip           hist_clip, then the BN scale/shift is fit to the pool labels (Adam).
    confidence_recalib_opt  clean BN running statistics, then the scale/shift fit (no clipping).
    reset_affine            clean BN running statistics, scale/shift of the last
                            `reset_last_n` BN layers reset to 1/0.
    fisher_noise            UniBP init: the top `fisher_ratio` BN scale/shift parameters by
                            Fisher information are replaced by N(0, noise_std) noise.
  isc_pct p     per-sample statistic clipping: before every BN layer, each image's
                per-channel mean and std are clipped to the clean [p, 100-p] range.
  fp_ratio r    BN fine-pruning: in the last `fp_layers` BN layers, the scale/shift of the
                fraction r of channels least active on the clean pool is set to 0. With
                `fp_refresh`, the running statistics of the BN layers after the first pruned one
                are then re-measured on the clean pool (their input changed).
The paper's FlashBN uses hist_clip (+ isc, + fp): no gradient step at all. The clipping is
part of the defended model and acts on each image on its own, in eval mode.
"""
import copy
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:   # TensorBoard not installed: no event files
    SummaryWriter = None

from .data import DATASETS, Transformed, get_transform
from .models import MODELS


def fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _forward_all(model, x, device, batch_size):
    model.eval()
    with torch.no_grad():
        return [model(x[i:i + batch_size].to(device)) for i in range(0, len(x), batch_size)]


@torch.no_grad()
def select_confident(model, x, device, threshold, min_samples=64):
    """Pool images classified with max-softmax >= threshold (at least the min_samples most confident)."""
    conf = torch.cat([torch.softmax(o, 1).max(1).values.cpu() for o in _forward_all(model, x, device, 128)])
    keep = conf >= threshold
    if keep.sum().item() < min_samples:
        keep = torch.zeros_like(keep)
        keep[torch.topk(conf, k=min(min_samples, len(conf))).indices] = True
    return x[keep]


def _hooked_pass(model, x, device, modules, hook, batch_size=128, pre=False):
    """One eval pass over x with hook(module, input, output) on each module."""
    register = (lambda m: m.register_forward_pre_hook(lambda mod, inp: hook(mod, inp, None))) if pre else \
        (lambda m: m.register_forward_hook(hook))
    handles = [register(m) for m in modules]
    _forward_all(model, x, device, batch_size)
    for h in handles:
        h.remove()


@torch.no_grad()
def recalibrate_bn(model, x, device, bns, reconstruct_affine):
    """Running stats := exact clean mean/var of each BN input over x. With reconstruct_affine,
    gamma/beta := least-squares fit of the layer's ORIGINAL output under the new statistics
    (gamma = cov(x, y) / std(x), beta = mean(y))."""
    sums = {m: {'n': 0} for m in bns}

    def hook(m, inp, out):
        xi = inp[0]
        dims = [d for d in range(xi.dim()) if d != 1]
        terms = {'x': xi, 'x2': xi ** 2}
        if reconstruct_affine:
            terms.update(y=out, xy=xi * out)
        for k, v in terms.items():
            sums[m][k] = sums[m].get(k, 0) + v.sum(dim=dims)
        sums[m]['n'] += xi.numel() // xi.shape[1]

    _hooked_pass(model, x, device, bns, hook)
    for m in bns:
        s = sums[m]
        mean = s['x'] / s['n']
        var = (s['x2'] / s['n'] - mean ** 2).clamp_min(0)
        if reconstruct_affine:
            mean_y = s['y'] / s['n']
            m.weight.copy_((s['xy'] / s['n'] - mean * mean_y) / torch.sqrt(var + m.eps))
            m.bias.copy_(mean_y)
        m.running_mean.copy_(mean)
        m.running_var.copy_(var)
        m.num_batches_tracked.zero_()


@torch.no_grad()
def clip_bounds(model, x, device, bns, low_pct, high_pct, spatial_per_sample=16):
    """Per-channel [low_pct, high_pct] percentiles of each BN input over x
    (16 random spatial positions per image)."""
    values = {m: [] for m in bns}

    def hook(m, inp, out):
        xi = inp[0]
        n_spatial = xi.shape[2:].numel()
        xf = xi.reshape(xi.shape[0], xi.shape[1], n_spatial)
        idx = torch.randperm(n_spatial, device=xi.device)[:min(spatial_per_sample, n_spatial)]
        values[m].append(xf[:, :, idx].permute(1, 0, 2).reshape(xi.shape[1], -1).cpu())

    _hooked_pass(model, x, device, bns, hook)
    out = {}
    for m in bns:
        v = torch.cat(values[m], dim=1)
        out[m] = (torch.quantile(v, low_pct / 100, dim=1).to(device), torch.quantile(v, high_pct / 100, dim=1).to(device))
    return out


def install_clipping(bounds):
    """Clamp each BN input to its per-channel [lo, hi] bounds."""
    def make_hook(lo, hi):
        return lambda m, inp: (torch.clamp(inp[0], min=lo.view(1, -1, 1, 1), max=hi.view(1, -1, 1, 1)),)
    for m, (lo, hi) in bounds.items():
        m.register_forward_pre_hook(make_hook(lo, hi))


@torch.no_grad()
def isc_bounds(model, x, device, bns, pct):
    """Per channel: [pct, 100-pct] percentiles over clean images of the per-image spatial
    mean and std of each BN input."""
    stats = {m: ([], []) for m in bns}

    def hook(m, inp, out):
        stats[m][0].append(inp[0].mean(dim=(2, 3)).cpu())
        stats[m][1].append(inp[0].std(dim=(2, 3)).cpu())

    _hooked_pass(model, x, device, bns, hook, batch_size=250)
    q = lambda t, p: torch.quantile(t, p / 100, dim=0).to(device)
    out = {}
    for m in bns:
        mu, sd = torch.cat(stats[m][0]), torch.cat(stats[m][1])
        out[m] = (q(mu, pct), q(mu, 100 - pct), q(sd, pct), q(sd, 100 - pct))
    return out


def install_isc(bounds):
    """Per-sample statistic clipping: before each BN layer, clip every image's per-channel
    spatial mean and std into the constant clean range."""
    def make_hook(b):
        lo_m, hi_m, lo_s, hi_s = [t.view(1, -1, 1, 1) for t in b]

        def hook(m, inp):
            x = inp[0]
            mu, sd = x.mean(dim=(2, 3), keepdim=True), x.std(dim=(2, 3), keepdim=True)
            mu_c, sd_c = torch.max(torch.min(mu, hi_m), lo_m), torch.max(torch.min(sd, hi_s), lo_s)
            return ((x - mu) * (sd_c / (sd + 1e-6)) + mu_c,)
        return hook
    for m, b in bounds.items():
        m.register_forward_pre_hook(make_hook(b))


@torch.no_grad()
def bn_fine_prune(model, x, device, bns, ratio):
    """Zero gamma/beta of the `ratio` fraction of channels with the lowest mean ReLU(BN output)
    over x, per BN layer. Returns [(module, mask)]."""
    acts = {m: [] for m in bns}
    _hooked_pass(model, x, device, bns, lambda m, inp, out: acts[m].append(torch.relu(out).mean(dim=(0, 2, 3))),
                 batch_size=256)
    masks = []
    for m in bns:
        mean_act = torch.stack(acts[m]).mean(0)
        mask = torch.ones_like(mean_act)
        mask[torch.argsort(mean_act)[:int(len(mean_act) * ratio)]] = 0
        m.weight.mul_(mask)
        m.bias.mul_(mask)
        masks.append((m, mask))
    return masks


def fit_bn_affine(model, x, y, device, bns, steps, lr, patience, batch_size=128):
    """Adam fit of only the BN scale/shift to (x, y), BN in eval mode. With patience > 0, stop
    once the mean max-softmax confidence has not improved for `patience` steps."""
    params = [p for m in bns for p in (m.weight, m.bias)]
    ids = {id(p) for p in params}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
    optimizer = torch.optim.Adam(params, lr=lr)
    model.eval()
    rng = torch.Generator().manual_seed(0)
    best, stale, step = -1.0, 0, 0
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(x), (min(batch_size, len(x)),), generator=rng)
        out = model(x[idx].to(device))
        loss = nn.functional.cross_entropy(out, y[idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if patience > 0:
            conf = torch.softmax(out.detach(), 1).max(1).values.mean().item()
            best, stale = (conf, 0) if conf > best + 1e-4 else (best, stale + 1)
            if stale >= patience:
                break
    for p in model.parameters():
        p.requires_grad_(True)
    return step


def fisher_noise(model, x, y, device, bns, ratio, std, batch_size=8):
    """UniBP init. Fisher information of every BN scale/shift parameter, computed on a copy with
    BN reset (scale/shift 1/0, running stats 0/1, train mode) and loss CE + log(sum over BN
    layers of ||mu_ref - mu_batch|| + 0.05 ||var_ref - var_batch||) (ref = the model's running
    stats); the top `ratio` fraction is replaced by N(0, std) noise."""
    net = copy.deepcopy(model)
    copies = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    ref = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
    for m in copies:
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
        m.running_mean.zero_()
        m.running_var.fill_(1.0)
    params = [p for m in copies for p in (m.weight, m.bias)]
    for p in net.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    reg = []

    def make_hook(i):
        def hook(m, inp, out):
            xi = inp[0]
            reg.append(torch.norm(ref[i][0] - xi.mean(dim=(0, 2, 3)), p=2)
                       + 0.05 * torch.norm(ref[i][1] - xi.var(dim=(0, 2, 3), unbiased=False), p=2))
        return hook
    handles = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(copies)]
    net.train()
    fisher, n = [torch.zeros_like(p) for p in params], 0
    for i in range(0, len(x), batch_size):
        reg.clear()
        loss = nn.functional.cross_entropy(net(x[i:i + batch_size].to(device)), y[i:i + batch_size].to(device))
        loss = loss + torch.log(sum(reg) + 1e-8)
        grads = torch.autograd.grad(loss, params)
        for f, g in zip(fisher, grads):
            f += torch.exp(loss.detach()) * g ** 2
        n += 1
    for h in handles:
        h.remove()
    total = sum(f.sum() for f in fisher) / n + 1e-12
    flat = torch.cat([f / n / total for f in fisher])
    k = max(1, int(flat.numel() * ratio))
    own = [p for m in bns for p in (m.weight, m.bias)]
    offsets = np.cumsum([0] + [p.numel() for p in own])
    with torch.no_grad():
        for idx in torch.topk(flat, k).indices.tolist():
            j = int(np.searchsorted(offsets, idx, side='right') - 1)
            own[j][idx - int(offsets[j])] = torch.randn((), device=own[j].device) * std
    return k


@torch.no_grad()
def accuracy(model, loader, device, label_index=1):
    """Share of images whose prediction equals field `label_index` of the batch."""
    model.eval()
    correct = total = 0
    for batch in loader:
        y = batch[label_index].to(device)
        correct += (model(batch[0].to(device)).argmax(1) == y).sum().item()
        total += len(y)
    return correct / max(1, total)


class FlashBN:
    """Step 1. `cfg` is the merged configuration (configs/<dataset>.yaml + overrides)."""

    def __init__(self, cfg, attacked, out_dir, device='cuda'):
        self.cfg, self.attacked, self.out_dir = cfg, attacked, out_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tb = SummaryWriter(out_dir, flush_secs=10) if SummaryWriter is not None else None

    def loader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=shuffle,
                          num_workers=self.cfg['num_workers'])

    def setup(self):
        """Seed, build the attacked model and the test loaders, score the undefended model."""
        cfg, a = self.cfg, self.attacked
        fix_random(cfg['seed'])
        model = MODELS[a.arch](num_classes=a.num_classes)
        model.load_state_dict(dict(zip(model.state_dict().keys(), a.state_dict.values())))
        self.model = model.to(self.device)
        self.bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        self.test_tf = get_transform(a.dataset, train=False)
        self.clean_loader = self.loader(Transformed(a.clean_test, self.test_tf))
        self.bd_loader = self.loader(a.bd_test)
        self.metrics = {}
        self.evaluate('original')

    def evaluate(self, stage):
        """Clean accuracy, ASR (poisoned test images predicted as the attack target) and RA
        (poisoned test images predicted as their true label), in %."""
        acc = accuracy(self.model, self.clean_loader, self.device)
        asr = accuracy(self.model, self.bd_loader, self.device, label_index=1)
        ra = accuracy(self.model, self.bd_loader, self.device, label_index=2)
        self.metrics[stage] = [round(100 * v, 2) for v in (acc, asr, ra)]
        logging.info(f'{stage:22s} Clean Acc {100 * acc:6.2f} | ASR {100 * asr:6.2f} | RA {100 * ra:6.2f}')
        self.log_scalars('stages', self.metrics[stage], len(self.metrics) - 1)   # 0 = original, then each stage
        return self.metrics[stage]

    def log_scalars(self, group, values, step):
        """TensorBoard: <group>/clean_acc, <group>/asr, <group>/ra at `step`."""
        if self.tb is not None:
            for name, v in zip(('clean_acc', 'asr', 'ra'), values):
                self.tb.add_scalar(f'{group}/{name}', v, step)

    def clean_pool(self):
        """`pool_size` clean training images (test transform), drawn with `pool_seed`."""
        train = self.attacked.clean_train
        rng = np.random.RandomState(self.cfg['pool_seed'])
        idx = rng.choice(len(train), size=min(self.cfg['pool_size'], len(train)), replace=False)
        xs, ys = zip(*[(self.test_tf(train[i][0]), train[i][1]) for i in idx])
        return torch.stack(xs), torch.tensor(ys)

    def init_bn(self, x, y):
        cfg, model, bns, dev = self.cfg, self.model, self.bns, self.device
        mode = cfg['init_mode']
        if mode == 'fisher_noise':
            k = fisher_noise(model, x, y, dev, bns, cfg['fisher_ratio'], cfg['noise_std'])
            logging.info(f'fisher_noise: {k} BN parameters replaced')
            return
        conf = select_confident(model, x, dev, cfg['confidence_threshold'])
        logging.info(f'{mode}: {len(conf)}/{len(x)} confidently classified pool images')
        if mode in ('hist_clip', 'pgd_hist_clip'):
            recalibrate_bn(model, conf, dev, bns, reconstruct_affine=True)
            conf = select_confident(model, x, dev, cfg['confidence_threshold'])
            self.clip = clip_bounds(model, conf, dev, bns, cfg['clip_low_pct'], cfg['clip_high_pct'])
            install_clipping(self.clip)
        else:
            recalibrate_bn(model, conf, dev, bns, reconstruct_affine=False)
            if mode == 'reset_affine':
                for m in bns[-cfg['reset_last_n']:] if cfg['reset_last_n'] > 0 else bns:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
        if mode in ('pgd_hist_clip', 'confidence_recalib_opt'):
            steps = fit_bn_affine(model, x, y, dev, bns, cfg['affine_opt_steps'], cfg['affine_opt_lr'],
                                  cfg['confidence_patience'])
            logging.info(f'BN scale/shift fit on the pool: {steps} steps')

    def step1(self, after_clipping=None):
        """BN init, per-sample statistic clipping, BN fine-pruning. after_clipping() runs
        before fine-pruning. Returns the fine-pruning masks."""
        cfg = self.cfg
        x, y = self.clean_pool()
        self.clip, self.isc, masks = {}, {}, []
        self.init_bn(x, y)
        self.evaluate(f"init ({cfg['init_mode']})")
        if cfg['isc_pct'] > 0:
            conf = select_confident(self.model, x, self.device, cfg['confidence_threshold'])
            self.isc = isc_bounds(self.model, conf, self.device, self.bns, cfg['isc_pct'])
            install_isc(self.isc)
            self.evaluate(f"isc ({cfg['isc_pct']})")
        if after_clipping is not None:
            after_clipping()
        if cfg['fp_ratio'] > 0:
            masks = bn_fine_prune(self.model, x, self.device, self.bns[-cfg['fp_layers']:], cfg['fp_ratio'])
            if cfg['fp_refresh'] and cfg['fp_layers'] > 1:
                after = self.bns[-cfg['fp_layers'] + 1:]
                conf = select_confident(self.model, x, self.device, cfg['confidence_threshold'])
                recalibrate_bn(self.model, conf, self.device, after, reconstruct_affine=False)
                logging.info(f'fine-pruning: running statistics of {len(after)} downstream BN layer(s) re-measured')
            self.evaluate(f"fine-pruning ({cfg['fp_ratio']})")
        self.metrics['step1'] = self.metrics[list(self.metrics)[-1]]
        return masks

    def save(self):
        """Defended model (BN state + clipping bounds) and metrics.json (final = last stage)."""
        self.metrics['final'] = self.metrics[list(self.metrics)[-1]]
        names = {m: n for n, m in self.model.named_modules()}
        if self.cfg['save_model']:
            torch.save({'state_dict': self.model.state_dict(),
                        'clip_bounds': {names[m]: [t.cpu() for t in b] for m, b in self.clip.items()},
                        'isc_bounds': {names[m]: [t.cpu() for t in b] for m, b in self.isc.items()}},
                       os.path.join(self.out_dir, 'defended_model.pt'))
        with open(os.path.join(self.out_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=1)
        acc, asr, ra = self.metrics['final']
        logging.info(f'Final  Clean Acc {acc:.2f} | ASR {asr:.2f} | RA {ra:.2f}')
        if self.tb is not None:
            self.log_scalars('final', self.metrics['final'], 0)
            self.tb.add_text('settings', '    ' + json.dumps(self.cfg, indent=1).replace('\n', '\n    '))
            self.tb.add_text('metrics', '    ' + json.dumps(self.metrics, indent=1).replace('\n', '\n    '))
            self.tb.close()

    def run(self):
        self.setup()
        self.step1()
        self.save()
        return self.metrics
