"""ReBN = FlashBN (step 1) + step 2: retrain only the BN scale/shift on a small clean subset.
Convolutions and the classifier stay frozen; channels zeroed by BN fine-pruning stay zero.

Step 2: SGD (momentum 0.9, cosine schedule) on `ft_ratio` of the clean training set for
`epochs` epochs, BN running statistics frozen or updated (`bn_stats`). Optional: Gaussian
input noise (`ft_noise_std`), an L2 anchor on the BN scale/shift (`affine_l2`, anchored at the
original values or at the values after step 1: `affine_l2_ref` = orig | init), and L1 on the
BN scale (`l1_gamma`). epochs = 0 gives step 1 only.

Filters that keep retraining from regrowing the backdoor (all off by default):
  protect_ratio r  per BN layer, the fraction r of channels whose clean statistics differ most
                   from the stored (poisoned-training) running statistics get no gradient
  kd_step1 w       loss = (1-w) CE + w T^2 KL(student/T || step-1 model/T): distill from the
                   backdoor-free step-1 model (kd_T = T)
"""
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from .data import Transformed, get_transform
from .flashbn import FlashBN, _hooked_pass


@torch.no_grad()
def protected_channel_masks(model, x, device, bns, ratio):
    """Per BN layer: mask (0 = protected) of the `ratio` fraction of channels with the largest
    shift between the clean statistics of the BN input over x and the stored running statistics,
    |mean_clean - mean_run| / std_run + |log(var_clean / var_run)|."""
    sums = {m: [0, 0, 0] for m in bns}

    def hook(m, inp, out):
        xi = inp[0]
        sums[m][0] = sums[m][0] + xi.sum(dim=(0, 2, 3))
        sums[m][1] = sums[m][1] + (xi ** 2).sum(dim=(0, 2, 3))
        sums[m][2] += xi.numel() // xi.shape[1]

    _hooked_pass(model, x, device, bns, hook)
    masks = {}
    for m in bns:
        s, s2, n = sums[m]
        mean = s / n
        var = (s2 / n - mean ** 2).clamp_min(1e-8)
        run_var = m.running_var.clamp_min(1e-8)
        score = (mean - m.running_mean).abs() / run_var.sqrt() + (var / run_var).log().abs()
        mask = torch.ones_like(score)
        mask[torch.argsort(score, descending=True)[:int(len(score) * ratio)]] = 0
        masks[m] = mask
    return masks


class ReBN(FlashBN):

    def run(self):
        cfg = self.cfg
        self.setup()
        model, bns, dev = self.model, self.bns, self.device
        params = [p for m in bns for p in (m.weight, m.bias)]
        anchor = [p.detach().clone() for p in params]   # original BN scale/shift

        def take_init_anchor():
            if cfg['affine_l2_ref'] == 'init':
                anchor[:] = [p.detach().clone() for p in params]

        protect = {}
        if cfg['protect_ratio'] > 0:                  # same clean pool as step 1 (drawn with pool_seed)
            protect = protected_channel_masks(model, self.clean_pool()[0], dev, bns, cfg['protect_ratio'])

        masks = self.step1(after_clipping=take_init_anchor)
        self.log_scalars('retrain', self.metrics['step1'], 0)          # epoch 0 = after step 1

        teacher = None
        if cfg['kd_step1'] > 0:                       # backdoor-free step-1 model (with its clipping)
            teacher = copy.deepcopy(model).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
        if protect:
            for m, mask in protect.items():
                m.weight.register_hook(lambda g, mask=mask: g * mask)
                m.bias.register_hook(lambda g, mask=mask: g * mask)
            n = sum(int((mk == 0).sum()) for mk in protect.values())
            logging.info(f'protected {n} BN channels (no gradient during retraining)')

        # Clean retraining subset (train transform: random crop [+ flip]).
        train = Transformed(self.attacked.clean_train, get_transform(self.attacked.dataset, train=True))
        _, idx = train_test_split(np.arange(len(train)), test_size=cfg['ft_ratio'], random_state=cfg['seed'])
        train_loader = self.loader(Subset(train, idx), shuffle=True)
        logging.info(f'retraining on {len(idx)} clean images for {cfg["epochs"]} epochs')

        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs'])

        for epoch in range(cfg['epochs']):
            model.train()
            if cfg['bn_stats'] == 'frozen':
                for m in bns:
                    m.eval()
            loss_sum = correct = seen = 0
            for x, y in train_loader:
                x, y = x.to(dev), y.to(dev)
                if cfg['ft_noise_std'] > 0:
                    x = x + torch.randn_like(x) * cfg['ft_noise_std']
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                if teacher is not None:
                    T, w = cfg['kd_T'], cfg['kd_step1']
                    with torch.no_grad():
                        t_logits = teacher(x)
                    kd = nn.functional.kl_div(nn.functional.log_softmax(logits / T, 1),
                                              nn.functional.softmax(t_logits / T, 1), reduction='batchmean')
                    loss = (1 - w) * loss + w * T * T * kd
                if cfg['l1_gamma'] > 0:
                    loss = loss + cfg['l1_gamma'] * sum(m.weight.abs().sum() for m in bns)
                if cfg['affine_l2'] > 0:
                    loss = loss + cfg['affine_l2'] * sum(((p - p0) ** 2).sum() for p, p0 in zip(params, anchor))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for m, mask in masks:
                        m.weight.mul_(mask)
                        m.bias.mul_(mask)
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                seen += len(y)
            if self.tb is not None:
                self.tb.add_scalar('retrain/train_loss', loss_sum / seen, epoch + 1)
                self.tb.add_scalar('retrain/train_acc', 100 * correct / seen, epoch + 1)
                self.tb.add_scalar('retrain/lr', optimizer.param_groups[0]['lr'], epoch + 1)
            scheduler.step()
            if epoch == cfg['epochs'] - 1 or epoch % cfg['test_interval'] == 0:
                self.log_scalars('retrain', self.evaluate(f'epoch {epoch}'), epoch + 1)

        self.save()
        return self.metrics
