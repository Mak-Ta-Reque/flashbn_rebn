"""ReBN = FlashBN (step 1) + step 2: retrain only the BN scale/shift on a small clean subset.
Convolutions and the classifier stay frozen; channels zeroed by BN fine-pruning stay zero.

Step 2: SGD (momentum 0.9, cosine schedule per epoch) on `ft_ratio` of the clean training set for
`epochs` epochs, BN running statistics frozen or updated (`bn_stats`). Optional: Gaussian input
noise (`ft_noise_std`), an L2 anchor on the BN scale/shift (`affine_l2`, anchored at the original
values or at the values after step 1: `affine_l2_ref` = orig | init), and L1 on the BN scale
(`l1_gamma`), and BN-only universal adversarial unlearning (`uap_weight`, `uap_eps`, `uap_steps`):
one input perturbation shared by all images (pixel L-inf <= uap_eps) is pushed `uap_steps`
sign-gradient steps per batch to raise the loss on clean images, and uap_weight * CE on the
perturbed images is added to the loss. `freeze_last_n` > 0 keeps the last n BN layers at their
step-1 values during retraining. Synthetic shortcuts: `shortcut_k` random fake backdoors (patch,
blend, sinusoid or low-frequency pattern on clean images, random target label) give BN-parameter
gradients whose span (`shortcut_mode`: global, or per BN layer) is removed from every retraining
step; `shortcut_aug_weight` adds the loss on clean images carrying a random fake trigger with their
true labels. `orig_block` (global | layer) removes from each step the part that moves the BN
scale/shift back toward their original (backdoored) values. epochs = 0 gives step 1 only.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from .data import DATASETS, Transformed, get_transform
from .flashbn import FlashBN


def add_random_trigger(x, mean, std, gen, kind):
    """x (normalized, CPU or GPU) with one random synthetic trigger of the given kind (0 patch, 1 blend,
    2 sinusoid, 3 low-frequency field), applied in pixel space."""
    H = x.shape[-1]
    dev = x.device
    xp = x * std + mean
    if kind == 0:
        s = int(torch.randint(max(3, H // 10), max(4, H // 4) + 1, (1,), generator=gen))
        r, c = [int(v) for v in torch.randint(0, H - s + 1, (2,), generator=gen)]
        xp = xp.clone()
        xp[:, :, r:r + s, c:c + s] = torch.rand(3, s, s, generator=gen).to(dev)
    elif kind == 1:
        a = 0.1 + 0.1 * float(torch.rand(1, generator=gen))
        xp = (1 - a) * xp + a * torch.rand(1, 3, H, H, generator=gen).to(dev)
    elif kind == 2:
        f, amp = int(torch.randint(3, 9, (1,), generator=gen)), 0.05 + 0.05 * float(torch.rand(1, generator=gen))
        xp = xp + amp * torch.sin(2 * np.pi * f * torch.arange(H, device=dev) / H).view(1, 1, 1, H)
    else:
        field = nn.functional.interpolate(torch.randn(1, 3, 4, 4, generator=gen), size=(H, H), mode='bilinear',
                                          align_corners=False)
        xp = xp + 0.1 * field.to(dev)
    return (xp.clamp(0, 1) - mean) / std


def shortcut_basis(model, params, x, num_classes, mean, std, k, device, gen, slices=None, batch=256):
    """Orthonormal basis (P x k, or one per parameter slice) of the BN-parameter gradients of k
    synthetic backdoors: CE(model(x + random trigger), random target) on random pool batches."""
    model.eval()
    grads = []
    for i in range(k):
        idx = torch.randperm(len(x), generator=gen)[:batch]
        xt = add_random_trigger(x[idx], mean.cpu(), std.cpu(), gen, i % 4).to(device)
        t = torch.full((len(idx),), int(torch.randint(0, num_classes, (1,), generator=gen)), device=device)
        g = torch.autograd.grad(nn.functional.cross_entropy(model(xt), t), params)
        grads.append(torch.cat([gi.flatten() for gi in g]))
    G = torch.stack(grads)                                      # k x P
    if slices is None:
        return [(slice(0, G.shape[1]), torch.linalg.qr(G.T)[0])]
    return [(sl, torch.linalg.qr(G[:, sl].T)[0]) for sl in slices]


def project_out(params, bases):
    """Remove the shortcut subspace from the current gradients of params (in place)."""
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
    for sl, Q in bases:
        g[sl] -= Q @ (Q.T @ g[sl])
    i = 0
    for p in params:
        n = p.numel()
        if p.grad is not None:
            p.grad.copy_(g[i:i + n].view_as(p))
        i += n


def block_toward(params, slices, dirs):
    """One-sided projection of the gradients of params: per slice, if the descent direction -g has a
    positive component along the unit direction u, remove it (in place)."""
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
    for sl, u in zip(slices, dirs):
        c = float(g[sl] @ u)
        if c < 0:                                 # -g points toward the original values
            g[sl] -= c * u
    i = 0
    for p in params:
        n = p.numel()
        if p.grad is not None:
            p.grad.copy_(g[i:i + n].view_as(p))
        i += n


class ReBN(FlashBN):

    def run(self):
        cfg = self.cfg
        self.setup()
        model, bns, dev = self.model, self.bns, self.device
        params = [p for m in bns for p in (m.weight, m.bias)]
        anchor = [p.detach().clone() for p in params]   # original BN scale/shift
        theta_orig = torch.cat([p.detach().flatten() for p in params])

        def take_init_anchor():
            if cfg['affine_l2_ref'] == 'init':
                anchor[:] = [p.detach().clone() for p in params]

        masks = self.step1(after_clipping=take_init_anchor)
        self.log_scalars('retrain', self.metrics['step1'], 0)          # epoch 0 = after step 1

        # Clean retraining subset (train transform: random crop [+ flip]).
        train = Transformed(self.attacked.clean_train, get_transform(self.attacked.dataset, train=True))
        _, idx = train_test_split(np.arange(len(train)), test_size=cfg['ft_ratio'], random_state=cfg['seed'])
        train_loader = self.loader(Subset(train, idx), shuffle=True)
        logging.info(f'retraining on {len(idx)} clean images for {cfg["epochs"]} epochs')

        for p in model.parameters():
            p.requires_grad_(False)
        frozen = {id(p) for m in (bns[-cfg['freeze_last_n']:] if cfg['freeze_last_n'] > 0 else [])
                  for p in (m.weight, m.bias)}
        for p in params:
            p.requires_grad_(id(p) not in frozen)
        optimizer = torch.optim.SGD([p for p in params if id(p) not in frozen], lr=cfg['lr'], momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs'])
        uap, std = None, torch.tensor(DATASETS[self.attacked.dataset]['std'], device=dev).view(1, 3, 1, 1)
        mean = torch.tensor(DATASETS[self.attacked.dataset]['mean'], device=dev).view(1, 3, 1, 1)
        gen = torch.Generator().manual_seed(cfg['seed'] + 12345)   # own RNG for synthetic triggers
        block = None
        if cfg['orig_block'] != 'off':
            u = theta_orig - torch.cat([p.detach().flatten() for p in params])
            sls, i = [], 0
            for m in (bns if cfg['orig_block'] == 'layer' else []):
                n = m.weight.numel() + m.bias.numel()
                sls.append(slice(i, i + n))
                i += n
            sls = sls or [slice(0, u.numel())]
            block = (sls, [u[sl] / (u[sl].norm() + 1e-12) for sl in sls])
            logging.info(f"blocking moves back toward the original BN values ({cfg['orig_block']})")
        bases = None
        if cfg['shortcut_k'] > 0:
            slices, i = [], 0
            for m in bns:
                n = m.weight.numel() + m.bias.numel()
                slices.append(slice(i, i + n))
                i += n
            pool_x, _ = self.clean_pool()
            bases = shortcut_basis(model, params, pool_x, self.attacked.num_classes, mean, std, cfg['shortcut_k'],
                                   dev, gen, slices if cfg['shortcut_mode'] == 'layer' else None)
            logging.info(f"shortcut subspace: {cfg['shortcut_k']} synthetic backdoors, mode {cfg['shortcut_mode']}")
        if cfg['uap_weight'] > 0:
            size = DATASETS[self.attacked.dataset]['size']
            uap = torch.zeros(1, 3, size, size, device=dev)   # pixel units, shared by all images

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
                if uap is not None:
                    eps = cfg['uap_eps']
                    for _ in range(cfg['uap_steps']):
                        uap.requires_grad_(True)
                        g, = torch.autograd.grad(nn.functional.cross_entropy(model(x + uap / std), y), uap)
                        uap = (uap.detach() + eps / 4 * g.sign()).clamp(-eps, eps)
                    loss = loss + cfg['uap_weight'] * nn.functional.cross_entropy(model(x + uap / std), y)
                if cfg['shortcut_aug_weight'] > 0:
                    kind = int(torch.randint(0, 4, (1,), generator=gen))
                    xt = add_random_trigger(x, mean, std, gen, kind)
                    loss = loss + cfg['shortcut_aug_weight'] * nn.functional.cross_entropy(model(xt), y)
                if cfg['l1_gamma'] > 0:
                    loss = loss + cfg['l1_gamma'] * sum(m.weight.abs().sum() for m in bns)
                if cfg['affine_l2'] > 0:
                    loss = loss + cfg['affine_l2'] * sum(((p - p0) ** 2).sum() for p, p0 in zip(params, anchor))
                optimizer.zero_grad()
                loss.backward()
                if bases is not None:
                    project_out(params, bases)
                if block is not None:
                    block_toward(params, *block)
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
