"""ReBN = FlashBN (step 1) + step 2: retrain only the BN scale/shift on a small clean subset.
Convolutions and the classifier stay frozen; channels zeroed by BN fine-pruning stay zero.

Step 2: SGD (momentum 0.9, cosine schedule per epoch) on `ft_ratio` of the clean training set for
`epochs` epochs, BN running statistics frozen or updated (`bn_stats`). Optional: Gaussian input
noise (`ft_noise_std`), an L2 anchor on the BN scale/shift (`affine_l2`, anchored at the original
values or at the values after step 1: `affine_l2_ref` = orig | init), and L1 on the BN scale
(`l1_gamma`). epochs = 0 gives step 1 only.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from .data import Transformed, get_transform
from .flashbn import FlashBN


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

        masks = self.step1(after_clipping=take_init_anchor)
        self.log_scalars('retrain', self.metrics['step1'], 0)          # epoch 0 = after step 1

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
