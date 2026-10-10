"""Run FlashBN or ReBN on one attacked model.

    python -m bn_defense.run flashbn attacked_models/tiny_preactresnet18_blended_0_1 --fp_ratio 0.05
    python -m bn_defense.run rebn attacked_models/tiny_preactresnet18_lf_0_1 \
        --init_mode confidence_recalib_opt --confidence_patience 2

Defaults come from configs/<dataset>.yaml (the dataset is read from the attack result); any
config key can be overridden on the command line. Outputs go to --out_dir (default
results/<model>/<method>/): log.txt, settings.json, metrics.json, defended_model.pt.
"""
import argparse
import json
import logging
import os
import sys

import yaml

from .data import AttackedModel
from .flashbn import FlashBN
from .rebn import ReBN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = {'flashbn': FlashBN, 'rebn': ReBN}
CHOICES = {'init_mode': ['hist_clip', 'pgd_hist_clip', 'confidence_recalib_opt', 'reset_affine', 'fisher_noise'],
           'bn_stats': ['frozen', 'update'], 'affine_l2_ref': ['orig', 'init'],
           'ft_optimizer': ['sgd', 'adam', 'adamw']}


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('method', choices=list(METHODS))
    p.add_argument('model', help='attacked model folder (model.pt + poisoned_test.npz)')
    p.add_argument('--out_dir', help='default: results/<model>/<method>/')
    p.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    p.add_argument('--device', default='cuda')
    known, rest = p.parse_known_args(argv)
    # Every config key is an option; its type comes from the config value.
    defaults = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'cifar10.yaml')))
    q = argparse.ArgumentParser()
    for k, v in defaults.items():
        typ = (lambda s: s.lower() in ('1', 'true', 'yes')) if isinstance(v, bool) else type(v)
        q.add_argument(f'--{k}', type=typ, choices=CHOICES.get(k))
    overrides = {k: v for k, v in vars(q.parse_args(rest)).items() if v is not None}
    return known, overrides


def main(argv=None):
    args, overrides = parse(sys.argv[1:] if argv is None else argv)
    attacked = AttackedModel(args.model, args.data_dir)
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', f'{attacked.dataset}.yaml')))
    cfg.update(overrides)
    out_dir = args.out_dir or os.path.join(ROOT, 'results', attacked.name, args.method)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', force=True,
                        handlers=[logging.FileHandler(os.path.join(out_dir, 'log.txt'), 'w'), logging.StreamHandler()])
    settings = {'method': args.method, 'model': attacked.name, 'overrides': overrides, 'config': cfg}
    json.dump(settings, open(os.path.join(out_dir, 'settings.json'), 'w'), indent=1)
    logging.info(f'{args.method} on {attacked.name} ({attacked.dataset}); overrides {overrides}')
    METHODS[args.method](cfg, attacked, out_dir, args.device).run()


if __name__ == '__main__':
    main()
