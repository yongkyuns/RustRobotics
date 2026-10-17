"""Offline float32 operation checks for PR #41; does NOT execute Rust/Burn.

Run with the retained Python 3.11.14 / Torch 2.8.0+cpu environment:
  python scripts/check_cleanrl_reference_parity.py --output parity.json

The 'burn_selector' case is a PyTorch reconstruction of Burn 0.20.1's
numeric.rs max_pair: self.mask_where(self < other, other). This distinguishes
source-derived operation semantics from a compiled Rust test. No environment
transitions, learning runs, or performance qualification take place here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import unittest
from pathlib import Path
import numpy as np
import torch


def maximum(a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == 'torch':
        return torch.maximum(a, b)
    if kind == 'burn_selector':
        return torch.where(a < b, b, a)
    if kind == 'split_ties':
        weight = (a > b).to(a.dtype) + .5 * (a == b).to(a.dtype)
        return a * weight + b * (1 - weight)
    raise ValueError(kind)


def operands():
    return (torch.tensor([[.25, -.8], [-.3, .4], [1.2, -.6], [-.8, .4]],
                         dtype=torch.float32, requires_grad=True),
            torch.tensor([.9, -.3, -.5, 1.8], dtype=torch.float32, requires_grad=True))


def objective(logits, values, kind):
    actions = torch.tensor([0, 1, 1, 0])
    oldlog = torch.tensor([-.8, -.1, -.7, -1.4])
    oldv = torch.tensor([.3, -.5, .1, 2.])
    target = torch.tensor([2., -1., .8, 0.])
    advantages = torch.tensor([-2., -.4, .7, 1.2])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    d = torch.distributions.Categorical(logits=logits)
    ratio = (d.log_prob(actions) - oldlog).exp()
    pg = maximum(-advantages * ratio, -advantages * ratio.clamp(.8, 1.2), kind).mean()
    unclipped = (values - target).square()
    clipped = (oldv + (values - oldv).clamp(-.2, .2) - target).square()
    vf = .5 * maximum(unclipped, clipped, kind).mean()
    entropy = d.entropy().mean()
    return pg + .5 * vf - .01 * entropy, (pg, vf, entropy)


def record(kind):
    x, v = operands()
    loss, terms = objective(x, v, kind)
    loss.backward()
    return {'loss': float(loss.detach()),
            'terms': [float(t.detach()) for t in terms],
            'logit_gradient': x.grad.tolist(), 'value_gradient': v.grad.tolist()}


def probability_witness(gap):
    x = torch.tensor([gap, 0.], dtype=torch.float32)
    shifted = x - x.max()
    exp = shifted.exp()
    left = exp[0] / exp.sum()
    old_right_log = (1. - left).clamp_min(torch.finfo(torch.float32).tiny).log()
    coherent_log = torch.log_softmax(x, 0)[1]
    return {'logit_gap': gap, 'old_right_log_probability': old_right_log.item(),
            'log_softmax_right': coherent_log.item(),
            'unchanged_policy_ratio_using_old_record': (coherent_log-old_right_log).exp().item()}


def optimizer_probe(kind):
    x, v = operands()
    optimizer = torch.optim.Adam([x, v], lr=2.5e-4, eps=1e-5, foreach=False)
    rows = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(x, v, kind)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([x, v], .5, error_if_nonfinite=True)
        optimizer.step()
        rows.append({'loss': float(loss.detach()), 'preclip_global_norm': norm.item(),
                     'logits': x.detach().tolist(), 'values': v.detach().tolist(),
                     'adam': [{k: z.tolist() for k, z in optimizer.state[p].items()}
                              for p in [x, v]]})
    return rows


class Tests(unittest.TestCase):
    def test_scalar_reference_values(self):
        r = record('torch')
        np.testing.assert_allclose(r['terms'], [.32307282, .95874995, .53926545], atol=1e-7, rtol=0)
        self.assertAlmostEqual(r['loss'], .79705513, places=7)

    def test_old_selector_hides_gradient_mismatch(self):
        a, b = record('torch'), record('burn_selector')
        self.assertEqual(a['loss'], b['loss'])
        self.assertAlmostEqual(a['value_gradient'][3], .1125, places=7)
        self.assertAlmostEqual(b['value_gradient'][3], .225, places=7)
        self.assertGreater(abs(a['value_gradient'][3] - b['value_gradient'][3]), .1)

    def test_split_ties_matches_all_fixture_gradients(self):
        a, b = record('torch'), record('split_ties')
        for key in ['loss', 'terms', 'logit_gradient', 'value_gradient']:
            np.testing.assert_allclose(a[key], b[key], atol=2e-7, rtol=0)

    def test_tie_gradient_is_half_for_both_operands(self):
        for kind in ['torch', 'split_ties']:
            a = torch.tensor([2., 4., 7.], requires_grad=True)
            b = torch.tensor([3., 4., 5.], requires_grad=True)
            maximum(a, b, kind).sum().backward()
            self.assertEqual(a.grad.tolist(), [0., .5, 1.])
            self.assertEqual(b.grad.tolist(), [1., .5, 0.])

    def test_many_finite_operand_pairs(self):
        gen = torch.Generator().manual_seed(45091)
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(1000, generator=gen, dtype=dtype)
            y = torch.randn(1000, generator=gen, dtype=dtype)
            y[::3] = x[::3]
            results = []
            for kind in ['torch', 'split_ties']:
                a, b = x.clone().requires_grad_(), y.clone().requires_grad_()
                z = maximum(a, b, kind)
                z.sum().backward()
                results.append((z.detach(), a.grad, b.grad))
            for a, b in zip(*results):
                self.assertTrue(torch.equal(a, b))

    def test_right_probability_cancellation_witness(self):
        r = probability_witness(16.)
        self.assertGreater(abs(r['unchanged_policy_ratio_using_old_record']-1), .05)
        self.assertAlmostEqual(r['log_softmax_right'], -16., places=5)

    def test_coherent_old_and_new_log_probability_has_unit_ratio(self):
        logits = torch.tensor([[16., 0.], [-16., 0.], [0., 0.], [100., -100.]])
        recorded = torch.log_softmax(logits, 1).detach().clone()
        replayed = torch.log_softmax(logits, 1)
        self.assertTrue(torch.equal((replayed-recorded).exp(), torch.ones_like(recorded)))
        self.assertTrue(torch.isfinite(recorded).all())

    def test_three_adam_transactions_match_corrected_semantics(self):
        a, b = optimizer_probe('torch'), optimizer_probe('split_ties')
        for x, y in zip(a, b):
            for key in ['loss','preclip_global_norm','logits','values']:
                np.testing.assert_allclose(x[key],y[key],atol=2e-7,rtol=0)
            for xs, ys in zip(x['adam'],y['adam']):
                for key in xs:
                    np.testing.assert_allclose(xs[key],ys[key],atol=2e-7,rtol=0)
        self.assertEqual(a[-1]['adam'][0]['step'],3.)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if torch.__version__ != '2.8.0+cpu' or platform.python_version() != '3.11.14':
        raise RuntimeError('Use pinned Python 3.11.14 / Torch 2.8.0+cpu')
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report = {'python':platform.python_version(), 'torch':torch.__version__, 'numpy':np.__version__,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tests_run':result.testsRun, 'tests_passed':result.wasSuccessful(),
              'burn_executed':False, 'simulation_steps':0, 'training_runs':0,
              'objective':{kind:record(kind) for kind in ['torch','burn_selector','split_ties']},
              'probability':[probability_witness(g) for g in [8.,12.,16.,20.]],
              'adam_leaf_operand_probe':{kind:optimizer_probe(kind) for kind in ['torch','split_ties']},
              'limitations':['Burn execution pending','No full-network Adam parity claim',
                             'Source-derived max-pair semantics reconstructed in PyTorch',
                             'Not evidence that these discrepancies caused historical training collapse']}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not result.wasSuccessful():
        raise SystemExit(1)

if __name__ == '__main__':
    main()
