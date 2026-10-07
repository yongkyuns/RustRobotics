#!/usr/bin/env python3
"""Harness regression checks. No development-seed main-training result is selected."""
from __future__ import annotations
import argparse, copy, json, math, sys, unittest
from pathlib import Path
import numpy as np
import torch
import value_batch_compare_20261007 as vb

REF=None
OUT=None

class HarnessTests(unittest.TestCase):
    def observer_control(self,n_steps):
        seed=778
        m,e=vb.build(REF,seed,n_steps,capture=True)
        try:
            m.learn(n_steps,reset_num_timesteps=False,log_interval=None)
            expected=vb.snapshot(m,e)
            trace=copy.deepcopy(e.envs[0].raw)
        finally: e.close()
        m,e=vb.build(REF,seed,n_steps,capture=True)
        path=OUT/f'diagnostics-{n_steps}.jsonl'
        try:
            rec=vb.observe(m,e,path)
            m.learn(n_steps,reset_num_timesteps=False,log_interval=None)
            self.assertTrue(vb.pool.eq(expected,vb.snapshot(m,e)))
            self.assertTrue(vb.pool.eq(trace,e.envs[0].raw))
            text=path.read_text()
            self.assertTrue(text.endswith('\n'))
            lines=text.splitlines()
            self.assertEqual(len(lines),1)
            self.assertEqual(json.loads(lines[0]),rec.rows[0])
            self.assertEqual(rec.rows[0]['adam_steps'],n_steps//64*10)
            self.assertEqual(rec.rows[0]['epochs'],10)
        finally: e.close()

    def test_observer_2048(self): self.observer_control(2048)
    def test_observer_8192(self): self.observer_control(8192)

    def test_additive_evaluator_keeps_original_fields(self):
        m,e=vb.build(REF,779,2048)
        try:
            original_lib=REF.LIB
            before=vb.snapshot(m,e)
            old=REF.evaluate(m,2,16970779,0)
            rows=vb.evaluate(REF,m,e,779,0)
            extra={'discounted_gamma999','evaluation_gamma','reward_trace'}
            stripped=[{k:v for k,v in row.items() if k not in extra} for row in rows]
            self.assertEqual(old,stripped)
            self.assertIs(REF.LIB,original_lib)
            self.assertTrue(vb.pool.eq(before,vb.snapshot(m,e)))
            self.assertEqual(len(rows),96)
            self.assertEqual(len({(r['mode'],r['episode'],r['seed']) for r in rows}),96)
            for r in rows:
                self.assertEqual(len(r['reward_trace']),r['steps'])
                self.assertEqual(r['evaluation_gamma'],.999)
                independent=math.fsum(float(reward)*(.999**tick) for tick,reward in enumerate(r['reward_trace']))
                self.assertTrue(math.isclose(independent,r['discounted_gamma999'],rel_tol=1e-12,abs_tol=1e-10))
            self.assertTrue(any(abs(r['discounted_gamma999']-r['discounted'])>1e-3 for r in rows))
            vb.write(OUT/'evaluation-records.json',rows)
        finally: e.close()

    def test_evaluator_restores_library_on_error(self):
        m,e=vb.build(REF,780,2048)
        original_evaluate=REF.evaluate
        original_lib=REF.LIB
        def fail(*args,**kwargs): raise RuntimeError('deliberate observer test')
        REF.evaluate=fail
        try:
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                vb.evaluate(REF,m,e,780,0)
            self.assertIs(REF.LIB,original_lib)
        finally:
            REF.evaluate=original_evaluate
            e.close()

    def test_stopped_score(self):
        self.assertEqual(vb.stopped_score([], .999),0)
        self.assertEqual(vb.stopped_score([1.,-10.],.999),1.-.999*10.)
        self.assertEqual(vb.stopped_score([1.,2.,3.],1.),6.)
        self.assertNotEqual(vb.stopped_score([1.,-10.],.999),vb.stopped_score([1.,-10.],.99))

    def test_summary_never_relabels_legacy_score(self):
        records=[dict(mode=mode,ending='timeout',steps=n,discounted=1.)
                 for mode,n in [('deterministic',1000),('stochastic',1500)]]
        old=vb.summarize(records)
        self.assertIsNone(old['stochastic']['mean_discounted_gamma999'])
        for r in records: r['discounted_gamma999']=2.
        new=vb.summarize(records)
        self.assertEqual(new['stochastic']['mean_discounted'],1.)
        self.assertEqual(new['stochastic']['mean_discounted_gamma999'],2.)

    def test_fixed_settings_and_total_budgets(self):
        self.assertEqual(vb.SEEDS,(201,202,203,204))
        self.assertEqual(vb.CPS,(0,131072,262144,524288,786432,1048576))
        self.assertEqual(vb.SCALE,1/(1-.999))
        for n_steps in vb.ARMS.values():
            updates=1048576//n_steps
            self.assertEqual(updates*(n_steps//64)*10,163840)
            self.assertEqual(updates*n_steps*10,10485760)
        m,e=vb.build(REF,781,2048)
        try:
            self.assertEqual(m.policy.optimizer.defaults['betas'],(.9,.999))
            self.assertEqual(m.gamma,.999)
            self.assertEqual(m.gae_lambda,1.)
            self.assertEqual(m.batch_size,64)
            self.assertEqual(m.n_epochs,10)
            self.assertTrue(m.normalize_advantage)
            self.assertIsNone(m.target_kl)
            self.assertIsNone(m.clip_range_vf)
        finally: e.close()

def main():
    global REF,OUT
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    OUT=args.out
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    REF,receipt=vb.pool.load(args.raw,OUT/'native')
    vb.write(OUT/'input.json',receipt)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(HarnessTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    vb.write(OUT/'TESTS.json',dict(tests=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),passed=result.wasSuccessful(),main_training_trials=0,
        control_training_interactions=2*(2048+8192)))
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__=='__main__': main()
