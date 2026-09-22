"""Critic pilot controls. Actual-archive tests require CRITIC_RESULT/CRITIC_BUILD.

Synthetic controls always execute. Artifact-dependent tests skip explicitly when
inputs are absent; they do not substitute synthetic observations for the witness.
"""
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import study as s


def panel():
    rng = np.random.default_rng(9)
    return s.Panel(rng.normal(size=(1024,4)), rng.normal(size=1024),
                   np.repeat(np.arange(2),512), np.tile(np.arange(512),2))


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_network_layout_and_exact_initial_bytes(self):
        w = np.random.default_rng(11).normal(0,.05,4545).astype('f4')
        model = s.Critic(w)
        self.assertEqual(sum(p.numel() for p in model.parameters()),4545)
        self.assertEqual(w.tobytes(), model.snapshot().tobytes())

    def test_network_forward_independent_numpy(self):
        w = np.random.default_rng(12).normal(0,.05,4545).astype('f4')
        x = np.random.default_rng(13).normal(size=(51,4)).astype('f4')
        actual = s.Critic(w)(torch.tensor(x)).detach().numpy()
        np.testing.assert_allclose(actual, s.base.forward(w.astype(float),x),atol=1e-7,rtol=1e-5)

    def test_network_mse_gradient_directional_difference(self):
        rng=np.random.default_rng(14)
        w=rng.normal(0,.1,4545).astype('f4')
        x=rng.normal(size=(19,4)); y=rng.normal(size=19)
        d=rng.normal(size=4545);d/=np.linalg.norm(d)
        m=s.Critic(w).double()
        loss=((m(torch.tensor(x))-torch.tensor(y))**2).mean();loss.backward()
        gradient=np.concatenate([p.grad.numpy().ravel() for p in m.parameters()])
        h=1e-5; objective=lambda a:np.mean((s.base.forward(a,x)-y)**2)
        fd=(objective(w.astype(float)+h*d)-objective(w.astype(float)-h*d))/(2*h)
        self.assertAlmostEqual(float(gradient@d),float(fd),places=8)

    def test_optimizer_explicit_fresh_configuration(self):
        opt=s.optimizer(s.Critic(np.ones(4545)*.01))
        self.assertEqual(len(opt.state),0)
        group=opt.param_groups[0]
        for k,v in dict(lr=3e-4,betas=(.9,.999),eps=1e-5,weight_decay=0,
                        amsgrad=False,foreach=False,fused=False).items():self.assertEqual(group[k],v)

    def test_balanced_sampler_exact_counts(self):
        p=panel()
        for idx in s.index_batches(p,True,100):
            self.assertEqual(len(idx),256)
            self.assertEqual(int((p.phase[idx]<64).sum()),128)
            self.assertTrue((idx>=0).all() and (idx<len(p.x)).all())

    def test_uniform_sampler_range_and_repeatability(self):
        a=np.stack(list(s.index_batches(panel(),False,16)))
        b=np.stack(list(s.index_batches(panel(),False,16)))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(16,256))
        self.assertTrue((a>=0).all() and (a<1024).all())

    def test_empty_stratum_rejected(self):
        p=panel();p=replace(p,phase=np.zeros(1024,dtype=int))
        with self.assertRaisesRegex(ValueError,'empty phase'):next(s.index_batches(p,True,1))

    def test_split_entire_replicates(self):
        p=s.Panel(np.zeros((32*512,4)),np.zeros(32*512),np.repeat(np.arange(32),512),np.tile(np.arange(512),32))
        a,b=s.split(p)
        self.assertEqual(set(a.group),set(range(24)))
        self.assertEqual(set(b.group),set(range(24,32)))
        self.assertEqual(len(a.x),12288); self.assertEqual(len(b.x),4096)

    def test_split_missing_replicate_rejected(self):
        with self.assertRaises(ValueError):s.split(panel())

    def test_split_duplicate_phase_rejected(self):
        p=s.Panel(np.zeros((32*512,4)),np.zeros(32*512),np.repeat(np.arange(32),512),np.tile(np.arange(512),32))
        p.phase[1]=0
        with self.assertRaisesRegex(ValueError,'phases'):s.split(p)

    def test_fit_api_cannot_read_evaluation_targets(self):
        self.assertEqual(list(inspect.signature(s.fit).parameters), ['flat','panel','balanced','out','steps','checkpoints'])
        source=inspect.getsource(s.fit)
        self.assertNotIn('holdout',source); self.assertNotIn('q_draws',source)
        self.assertNotIn('v_draws',source);self.assertNotIn('witness',source)

    def test_small_fits_bitwise_repeatable(self):
        p=panel();w=np.random.default_rng(3).normal(0,.05,4545).astype('f4')
        with tempfile.TemporaryDirectory() as d:
            a=s.fit(w,p,True,Path(d)/'a',steps=16,checkpoints=(0,16))
            b=s.fit(w,p,True,Path(d)/'b',steps=16,checkpoints=(0,16))
            np.testing.assert_array_equal(a[16],b[16]);self.assertFalse(np.array_equal(w,a[16]))
            self.assertEqual((Path(d)/'a/sampled-indices.u32').read_bytes(),(Path(d)/'b/sampled-indices.u32').read_bytes())
            self.assertEqual((Path(d)/'a/losses.csv').read_bytes(),(Path(d)/'b/losses.csv').read_bytes())

    def test_fitting_out_exists_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):s.fit(np.zeros(4545),panel(),False,Path(d),steps=1)

    def test_error_metrics_group_and_phase(self):
        p=panel(); error=np.where(p.phase<64,2.,1.)
        m=s.errors(p.y+error,p.y,p.group,p.phase)
        self.assertEqual(m['early_rmse'],2.);self.assertEqual(m['late_rmse'],1.)
        self.assertAlmostEqual(m['rmse'],np.sqrt(1.375))
        self.assertEqual(len(m['groups']),2)

    def test_screen_requires_every_condition(self):
        old={'witness':{'rmse':2.},'holdout':{'early_rmse':2.,'late_rmse':2.}}
        new={'witness':{'rmse':1.},'holdout':{'early_rmse':1.,'late_rmse':1.},'cosine_draws0_3':.1,'cosine_draws4_7':.2}
        self.assertTrue(s.screen(new,old)['passed'])
        new['cosine_draws4_7']=-.1
        self.assertFalse(s.screen(new,old)['passed'])

    def test_nan_weights_rejected(self):
        w=np.zeros(4545);w[1]=np.nan
        with self.assertRaises(ValueError):s.Critic(w)

    def test_nan_loss_rejected(self):
        p=panel();p.y[:]=np.nan
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'non-finite fitting loss'):
                s.fit(np.zeros(4545),p,False,Path(d)/'nan',steps=1)


@unittest.skipUnless(os.environ.get('CRITIC_RESULT') and os.environ.get('CRITIC_BUILD'), 'actual native archives not supplied')
class Actual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.inp=s.load_inputs(Path(os.environ['CRITIC_RESULT']),Path(os.environ['CRITIC_BUILD']))

    def test_all_manifests(self):self.assertEqual(self.inp.provenance['payload_hashes_verified'],337)

    def test_whole_trajectory_and_exact_initial_identity(self):
        self.assertEqual(set(self.inp.train.group),set(range(24)))
        self.assertEqual(set(self.inp.holdout.group),set(range(24,32)))
        self.assertEqual(s.sha(s.Critic(self.inp.critic).snapshot().astype('<f4').tobytes()),self.inp.provenance['critic_sha256'])

    def test_original_actor_gradient_binds_previous_report(self):
        m,_=s.evaluate(self.inp,self.inp.critic,self.inp.train)
        self.assertAlmostEqual(m['cosine_all8'],-.503033,places=5)
        self.assertAlmostEqual(m['witness']['rmse'],2.622377,places=4)

    def test_actual_numpy_torch_critic_all_rows(self):
        x=np.concatenate([self.inp.train.x,self.inp.holdout.x,self.inp.witness.x])
        actual=s.Critic(self.inp.critic)(torch.tensor(x,dtype=torch.float32)).detach().numpy()
        expected=s.base.forward(self.inp.critic,x)
        self.assertLess(float(np.max(np.abs(actual-expected))),1e-4)

    def test_witness_reference_not_present_in_training(self):
        training={row.astype('<f4').tobytes() for row in self.inp.train.x}
        self.assertFalse(any(row.astype('<f4').tobytes() in training for row in self.inp.witness.x))

    def test_holdout_perturbation_cannot_mutate_training(self):
        inp=replace(self.inp,holdout=replace(self.inp.holdout,y=self.inp.holdout.y+1e6),
                    v_draws=self.inp.v_draws+1e6)
        np.testing.assert_array_equal(inp.train.x,self.inp.train.x)
        np.testing.assert_array_equal(inp.train.y,self.inp.train.y)

    def test_hash_failure_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'corrupted.zip';p.write_bytes(b'not authentic')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                s.load_inputs(p,Path(os.environ['CRITIC_BUILD']))

    @unittest.skipUnless(os.environ.get('CRITIC_OUTPUT'),'completed output not supplied')
    def test_all_recorded_samplers_and_budgets(self):
        root=Path(os.environ['CRITIC_OUTPUT'])
        for arm in s.ARMS:
            p=self.inp.train if arm.startswith('independent') else self.inp.witness
            actual=np.fromfile(root/arm/'sampled-indices.u32',dtype='<u4').reshape(s.STEPS,s.BATCH)
            expected=np.stack(list(s.index_batches(p,arm=='independent-early-balanced')))
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(len(np.loadtxt(root/arm/'losses.csv',delimiter=',',skiprows=1)),s.STEPS)

    @unittest.skipUnless(os.environ.get('CRITIC_OUTPUT'),'completed output not supplied')
    def test_every_retained_metric_and_prediction_recomputes(self):
        root=Path(os.environ['CRITIC_OUTPUT']);report=json.loads((root/'report.json').read_text())
        for arm in s.ARMS:
            training=self.inp.train if arm.startswith('independent') else replace(self.inp.witness,y=self.inp.witness_returns)
            for step in s.CHECKPOINTS:
                flat=s.base.network((root/arm/f'critic-{step}.bin').read_bytes())
                actual,preds=s.evaluate(self.inp,flat,training)
                expected=report['arms'][arm]['checkpoints'][str(step)]
                self.assertEqual(actual,expected)
                saved=np.load(root/arm/f'predictions-{step}.npz')
                for k,v in preds.items():np.testing.assert_array_equal(saved[k],v)


if __name__=='__main__':unittest.main()
