"""Software tests using genuine ten-pass controls and labeled algebra fixtures.
No test fabricates a DECAY4 training result or executes a learning update.
"""
from __future__ import annotations
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import numpy as np
from crossover_core import *
from pack_inputs import wanted_paths, validate_zip_members, export
from run_crossover import check_input_bundle
from physical_reader import read_group
from analyze import group_summary, summarize_factorial, reduce_group
ROOT=Path(__file__).resolve().parents[1]

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=ROOT/'reference/results/201-decay/1310720'
        cls.weights=load_npz(p/'policy.npz')
        cls.donor=load_npz(ROOT/'reference/results/201-constant/1310720/policy.npz')
        cls.norm=json.loads((p/'NORMALIZER.json').read_text())
        cls.group=load_npz(ROOT/'controls/201/broad-stochastic.npz')
        cls.terminal_group=load_npz(ROOT/'controls/204/difficult-stochastic.npz')
        cls.rows=read_group(cls.group,cls.weights,cls.norm,'stochastic')[0]
        cls.costs=effort_metrics(cls.group)

    def test_real_cross_changes_only_sigma(self):
        w=crossed_policy(self.weights,self.donor)
        self.assertNotEqual(w['log_std'].tobytes(),self.weights['log_std'].tobytes())
        self.assertEqual(w['log_std'].tobytes(),self.donor['log_std'].tobytes())
        for k in w:
            if k!='log_std':self.assertEqual(w[k].tobytes(),self.weights[k].tobytes())
    def test_cross_no_alias_or_source_mutation(self):
        before={k:v.tobytes() for k,v in self.weights.items()}
        w=crossed_policy(self.weights,self.donor)
        for k in w:self.assertFalse(np.shares_memory(w[k],self.weights[k]))
        w['action_net.bias'][0]+=1
        self.assertEqual(before,{k:v.tobytes() for k,v in self.weights.items()})
    def test_identical_diagonal_parameters(self):
        self.assertTrue(equal_arrays(self.weights,crossed_policy(self.weights,self.weights)))
    def test_policy_rejects_missing_extra_shape_dtype_nonfinite(self):
        variants=[]
        w=copy.deepcopy(self.weights);del w['log_std'];variants.append(w)
        w=copy.deepcopy(self.weights);w['extra']=np.zeros(1,np.float32);variants.append(w)
        w=copy.deepcopy(self.weights);w['log_std']=np.zeros(2,np.float32);variants.append(w)
        w=copy.deepcopy(self.weights);w['log_std']=w['log_std'].astype(float);variants.append(w)
        w=copy.deepcopy(self.weights);w['action_net.bias'][0]=np.nan;variants.append(w)
        for w in variants:
            with self.subTest(keys=list(w)):
                with self.assertRaises(ValueError):policy_checked(w)
    def test_sigma_underflow_is_rejected(self):
        w=copy.deepcopy(self.weights);w['log_std'][0]=-1000
        with self.assertRaises(ValueError):policy_checked(w)
    def test_normalizer_contract(self):
        normalizer_checked(self.norm)
        changes=[('var',[1,1,0,1]),('mean',[1,2,3]),('clip',9.),('epsilon',0.),('enabled',False),('count',True),('count',float('nan'))]
        for k,v in changes:
            n=copy.deepcopy(self.norm);n[k]=v
            with self.subTest(field=k):
                with self.assertRaises(ValueError):normalizer_checked(n)
    def test_genuine_physical_reader_and_cost_reconciliation(self):
        self.assertEqual(len(self.rows),64)
        for r,c in zip(self.rows,self.costs):
            expected=c['reward_base']-c['state_penalty_nonterminal']-c['effort_penalty_nonterminal']+c['reward_rounding_residual']
            self.assertAlmostEqual(r['raw_return'],expected,places=10)
            self.assertEqual(r['steps'],c['valid_steps'])
    def test_terminal_effort_exclusion_on_genuine_records(self):
        cs=effort_metrics(self.terminal_group)
        failed=np.flatnonzero(self.terminal_group['endings']==1)
        self.assertGreater(len(failed),0)
        for i in failed:
            n=int(self.terminal_group['lengths'][i]);f=self.terminal_group['trace'][i,:n,9].astype(float)
            self.assertEqual(cs[i]['nonterminal_steps'],n-1)
            self.assertAlmostEqual(cs[i]['effort_penalty_nonterminal'],float((.001*f[:-1]**2).sum()),places=12)
            self.assertAlmostEqual(cs[i]['force_square_sum_all_steps'],float(f@f),places=10)
    def test_live_horizon_includes_last_effort(self):
        i=int(np.flatnonzero(self.group['endings']==2)[0]);f=self.group['trace'][i,:,9].astype(float)
        self.assertEqual(self.costs[i]['nonterminal_steps'],1500)
        self.assertAlmostEqual(self.costs[i]['effort_penalty_nonterminal'],float((.001*f*f).sum()),places=10)
    def test_one_step_terminal_algebra_fixture(self):
        g=copy.deepcopy(self.group)
        g['trace'][0]=0;g['trace'][0,0,0]=np.float32(3.)
        g['trace'][0,0,8]=-10;g['trace'][0,0,9]=20;g['trace'][0,0,10]=1
        g['lengths'][0]=1;g['endings'][0]=1
        c=effort_metrics(g)[0]
        self.assertEqual(c['effort_penalty_nonterminal'],0.)
        self.assertEqual(c['force_rms_all_steps'],20.)
        self.assertEqual(c['reward_base'],-10.)
    def test_padding_rejected(self):
        g=copy.deepcopy(self.terminal_group);i=int(np.flatnonzero(g['endings']==1)[0]);n=int(g['lengths'][i])
        g['trace'][i,n,9]=1
        with self.assertRaises(ValueError):effort_metrics(g)
    def test_integer_length_required(self):
        g=copy.deepcopy(self.group);g['lengths']=g['lengths'].astype(float)
        with self.assertRaises(ValueError):effort_metrics(g)
    def test_false_ending_rejected(self):
        g=copy.deepcopy(self.group);i=int(np.flatnonzero(g['endings']==2)[0]);g['endings'][i]=1
        with self.assertRaises(ValueError):effort_metrics(g)
    def test_force_mapping_rejected(self):
        g=copy.deepcopy(self.group);g['trace'][0,0,9]+=1
        with self.assertRaises(ValueError):effort_metrics(g)
    def test_one_ulp_physical_reward_corruption_rejected(self):
        g=copy.deepcopy(self.group);g['trace'][0,0,8]=np.nextafter(g['trace'][0,0,8],np.float32(np.inf))
        with self.assertRaises(AssertionError):read_group(g,self.weights,self.norm,'stochastic')
    def test_changed_policy_rejected_by_independent_inference(self):
        w=copy.deepcopy(self.weights);w['action_net.bias'][0]+=.1
        with self.assertRaises(AssertionError):read_group(self.group,w,self.norm,'stochastic')
    def test_changed_normalizer_rejected_by_independent_inference(self):
        n=copy.deepcopy(self.norm);n['mean'][0]+=.5
        with self.assertRaises(AssertionError):read_group(self.group,self.weights,n,'stochastic')
    def test_key_reordering_rejected(self):
        a=[dict(key=i,value=float(i)) for i in range(64)];b=list(reversed(a))
        with self.assertRaises(ValueError):paired_metrics(a,b,'value')
    def test_casewise_gains_not_inferred_from_totals(self):
        a=[dict(key=i,value=i<32) for i in range(64)];b=[dict(key=i,value=i>=32) for i in range(64)]
        z=paired_metrics(a,b,'value')
        self.assertEqual(z,dict(mean_difference=0.,increases=32,decreases=32))
    def test_factorial_orientation(self):
        f=factorial(dict(M4S4=1.,M4S10=3.,M10S4=4.,M10S10=8.))
        self.assertEqual(f['sigma_S10_minus_S4_at_M4'],2.)
        self.assertEqual(f['sigma_S10_minus_S4_at_M10'],4.)
        self.assertEqual(f['interaction'],2.)
        self.assertEqual(f['diagonal_M4S4_minus_M10S10'],-7.)
    def test_factorial_incomplete_nonfinite_rejected(self):
        for v in ({'M4S4':1.},dict(M4S4=1.,M4S10=2.,M10S4=3.,M10S10=np.nan)):
            with self.assertRaises(ValueError):factorial(v)
    def test_exact_four_seed_uncertainty(self):
        f=four_seed_interval({201:1.,202:2.,203:3.,204:4.})
        self.assertEqual(f['n_training_seeds'],4);self.assertEqual(f['mean'],2.5)
        self.assertAlmostEqual(f['standard_error'],np.sqrt(5/3)/2)
        # Fixed df=3 two-sided 99% Student critical value, independently tabulated.
        self.assertAlmostEqual((f['pointwise_99_interval'][1]-2.5)/f['standard_error'],5.8409093097,places=8)
    def test_modes_cannot_be_extra_seeds(self):
        with self.assertRaises(ValueError):four_seed_interval({s:1. for s in range(201,209)})
    def test_nonfinite_seed_effect_rejected(self):
        with self.assertRaises(ValueError):four_seed_interval({201:0.,202:0.,203:0.,204:np.inf})
    def test_complete_factorial_groups_required(self):
        with self.assertRaises(ValueError):summarize_factorial({})
    def test_full_factorial_reduction_algebra_fixture(self):
        rows=[dict(p,**{k:v for k,v in c.items() if k!='key'}) for p,c in zip(self.rows,self.costs)]
        groups={}
        for s in SEEDS:
            for arm,(mi,si) in ARMS.items():
                for p in PANELS:
                    for m in MODES:
                        r=copy.deepcopy(rows)
                        for row in r:row['raw_return']+=float(mi+si+s-201)
                        groups[s,arm,p,m]=r
        r=summarize_factorial(groups)
        self.assertEqual(len(r['per_seed']),16)
        entry=next(x for x in r['pooled'] if x['metric']=='raw_return_mean' and x['contrast']=='interaction')
        self.assertAlmostEqual(entry['mean'],0.,places=10)
        self.assertEqual(entry['n_training_seeds'],4)
    def test_required_input_count_and_final_checkpoint(self):
        names=wanted_paths();self.assertEqual(len(names),65);self.assertEqual(len(set(names)),65)
        self.assertEqual(sum(n.endswith('policy.npz') for n in names),8)
        self.assertEqual(sum('stochastic.npz' in n or 'deterministic.npz' in n for n in names),32)
        self.assertTrue(all('/1310720/' in n for n in names if n.startswith('results/')))
    def test_zip_traversal_and_duplicate_rejected(self):
        for names in (['../x'],['/x'],['a\\b'],['a','a']):
            with self.assertRaises(ValueError):validate_zip_members(names)
    def test_wrong_archive_no_input_output(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);archive=p/'wrong.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('README','Not the original dataset')
            with self.assertRaisesRegex(ValueError,'digest'):export(archive,p/'export')
            self.assertFalse((p/'export').exists())
    def test_missing_input_fails_before_runtime(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'missing'
            script=f"import sys;from pathlib import Path;from run_crossover import check_input_bundle\ntry: check_input_bundle(Path({str(p)!r}))\nexcept ValueError: pass\nelse: raise AssertionError('accepted missing data')\nassert 'torch' not in sys.modules\n"
            c=subprocess.run([sys.executable,'-c',script],cwd=ROOT/'code',capture_output=True,text=True)
            self.assertEqual(c.returncode,0,c.stderr)
    def test_cli_missing_input_has_no_success_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);out=p/'never.json'
            c=subprocess.run([sys.executable,str(ROOT/'code/run_crossover.py'),'check-inputs','--inputs',str(p/'missing'),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(c.returncode,2);self.assertFalse(out.exists())
            self.assertFalse(json.loads(c.stderr)['registered_crossover_complete'])
    def test_receipt_hash_and_count_corruption_rejected(self):
        src=ROOT/'controls/201/broad-stochastic.npz'
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'group.npz';p.write_bytes(src.read_bytes())
            receipt=json.loads(src.with_suffix('.json').read_text());receipt['native_transitions']+=1
            p.with_suffix('.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):reduce_group(p,self.weights,self.norm,'stochastic')
            receipt['native_transitions']-=1;receipt['trace_sha256']='0'*64
            p.with_suffix('.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):reduce_group(p,self.weights,self.norm,'stochastic')
    def test_actual_deterministic_sigma_invariance(self):
        for s in SEEDS:
            a=load_npz(ROOT/f'controls/{s}/broad-deterministic.npz')
            b=load_npz(ROOT/f'controls/{s}/sigma_changed_deterministic_control.npz')
            self.assertTrue(equal_arrays(a,b))
    def test_control_not_labeled_crossover_completion(self):
        r=json.loads((ROOT/'controls/CONTROL_VERIFIED.json').read_text())
        self.assertEqual(r['status'],'software_controls_only')
        self.assertFalse(r['registered_crossover_complete'])
        self.assertFalse(r['four_pass_checkpoint_available'])
        self.assertEqual(r['training_steps'],0);self.assertEqual(r['optimizer_steps'],0)

if __name__=='__main__':unittest.main()
