import copy
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import evidence as e
import prepare


def example_cases(seed=41001,arm='extra-ordinary'):
    result=[]
    for cp in e.CPS:
        for panel in e.SHORT+(e.LONG if cp==4608 else ()):
            n=32 if panel in e.LONG else 256 if cp==4608 else 64
            for rep in range(n):
                result.append(dict(seed=seed,arm=arm,checkpoint=cp,panel=panel,rep=rep,key=100000+e.CAPS[panel]+rep,
                    cap=e.CAPS[panel],steps=e.CAPS[panel],ending='timeout',total=20.0,discounted=10.0,
                    max_position=.2,max_angle=.1,centered='true',force_rms=.5))
    return result


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def check_cases(self,data):
        p=self.root/'cases.csv';write_csv(p,data);return e.read_cases(p,41001,'extra-ordinary')
    def test_complete_registered_case_shape(self):
        self.assertEqual(len(self.check_cases(example_cases())),2336)
    def test_missing_and_duplicate_cases_rejected(self):
        data=example_cases()
        with self.assertRaises(ValueError):self.check_cases(data[:-1])
        data[-1]=data[0]
        with self.assertRaises(ValueError):self.check_cases(data)
    def test_nonfinite_and_false_timeout_rejected(self):
        for field,value in [('discounted',float('nan')),('steps',3),('max_position',3.0)]:
            data=example_cases();data[0][field]=value
            with self.assertRaises(ValueError):self.check_cases(data)
    def test_wrong_seed_or_arm_rejected(self):
        data=example_cases();data[0]['seed']=41002
        with self.assertRaises(ValueError):self.check_cases(data)
    def test_boundary_is_float32_native_boundary(self):
        data=example_cases();data[0]['max_position']=2.4
        self.assertEqual(len(self.check_cases(data)),2336)
        data[0]['max_position']=2.5;data[0]['ending']='position';data[0]['centered']='false';data[0]['steps']=12
        self.assertEqual(len(self.check_cases(data)),2336)
    def test_casewise_losses_not_hidden_by_same_total(self):
        a=[dict(rep=i,key=i,ending='timeout' if i else 'position',discounted=2,total=4) for i in range(3)]
        b=[dict(rep=i,key=i,ending='timeout' if i!=1 else 'position',discounted=1,total=3) for i in range(3)]
        p=e.paired(a,b);self.assertEqual((p['gained'],p['lost'],p['net']),(1,1,0))
    def test_unpaired_key_rejected(self):
        a=[dict(rep=i,key=i,ending='timeout',discounted=1,total=2) for i in range(3)]
        b=copy.deepcopy(a);b[0]['key']=42
        with self.assertRaises(ValueError):e.paired(a,b)
    def test_training_seed_interval_uses_four_not_episode_count(self):
        x=e.interval([1,2,3,4],5.840909309733352)
        self.assertEqual(x['n'],4);self.assertAlmostEqual(x['mean'],2.5)
        self.assertGreater(x['interval_99'][1]-x['mean'],3)
    def test_incomplete_cohort_never_becomes_zero_score(self):
        with self.assertRaises(ValueError):e.combine([])
    def test_initial_training_and_evaluation_budgets(self):
        self.assertEqual(4608*24,110592);self.assertEqual(4608*1536*4,28311552)
        self.assertEqual(4608*4*1152,21233664)
        self.assertEqual(len(example_cases()),2336)
    def test_training_keys_have_no_collisions(self):
        keys={0x5243545241490001^(seed<<32)^(u<<8)^stream for seed in e.SEEDS for u in range(1,4609) for stream in range(4)}
        self.assertEqual(len(keys),4*4608*4)
        evaluation=set()
        for seed in e.SEEDS:
            for stress in (False,True):
                for tag in range(1,7):
                    for rep in range(256):
                        for domain in (0x5250545245560001,0x5250545345560001):
                            evaluation.add(domain^((seed+0x78000000+(0x00800000 if stress else 0))<<32)^(tag<<16)^rep)
        self.assertFalse(keys&evaluation)
    def test_exact_normalization_constant_and_signed_zero(self):
        self.assertEqual(e.normalize([2,2,2]),[0,0,0]);self.assertEqual(e.normalize([-1,1]),[-1,1])
        self.assertNotEqual(e.float_bits([-0.0]),e.float_bits([0.0]))
    def test_return_boundaries_and_far_bootstrap(self):
        d=dict(rewards=[-10,2],values=[0,0],terminals=[True,False],timeouts=[False,False],ends=[True,True],bootstraps=[0,4],remaining=[1,1],raw1=[-10,e.f32(2+e.f32(e.f32(.995)*4))],returns1=[-10,e.f32(2+e.f32(e.f32(.995)*4))])
        e.check_targets(d)
        for field,value in [('bootstraps',[7,4]),('raw1',[-9,6]),('ends',[False,True])]:
            bad=copy.deepcopy(d);bad[field]=value
            with self.assertRaises(ValueError):e.check_targets(bad)
    def test_missing_source_anchor_and_duplicate_attachment_rejected(self):
        text='fn example() {\n'+prepare.UNION+'\n    '+prepare.OLD_BATCH+'\n'+prepare.EVAL_RETURN+'\n}'
        changed=prepare.transform_repeat(text)
        self.assertEqual(prepare.restore_repeat(changed),text)
        with self.assertRaises(ValueError):prepare.transform_repeat(changed)
        with self.assertRaises(ValueError):prepare.transform_repeat(text.replace(prepare.UNION,''))
    def test_actual_prepared_source_transform_is_reversible(self):
        path=os.environ.get('COVERAGE_REPEAT_FIXTURE')
        self.assertIsNotNone(path,'set COVERAGE_REPEAT_FIXTURE to the inherited prepared source')
        text=Path(path).read_text()
        self.assertEqual(prepare.restore_repeat(prepare.transform_repeat(text)),text)
    def test_only_three_changes_to_inherited_driver(self):
        text=Path(os.environ['COVERAGE_REPEAT_FIXTURE']).read_text()
        modified=prepare.transform_repeat(text)
        self.assertEqual(modified.count('reset_coverage_hooks'),2)
        self.assertIn('cost.actor_steps = s.config.ppo.epochs_per_update',modified)
    def test_no_direct_critic_or_actor_optimizer_replacement_in_hook(self):
        source=Path(__file__).with_name('hooks.rs').read_text()
        self.assertNotIn('optimizer.step',source)
        self.assertNotIn('PpoTrainerSession',source)
        self.assertIn('batch.advantages = normalize(&raw)',source)
    def test_manifest_refuses_to_invent_outputs(self):
        e.manifest(self.root)
        self.assertEqual(json.loads((self.root/'manifest.json').read_text()),{})


if __name__=='__main__':unittest.main()
