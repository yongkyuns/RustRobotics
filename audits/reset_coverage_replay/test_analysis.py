"""Synthetic mutation tests; optional complete actual-cohort qualification."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('replay_analysis',Path(__file__).with_name('analysis.py'))
a=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(a)


def synthetic():
    result={}
    for seed in a.SEEDS:
        for arm in a.ARMS:
            for k in a.expected_keys(seed,arm):
                s,ar,cp,p,rep=k
                cap=30000 if p in a.LONG[:2] else 6000 if p=='outward-long' else 2048
                result[k]={'seed':str(s),'arm':ar,'checkpoint':str(cp),'panel':p,'rep':str(rep),
                           'key':f'{s}-{p}-{rep}','cap':str(cap),'steps':str(cap),'ending':'timeout',
                           'total':'1.0','discounted':'1.0','max_position':'0.0','max_angle':'0.0',
                           'centered':'true','force_rms':'0.0'}
    return result


def fail(row):
    row.update(ending='position',steps='10',max_position='2.41',centered='false')


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=synthetic()

    def test_full_accounting_and_controls(self):
        result,plan=a.analyze(self.base)
        self.assertEqual(result['evaluation_rows'],18688)
        self.assertEqual(len(plan),64)
        self.assertEqual(result['final_lost_cases'],0)

    def test_final_losses_include_both_arms_and_late_matched_policy(self):
        data=copy.deepcopy(self.base);k=(41002,a.ARMS[1],4608,'stress-stoch',14);fail(data[k])
        result,plan=a.analyze(data)
        self.assertEqual(result['final_lost_cases'],1)
        self.assertEqual(result['late_lost_cases'],1)
        for ar,cp in [(a.ARMS[0],4608),(a.ARMS[1],4608),(a.ARMS[1],4096)]:
            self.assertIn(a.case_id((41002,ar,cp,'stress-stoch',14)),plan)

    def test_no_unmeasured_4096_rep_is_invented(self):
        data=copy.deepcopy(self.base);k=(41002,a.ARMS[1],4608,'stress-stoch',189);fail(data[k])
        result,plan=a.analyze(data)
        self.assertEqual(result['final_lost_cases'],1)
        self.assertEqual(result['late_lost_cases'],0)
        self.assertNotIn(a.case_id((41002,a.ARMS[1],4096,'stress-stoch',189)),plan)

    def test_ordinary_arm_late_loss_is_not_dropped(self):
        data=copy.deepcopy(self.base);fail(data[41001,a.ARMS[0],4608,'stress-det',7])
        result,plan=a.analyze(data)
        self.assertEqual(result['late_lost_cases'],1)
        self.assertEqual(result['final'][str(41001)]['stress-det']['gained'],1)
        self.assertIn(a.case_id((41001,a.ARMS[0],4096,'stress-det',7)),plan)

    def test_changed_random_key_rejected(self):
        data=copy.deepcopy(self.base);data[41002,a.ARMS[1],4608,'reset-det',0]['key']='changed'
        with self.assertRaisesRegex(ValueError,'random key'):
            a.analyze(data)

    def test_missing_measurement_rejected(self):
        data=dict(self.base);data.pop(next(iter(data)))
        with self.assertRaisesRegex(ValueError,'incomplete cohort'):
            a.analyze(data)

    def test_input_duplicate_rejected(self):
        rows=[r for k,r in self.base.items() if k[:2]==(41001,a.ARMS[0])]
        with self.assertRaisesRegex(ValueError,'duplicate case'):
            a.validate_cases(rows+[rows[0]],41001,a.ARMS[0])

    def test_false_timeout_rejected(self):
        rows=copy.deepcopy([r for k,r in self.base.items() if k[:2]==(41001,a.ARMS[0])]);rows[0]['steps']='10'
        with self.assertRaisesRegex(ValueError,'false completion'):
            a.validate_cases(rows,41001,a.ARMS[0])

    def test_boundary_mismatch_rejected(self):
        rows=copy.deepcopy([r for k,r in self.base.items() if k[:2]==(41001,a.ARMS[0])]);rows[0]['max_position']='3'
        with self.assertRaisesRegex(ValueError,'boundary mismatch'):
            a.validate_cases(rows,41001,a.ARMS[0])

    def test_missing_seed_not_treated_as_more_episodes(self):
        with self.assertRaisesRegex(ValueError,'four pairs'):
            a.interval([0]*256)

    def test_seed_statistics(self):
        r=a.interval([10/256,-14/256,10/256,0])
        self.assertAlmostEqual(r['mean'],6/1024)
        self.assertAlmostEqual(r['interval_99'][0],-.12371089307073144)

    def test_manifest_corruption_and_membership(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'x').write_text('original')
            (p/'manifest.json').write_text(json.dumps({'x':a.hashlib.sha256(b'original').hexdigest()}))
            self.assertEqual(a.manifest(p),1)
            (p/'x').write_text('changed')
            with self.assertRaisesRegex(ValueError,'payload hash'):
                a.manifest(p)
            (p/'extra').write_text('unexpected')
            with self.assertRaisesRegex(ValueError,'membership'):
                a.manifest(p)


@unittest.skipUnless(os.environ.get('REPLAY_INPUT'),'complete original artifact bundle not supplied')
class ActualCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['REPLAY_INPUT'])
        cls.cases,cls.members=a.read_cohort(cls.root)
        cls.result,cls.plan=a.analyze(cls.cases)

    def test_manifest_population(self):
        self.assertEqual(self.members['build'],122)
        self.assertEqual(sum(self.members.values()),2930)

    def test_independent_statistics_match_original(self):
        original=json.loads((self.root/'reset-coverage-comparison/cohort-result.json').read_text())
        a.compare_original(self.result,original)
        self.assertFalse(original['local_component_screen'])

    def test_all_losses_retained(self):
        self.assertEqual(self.result['final_lost_cases'],48)
        self.assertEqual(self.result['late_lost_cases'],28)
        self.assertEqual(self.result['final_loss_endings'],{'position':44,'angle':4,'both':0})
        self.assertEqual(len(self.plan),202)

    def test_seed41002_late_deterioration(self):
        x=self.result['late_same_64']['41002-extra-transient']
        self.assertEqual((x['stress-det']['before'],x['stress-det']['after']),(60,57))
        self.assertEqual((x['stress-stoch']['before'],x['stress-stoch']['after']),(62,56))
        self.assertEqual(sum(v['lost'] for v in x.values()),12)
        self.assertEqual(sum(v['gained'] for v in x.values()),0)

    def test_seed41002_early_advantage_is_not_final_advantage(self):
        for p in a.SHORT:
            counts=[sum(self.cases[41002,arm,1024,p,i]['ending']=='timeout' for i in range(64)) for arm in a.ARMS]
            self.assertGreater(counts[1],counts[0])
            self.assertLess(self.result['final']['41002'][p]['net'],0)

    def test_repeat_analysis_is_identical(self):
        again,plan=a.analyze(self.cases)
        self.assertEqual(again,self.result);self.assertEqual(plan,self.plan)


if __name__=='__main__':unittest.main()
