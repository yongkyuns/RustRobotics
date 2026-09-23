import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import zipfile
import evidence
import prepare


def fixture(scores):
    result=[];previous=None
    for u,v in enumerate(scores,4128):
        result.append({'update':u,'nominal_det_mean':v,'previous_minus_current':0.0 if previous is None else previous-v})
        previous=v
    return result


def archive(payload):
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
        z.writestr('manifest.json',json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in payload.items()}))
    return output.getvalue()


class Tests(unittest.TestCase):
    def test_selection_largest_positive_earliest(self):
        s=[10.0]*97;s[1:]=[9.0]*96;s[3:]=[8.0]*94
        self.assertEqual(evidence.select(fixture(s)),4129)
        s[6:]=[6.0]*91
        self.assertEqual(evidence.select(fixture(s)),4134)

    def test_no_drop_does_not_manufacture_witness(self):
        self.assertIsNone(evidence.select(fixture(list(range(97)))))

    def test_bad_curve_rejected(self):
        rows=fixture([1.0]*97)
        with self.assertRaises(ValueError):evidence.select(rows[:-1])
        rows[1]['previous_minus_current']=1.0
        with self.assertRaises(ValueError):evidence.select(rows)
        rows=fixture([math.nan]*97)
        with self.assertRaises(ValueError):evidence.select(rows)

    def test_immutable_verified_archive(self):
        raw=archive({'a':b'abc'});sha=hashlib.sha256(raw).hexdigest()
        self.assertEqual(evidence.verified_zip(raw,sha),{'a':b'abc'})
        with self.assertRaises(ValueError):evidence.verified_zip(raw,'0'*64)

    def test_paths_rejected(self):
        for name in ('../bad','/bad','a\\bad'):
            raw=archive({name:b'x'})
            with self.assertRaises(ValueError):evidence.verified_zip(raw,hashlib.sha256(raw).hexdigest())

    def test_member_corruption_rejected(self):
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w') as z:
            z.writestr('a',b'bad');z.writestr('manifest.json',json.dumps({'a':hashlib.sha256(b'good').hexdigest()}))
        b=raw.getvalue()
        with self.assertRaises(ValueError):evidence.verified_zip(b,hashlib.sha256(b).hexdigest())

    def test_real_hook_insertion_only_adds_observers(self):
        path=Path(os.environ['LATE_TRAINER_FIXTURE'])
        text=path.read_text();changed=prepare.instrument(text)
        self.assertEqual(changed.count('late_nominal_adam::'),4)
        restored=changed.removesuffix(prepare.MODULE)
        for _,addition,_ in prepare.INSERTIONS:restored=restored.replace(addition,'')
        self.assertEqual(restored,text)
        with self.assertRaises(ValueError):prepare.instrument(changed)
        with self.assertRaises(ValueError):prepare.instrument('missing')

    def test_child_attachment_preserves_original(self):
        self.assertEqual(prepare.attach('original'), 'original'+prepare.CHILD)
        with self.assertRaises(ValueError):prepare.attach(prepare.attach('original'))

    def test_paired_gains_and_losses_are_separate(self):
        a=[dict(key=1,discounted=3.,total=5.,ending='timeout'),dict(key=2,discounted=4.,total=2.,ending='position')]
        b=[dict(key=1,discounted=2.,total=2.,ending='position'),dict(key=2,discounted=3.,total=5.,ending='timeout')]
        r=evidence.contrast(a,b)
        self.assertEqual((r['gained'],r['lost']),(1,1));self.assertEqual(r['discounted']['mean'],1.)
        self.assertEqual(r['total']['mean'],0.)
        with self.assertRaises(ValueError):evidence.contrast(a,list(reversed(b)))

    def test_declared_budgets(self):
        self.assertEqual(97*3*64,18624)
        self.assertEqual(3*3*512,4608)
        self.assertEqual(1024*8,8192)
        self.assertEqual(128*48,6144)

    def test_outcomes_require_all_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'outcomes.csv'
            path.write_text('seed,arm,checkpoint,panel,rep,key,cap,steps,ending,discounted,total,max_position,max_angle,force_rms\n')
            with self.assertRaises(ValueError):evidence.outcomes(path,('incoming',),{'incoming':{4140}},512)

if __name__=='__main__': unittest.main()
