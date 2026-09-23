import importlib.util
from pathlib import Path
import unittest

import analyze
import prepare

class Tests(unittest.TestCase):
    def test_unchanged_empty_child_attachment(self):
        text = 'fn original() {}\n'
        result = prepare.attach(text)
        self.assertEqual(result.removesuffix(prepare.HOOK), text)
        with self.assertRaises(ValueError): prepare.attach(result)

    def test_paired_case_tradeoff_not_hidden_by_totals(self):
        a = [dict(key=1,discounted=3.,ending='timeout'),dict(key=2,discounted=1.,ending='position')]
        b = [dict(key=1,discounted=1.,ending='position'),dict(key=2,discounted=2.,ending='timeout')]
        r = analyze.paired(a,b)
        self.assertEqual((r['completion_delta'],r['gained_cases'],r['lost_cases']), (0,1,1))
        self.assertEqual(r['return_delta'], .5)

    def test_reject_unpaired(self):
        a = [dict(key=i,discounted=1.,ending='timeout') for i in range(2)]
        b = list(reversed(a))
        with self.assertRaises(ValueError): analyze.paired(a,b)
        with self.assertRaises(ValueError): analyze.paired(a,a[:1])

    def test_identical_data_gives_zero_contrast(self):
        rows = [dict(key=i,discounted=float(i),ending='timeout') for i in range(8)]
        r = analyze.paired(rows,rows)
        self.assertEqual(r['return_delta'],0)
        self.assertEqual(r['descriptive_99_interval'],[0,0])

    def test_exact_declared_evaluation_budget(self):
        n = 0
        for arm in analyze.ARMS:
            for cp in analyze.CPS:
                for panel in analyze.SHORT+(analyze.LONG if cp==4224 else ()):
                    n += 512 if cp==4224 and panel in analyze.SHORT else 64
        self.assertEqual(n,6912)
        self.assertEqual(3*128*8*1536,4718592)
        self.assertEqual(3*3*8*2048,147456)

    def test_fit_budget_is_matched(self):
        self.assertEqual(12*1024,3*4096)
        self.assertEqual(12*1024//256,48)
        self.assertEqual((16+48)//16,4)

if __name__ == '__main__': unittest.main()
