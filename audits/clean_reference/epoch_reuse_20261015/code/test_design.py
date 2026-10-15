import unittest,copy,math
from types import SimpleNamespace
from fractions import Fraction
from design import *
from decision import assess

class Design(unittest.TestCase):
    def test_orders_complete(self):
        self.assertEqual(set(ORDER),set(SEEDS))
        for seq in ORDER.values():self.assertEqual(set(seq),set(ARMS));self.assertEqual(len(seq),3)
    def test_transaction_counts(self):
        self.assertEqual(transactions(TOTAL,'constant10'),204800)
        self.assertEqual(transactions(TOTAL,'decay10'),204800)
        self.assertEqual(transactions(TOTAL,'decay4'),81920)
        self.assertEqual(transactions(TOTAL,'decay4')-transactions(1048576,'decay4'),16384)
    def test_every_rate_independent_rational(self):
        for a in ARMS:
            for u in range(1,641):
                p=min(Fraction(u*2048,1048576),Fraction(1))
                expected=Fraction(3,10000)*(1-Fraction(9,10)*p) if a!='constant10' else Fraction(3,10000)
                self.assertLess(abs(rate(u*2048,a)-float(expected)),1e-18)
    def test_nonzero_floor(self):
        for a in ARMS:
            self.assertGreater(rate(TOTAL,a),0)
        self.assertEqual(rate(TOTAL,'decay4'),.00003)
    def test_shared_decay_schedule(self):
        for t in CHECKPOINTS:self.assertEqual(rate(t,'decay10'),rate(t,'decay4'))
    def test_progress_ignored(self):
        m=SimpleNamespace(num_timesteps=786432);s=AbsoluteSchedule(m,'decay4')
        self.assertEqual(s(.1),s(.9));m.num_timesteps=1310720;self.assertEqual(s(1.),.00003)
    def test_invalid(self):
        for t in [-1,True,1.,33]:
            with self.assertRaises(ValueError):transactions(t,'decay4')
        for a in ['decay','constant','four']:
            with self.assertRaises(ValueError):rate(0,a)
        for x in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):AbsoluteSchedule(SimpleNamespace(num_timesteps=1),'decay4')(x)
    def data(self):
        comp=[dict(seed=s,mode=m,checkpoint=cp,difficult_completion_delta=4,difficult_margin_delta=4,ordinary_return_delta=1.) for s in SEEDS for m in ('deterministic','stochastic') for cp in (1048576,1310720)]
        tail=[dict(seed=s,mode=m,completion_delta=0,margin_delta=0) for s in SEEDS for m in ('deterministic','stochastic')]
        return comp,tail
    def test_original_rule_pass(self):self.assertTrue(assess(*self.data())['development_screen_passed'])
    def test_difficult_loss_fails(self):
        c,t=self.data();c[0]['difficult_completion_delta']=-1;self.assertFalse(assess(c,t)['development_screen_passed'])
    def test_ordinary_loss_fails(self):
        c,t=self.data();c[0]['ordinary_return_delta']=-.00001;self.assertFalse(assess(c,t)['development_screen_passed'])
    def test_tail_failure(self):
        for k in ('completion_delta','margin_delta'):
            c,t=self.data();t[0][k]=-1;self.assertFalse(assess(c,t)['development_screen_passed'])
    def test_five_percent_not_rounded(self):
        c,t=self.data()
        for x in c:x['difficult_margin_delta']=3
        self.assertFalse(assess(c,t)['development_screen_passed'])
    def test_missing_or_duplicate(self):
        c,t=self.data()
        with self.assertRaises(ValueError):assess(c[:-1],t)
        c[-1]=c[0]
        with self.assertRaises(ValueError):assess(c,t)
    def test_fractional_counts(self):
        c,t=self.data();c[0]['difficult_margin_delta']=4.0
        with self.assertRaises(ValueError):assess(c,t)
    def test_nan(self):
        c,t=self.data();c[0]['ordinary_return_delta']=math.nan
        with self.assertRaises(ValueError):assess(c,t)
if __name__=='__main__':unittest.main()
