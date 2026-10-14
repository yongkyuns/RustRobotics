import unittest
import numpy as np
from schedules import SCHEDULES,policy_at,prefix_reference
class SchedulesTest(unittest.TestCase):
    def test_all_ticks(self):
        t=np.arange(1500)
        expected={'OLD':np.zeros(1500,bool),'NEW':np.ones(1500,bool),'FIRST1':t==0,
                  'EARLY64':t<64,'LATE64':t>=64,'EARLY128':t<128,'LATE128':t>=128}
        for k,w in expected.items():
            np.testing.assert_array_equal([policy_at(k,int(x))==411 for x in t],w)
    def test_exact_boundary(self):
        for k in (64,128):
            self.assertEqual(policy_at('EARLY'+str(k),k-1),411)
            self.assertEqual(policy_at('EARLY'+str(k),k),410)
            self.assertEqual(policy_at('LATE'+str(k),k-1),410)
            self.assertEqual(policy_at('LATE'+str(k),k),411)
    def test_first_only(self):
        self.assertEqual(policy_at('FIRST1',0),411)
        self.assertEqual(policy_at('FIRST1',1),410)
        self.assertEqual(policy_at('FIRST1',1499),410)
    def test_unknown_schedule(self):
        with self.assertRaises(ValueError):policy_at('early64',0)
    def test_invalid_tick(self):
        for t in (-1,1500,1.0,True):
            with self.assertRaises(ValueError):policy_at('OLD',t)
    def test_prefix_contract(self):
        for k in SCHEDULES:
            arm,n=prefix_reference(k)
            self.assertTrue(all(policy_at(k,t)==410+arm for t in range(n)))
    def test_complementary_schedules(self):
        for k in (64,128):
            for t in range(1500):
                self.assertEqual(policy_at('EARLY'+str(k),t)+policy_at('LATE'+str(k),t),821)
    def test_single_switch(self):
        for k in SCHEDULES:
            seq=np.array([policy_at(k,t) for t in range(1500)])
            self.assertEqual(int(np.count_nonzero(np.diff(seq))),0 if k in ('OLD','NEW') else 1)
if __name__=='__main__':unittest.main()
