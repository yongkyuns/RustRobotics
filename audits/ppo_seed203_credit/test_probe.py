import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import probe

class Controls(unittest.TestCase):
    def test_selection_uses_first_eligible_state_not_rewards_or_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);n=60;keys=np.repeat(np.arange(n,dtype='u8')+10,100)
            ticks=np.tile(np.arange(100),n);obs=np.zeros((n*100,4),dtype='f4')
            obs[:,0]=.55;obs[:,1]=.1
            state=obs.copy()
            data=dict(keys=keys,ticks=ticks,horizon=np.full(len(keys),1500),observation=obs,
                after_state=state,terminated=np.zeros(len(keys),bool),truncated=np.zeros(len(keys),bool),reward=np.zeros(len(keys)))
            np.savez(root/'final-trajectories.npz',**data)
            a=probe.select_panel(root)
            self.assertEqual(a['eligible_episodes'],60)
            self.assertEqual(len(a['states']),16)
            self.assertTrue(all(r['tick']==25 for r in a['states']))
            data['reward']=np.arange(len(keys))*-100.0
            np.savez(root/'final-trajectories.npz',**data)
            self.assertEqual(a,probe.select_panel(root))
    def test_rejects_insufficient_distinct_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);o=np.array([[.55,.1,0,0]],dtype='f4')
            np.savez(root/'final-trajectories.npz',keys=np.array([3]),ticks=np.array([25]),horizon=np.array([1500]),observation=o)
            with self.assertRaises(AssertionError):probe.select_panel(root)
    def test_network_layout_matches_independent_f64(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'network.bin';rng=np.random.default_rng(56)
            raw=rng.normal(0,.02,4545).astype('<f4');raw.tofile(p)
            obs=rng.normal(0,.1,(17,4)).astype('f4')
            self.assertLess(np.max(abs(probe.predict(probe.load_network(p),obs)-probe.independent_inference(raw,obs))),1e-6)
    def test_network_rejects_invalid_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.bin';np.zeros(3,dtype='f4').tofile(p)
            with self.assertRaises(AssertionError):probe.load_network(p)
    def test_lambda_one_telescopes_with_live_bootstrap(self):
        gamma=.75;r=np.array([1.,2.,3.]);v=np.array([4.,7.,2.,9.])
        td=r+gamma*v[1:]-v[:-1]
        self.assertAlmostEqual(np.dot(gamma**np.arange(3),td)+v[0]-gamma**3*v[-1],np.dot(gamma**np.arange(3),r))
    def test_lambda_one_terminal_discards_future_value(self):
        gamma=.75;r=np.array([1.,2.,-10.]);v=np.array([4.,7.,2.,999.])
        td=r+gamma*v[1:]*np.array([1,1,0])-v[:-1]
        self.assertAlmostEqual(np.dot(gamma**np.arange(3),td)+v[0],np.dot(gamma**np.arange(3),r))
    def test_initial_value_cancels_from_paired_action_credit(self):
        gamma=.99;lam=.95
        r=np.array([[1.,2.,-10.],[1.2,2.1,-10.]])
        v=np.array([[3.,4.,5.,0.],[3.,5.,6.,0.]])
        q=(gamma*lam)**np.arange(3)
        credit=(r+gamma*v[:,1:]-v[:,:-1])@q
        altered=v.copy();altered[:,0]+=80
        credit2=(r+gamma*altered[:,1:]-altered[:,:-1])@q
        self.assertAlmostEqual(credit[1]-credit[0],credit2[1]-credit2[0])
    def test_critic_terms_can_reverse_reward_ranking(self):
        # A comparator sentinel, not a measured physical trajectory.
        gamma=.99;lam=.95
        r=np.array([[1.,0.],[.9,0.]])
        v=np.array([[0.,0.,0.],[0.,10.,0.]])
        direct=(r@np.array([1.,gamma]))
        credit=(r+gamma*v[:,1:]-v[:,:-1])@np.array([1.,gamma*lam])
        self.assertLess(direct[1]-direct[0],0)
        self.assertGreater(credit[1]-credit[0],0)

if __name__=='__main__':unittest.main()
