import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import diagnose as d

class DiagnosticControls(unittest.TestCase):
    def save_linear(self,root,k,bias):
        w1=np.zeros((64,4),dtype='f4');w1[:4]=np.eye(4);b1=np.ones(64,dtype='f4')*5
        w2=np.eye(64,dtype='f4');b2=np.zeros(64,dtype='f4')
        w3=np.zeros((1,64),dtype='f4');w3[0,:4]=np.asarray(k,dtype='f4')/20
        b3=np.array([bias/20-5*w3.sum()],dtype='f4')
        path=root/'linear.bin'
        np.concatenate([p for w,b in [(w1,b1),(w2,b2),(w3,b3)] for p in [w.T.copy().ravel(),b]]).astype('<f4').tofile(path)
        return path
    def test_known_stable_nonzero_equilibrium(self):
        with tempfile.TemporaryDirectory() as t:
            r=d.scalar_pieces(self.save_linear(Path(t),[4,8,-100,-40],-2))
            self.assertEqual(len(r['all_equilibria']),1)
            self.assertAlmostEqual(r['all_equilibria'][0]['position'],.5,places=4)
            self.assertEqual(r['stable_equilibria'],1)
            self.assertFalse(r['origin_is_equilibrium'])
            np.testing.assert_allclose(r['all_equilibria'][0]['force_jacobian'],[4,8,-100,-40],atol=1e-5)
    def test_unstable_cart_mode_not_hidden_by_stable_pole(self):
        with tempfile.TemporaryDirectory() as t:
            r=d.scalar_pieces(self.save_linear(Path(t),[-4,8,-100,-40],2))
            self.assertEqual(r['stable_equilibria'],0)
            self.assertLess(r['all_equilibria'][0]['pole_subsystem_radius'],1)
            self.assertGreater(r['all_equilibria'][0]['spectral_radius'],1)
    def test_nonzero_constant_force_has_no_rest_equilibrium(self):
        with tempfile.TemporaryDirectory() as t:
            r=d.scalar_pieces(self.save_linear(Path(t),[0,0,-100,-40],1))
            self.assertEqual(r['all_equilibria'],[])
    def test_reject_unhandled_continuum(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(AssertionError):
                d.scalar_pieces(self.save_linear(Path(t),[0,0,0,0],0))
    def test_bad_weight_length(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'bad.bin';np.ones(4546,dtype='<f4').tofile(path)
            with self.assertRaises(AssertionError):d.arrays(path)
    def test_discounted_terminal_decomposition(self):
        n,horizon=100,1500;gamma=d.GAMMA
        rewards=np.ones(n);rewards[-1]=-10
        actual=np.dot(gamma**np.arange(n),rewards)
        deficit=(1-gamma**horizon)/(1-gamma)-actual
        expected=11*gamma**(n-1)+(gamma**n-gamma**horizon)/(1-gamma)
        self.assertAlmostEqual(deficit,expected,places=10)
    def test_saved_layout_and_identity_exchange(self):
        with tempfile.TemporaryDirectory() as t:
            p=self.save_linear(Path(t),[4,8,-100,-40],-2)
            net=d.Network(p);s=torch.tensor([[.7,.1,.02,-.01]],dtype=torch.float32)
            zero=s.clone();zero[:,:2]=0
            self.assertAlmostEqual(net(s).item(),(4*.7+8*.1-100*.02-40*(-.01)-2)/20,places=5)
            torch.testing.assert_close(net(zero)+(net(s)-net(zero)),net(s))

if __name__=='__main__':unittest.main()
