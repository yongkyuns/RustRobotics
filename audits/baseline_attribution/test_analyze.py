"""Synthetic fail-closed/gradient controls plus optional actual-witness autograd."""
import hashlib
import io
import json
import os
from pathlib import Path
import unittest
import zipfile
import numpy as np
import analyze as a


class ScoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21206)
        self.w = rng.normal(0, .1, 4545)
        self.x = rng.normal(0, .2, (19, 4))
        self.lat = a.forward(self.w, self.x) + rng.normal(0, .1, 19)
        self.adv = rng.normal(size=19)
        self.direction = rng.normal(size=4545)
        self.direction /= np.linalg.norm(self.direction)

    def test_directional_finite_difference(self):
        eps = 1e-6
        g = a.score_gradient(self.w, self.x, self.lat, self.adv)
        plus = np.mean(self.adv * a.log_prob(self.w+eps*self.direction, self.x, self.lat))
        minus = np.mean(self.adv * a.log_prob(self.w-eps*self.direction, self.x, self.lat))
        np.testing.assert_allclose((plus-minus)/(2*eps), g@self.direction, atol=1e-8, rtol=1e-6)

    def test_gradient_is_ascent(self):
        g = a.score_gradient(self.w, self.x, self.lat, self.adv)
        objective = lambda w: np.mean(self.adv*a.log_prob(w, self.x, self.lat))
        self.assertGreater(objective(self.w+1e-6*g), objective(self.w))

    def test_linear_components(self):
        g = lambda v: a.score_gradient(self.w, self.x, self.lat, v)
        np.testing.assert_allclose(g(self.adv+2), g(self.adv)+g(np.full(19, 2)), atol=1e-12)

    def test_zero_advantage(self):
        self.assertEqual(np.linalg.norm(a.score_gradient(self.w, self.x, self.lat, np.zeros(19))), 0)

    def test_mean_action_zero_score(self):
        g = a.score_gradient(self.w, self.x, a.forward(self.w, self.x), self.adv)
        self.assertEqual(np.linalg.norm(g), 0)

    def test_observation_order_invariance(self):
        ids = np.arange(19)[::-1]
        np.testing.assert_allclose(a.score_gradient(self.w, self.x, self.lat, self.adv),
                                   a.score_gradient(self.w, self.x[ids], self.lat[ids], self.adv[ids]), atol=1e-12)

    def test_state_baseline_not_zero_for_finite_sample(self):
        g = a.score_gradient(self.w, self.x, self.lat, self.x[:,0]**2)
        self.assertGreater(np.linalg.norm(g), 1e-4)

    def test_group_decomposition(self):
        groups = np.array([[i//5] for i in range(19)])
        m = a.group_means(self.adv, groups)
        g = lambda v: a.score_gradient(self.w, self.x, self.lat, v)
        np.testing.assert_allclose(g(self.adv), g(m)+g(self.adv-m), atol=1e-12)

    def test_one_group_mean_after_centering_is_zero(self):
        result = a.group_means(a.normalize(self.adv), np.zeros((19,1)))
        np.testing.assert_allclose(result, 0, atol=1e-15)

    def test_normalization_shift_invariance(self):
        np.testing.assert_allclose(a.normalize(self.adv), a.normalize(self.adv+300), atol=1e-12)

    def test_normalization_constant(self):
        np.testing.assert_array_equal(a.normalize(np.ones(19)), np.zeros(19))

    def test_nonfinite_rejected(self):
        bad=self.adv.copy(); bad[0]=np.nan
        with self.assertRaises(ValueError): a.normalize(bad)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError): a.score_gradient(self.w,self.x,self.lat[:-1],self.adv)

    def test_zero_cosine_rejected(self):
        with self.assertRaises(ValueError): a.cosine(np.zeros(4),np.ones(4))

    def test_bad_network_length_rejected(self):
        with self.assertRaises(ValueError): a.network(b'\0'*8)


class ArchiveTests(unittest.TestCase):
    @staticmethod
    def archive(extra=False, broken=False):
        data=b'example payload'
        manifest={'payload.txt':hashlib.sha256(data).hexdigest()}
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as z:
            z.writestr('payload.txt',b'broken' if broken else data)
            z.writestr('manifest.json',json.dumps(manifest))
            if extra:z.writestr('unlisted.txt',b'extra')
        result=output.getvalue()
        return result,hashlib.sha256(result).hexdigest()

    def test_valid_manifest(self):
        z,n=a.verified_zip(*self.archive())
        self.assertEqual(n,1);z.close()

    def test_outer_hash(self):
        data,_=self.archive()
        with self.assertRaises(ValueError):a.verified_zip(data,'0'*64)

    def test_payload_hash(self):
        with self.assertRaises(ValueError):a.verified_zip(*self.archive(broken=True))

    def test_unmanifested_member(self):
        with self.assertRaises(ValueError):a.verified_zip(*self.archive(extra=True))

    def test_numeric_summary_mismatch(self):
        with self.assertRaises(ValueError):a.numeric_diff({'a':1},{'b':1})


class PairTests(unittest.TestCase):
    @staticmethod
    def records():
        return [dict(row=str(i),draw=str(j),key=f'{i}:{j}',recorded_latent='0',q_estimate='2',
                     v_estimate='1',difference='1',q_discounted='2',q_coefficient='0',q_bootstrap='0',
                     v_discounted='1',v_coefficient='0',v_bootstrap='0',q_steps='1',v_steps='1',
                     q_terminal='true',v_terminal='true') for i in range(1024) for j in range(8)]

    def test_pair_order_independence(self):
        q,v=a.pair_values(self.records()[::-1],np.zeros(1024))
        np.testing.assert_array_equal(q,np.full((1024,8),2))
        np.testing.assert_array_equal(v,np.ones((1024,8)))

    def test_missing_pair(self):
        with self.assertRaises(ValueError):a.pair_values(self.records()[:-1],np.zeros(1024))

    def test_duplicate_pair(self):
        r=self.records();r[-1]=r[0]
        with self.assertRaises(ValueError):a.pair_values(r,np.zeros(1024))

    def test_terminal_bootstrap_leak(self):
        r=self.records();r[0]['q_coefficient']='1'
        with self.assertRaises(ValueError):a.pair_values(r,np.zeros(1024))

    def test_action_mismatch(self):
        r=self.records();r[0]['recorded_latent']='.01'
        with self.assertRaises(ValueError):a.pair_values(r,np.zeros(1024))


@unittest.skipUnless(os.environ.get('BASELINE_RESULT_ZIP'), 'actual-witness ZIP not supplied')
class WitnessAutogradTest(unittest.TestCase):
    def test_all_rows_against_torch_autograd(self):
        import torch
        torch.set_num_threads(1)
        result,_=a.verified_zip(Path(os.environ['BASELINE_RESULT_ZIP']).read_bytes(),a.RESULT_SHA)
        p,_=a.verified_zip(result.read('paired-credit-41006.zip'),a.PAIRED_SHA)
        flat=a.network(result.read('actor-4139.bin'))
        union=json.loads(p.read('original/union.json'))
        x=np.array(union['observations']); lat=np.array(union['latents'])
        q,v=a.pair_values(a.read_csv(p.read('paired-credit.csv')),lat)
        model=torch.nn.Sequential(torch.nn.Linear(4,64),torch.nn.ReLU(),
                                  torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1)).double()
        off=0
        with torch.no_grad():
            for layer in [model[0],model[2],model[4]]:
                ni,no=layer.in_features,layer.out_features
                layer.weight.copy_(torch.tensor(flat[off:off+ni*no].reshape(ni,no).T));off+=ni*no
                layer.bias.copy_(torch.tensor(flat[off:off+no]));off+=no
        self.assertEqual(off,4545)
        cases={'original':np.array(union['raw']),
               'baseline_only':np.array(union['returns'])-v.mean(1),
               'paired':(q-v).mean(1)}
        max_error=0.
        for name,adv in cases.items():
            adv=a.normalize(adv);model.zero_grad()
            mu=model(torch.tensor(x)).flatten()
            objective=(-.5*((torch.tensor(lat)-mu)/a.LATENT_SIGMA)**2*torch.tensor(adv)).mean()
            objective.backward()
            chunks=[]
            for layer in [model[0],model[2],model[4]]:
                chunks.extend([layer.weight.grad.T.detach().numpy().ravel(),layer.bias.grad.detach().numpy()])
            actual=np.concatenate(chunks)
            expected=a.score_gradient(flat,x,lat,adv)
            error=np.max(abs(actual-expected));max_error=max(max_error,error)
            np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=1e-9,err_msg=name)
        print(f'Actual 1024-row Torch autograd max gradient error: {max_error:.3g}')


if __name__=='__main__':unittest.main(verbosity=2)
