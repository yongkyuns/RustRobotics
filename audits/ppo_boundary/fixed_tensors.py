#!/usr/bin/env python3
"""Fixed-derived-tensor PPO diagnostic. Not a simulator or learning benchmark.

The primary reference uses unmodified SB3 PPO.train/evaluate_actions and Torch
Adam. Only its buffer and fixed-scale squashed-density adapter are specialized.
All native input tensors and minibatch orders are checked byte-for-byte. Native
weights are loaded only initially in the primary mode. A separate conditional
mode restores incoming native weights, never Adam state, before each transaction.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import gymnasium
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
import torch

# Original limits, unchanged. The parameter threshold applies to the entire path.
LIMITS = {'loss': (5e-4, 1e-4), 'parameters': (2e-4, 0.0)}
CASE_NAMES = ['default-201', 'default-204', 'terminals-pool-201', 'timeouts-pool-201']


def compare(actual, expected, kind):
    a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == e.shape and np.isfinite(a).all() and np.isfinite(e).all()
    err = np.abs(a-e)
    atol, rtol = LIMITS[kind]
    return {'max_abs': float(err.max(initial=0)), 'pass': bool(np.all(err <= atol + rtol*np.abs(e))),
            'argmax': int(err.argmax()) if err.size else None}


def exact(actual, expected):
    a, e = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    assert a.shape == e.shape and a.tobytes() == e.tobytes(), 'native input changed'


def network_layers(model, critic=False):
    p = model.policy
    return ([p.mlp_extractor.value_net[0], p.mlp_extractor.value_net[2], p.value_net] if critic
            else [p.mlp_extractor.policy_net[0], p.mlp_extractor.policy_net[2], p.action_net])


def load_state(model, state):
    with torch.no_grad():
        for key, critic in [('actor', False), ('critic', True)]:
            raw = np.asarray(state[key], dtype=np.float32)
            offset = 0
            for layer in network_layers(model, critic):
                ni, no = layer.in_features, layer.out_features
                end = offset + ni*no
                layer.weight.copy_(torch.from_numpy(raw[offset:end].reshape(ni, no).T.copy()))
                layer.bias.copy_(torch.from_numpy(raw[end:end+no].copy()))
                offset = end+no
            assert offset == len(raw)


def export(model, critic=False, gradients=False):
    parts = []
    for layer in network_layers(model, critic):
        w, b = (layer.weight.grad, layer.bias.grad) if gradients else (layer.weight, layer.bias)
        assert w is not None and b is not None
        parts += [w.detach().cpu().numpy().T.copy().ravel(), b.detach().cpu().numpy().ravel()]
    return np.concatenate(parts)


class SquashedLatentDensity(DiagGaussianDistribution):
    """Density of force L*tanh(z), evaluated at stored z, not rounded force.

    Std is explicitly fixed from the recorded native configuration. This avoids
    exp(log(std)) rounding a fixed scale differently; no trainable parameter is
    discarded because log_std is frozen. The analytic Gaussian is still Torch's
    Normal. The action-only Jacobian cannot change the mean derivative.
    """
    def __init__(self, std, limit):
        super().__init__(1)
        self.fixed_std = float(np.float32(std))
        self.log_limit = float(np.log(np.float32(limit)))

    def proba_distribution(self, mean_actions, log_std):
        assert not log_std.requires_grad
        self.distribution = torch.distributions.Normal(mean_actions, torch.ones_like(mean_actions)*self.fixed_std)
        return self

    def log_prob(self, actions):
        magnitude = actions.abs()
        jac = ((-magnitude + float(np.float32(np.log(np.float32(2))))) - torch.log(torch.exp(-2*magnitude)+1))*2
        return super().log_prob(actions) - self.log_limit - jac.sum(dim=1)


def check_density():
    means = torch.tensor([[-.3], [.0], [.2], [2.0]], requires_grad=True)
    z = torch.tensor([[-12.0], [.02], [.7], [12.0]])
    density = SquashedLatentDensity(.1, 20)
    density.proba_distribution(means, torch.tensor([math.log(.1)]))
    lp = density.log_prob(z)
    derivative = torch.autograd.grad(lp.sum(), means)[0]
    expected = (z-means.detach())/(density.fixed_std*density.fixed_std)
    assert torch.allclose(derivative, expected, atol=2e-4, rtol=2e-6)
    # Independent f64 density including saturated-force latents.
    zz, mm = z.numpy().ravel().astype('f8'), means.detach().numpy().ravel().astype('f8')
    jac = math.log(20)+2*(math.log(2)-np.abs(zz)-np.log1p(np.exp(-2*np.abs(zz))))
    reference = -.5*((zz-mm)/density.fixed_std)**2 - math.log(density.fixed_std) - .5*math.log(2*math.pi) - jac
    assert np.allclose(lp.detach().numpy(), reference, atol=2e-3, rtol=2e-6)
    return {'analytic_mean_gradient': True, 'f64_density': True, 'finite_saturated_latents': True}


class FrozenBuffer(RolloutBuffer):
    def install(self, native, owner, mode, previous_state):
        batch = native['batch']
        self.observations = np.asarray(batch['observations'], dtype=np.float32)
        self.actions = np.asarray(batch['latent_actions'], dtype=np.float32).reshape(-1,1)
        self.values = np.asarray([r['value'] for r in native['rows']], dtype=np.float32).reshape(-1,1)
        self.log_probs = np.asarray(batch['old_log_probs'], dtype=np.float32).reshape(-1,1)
        self.returns = np.asarray(batch['returns'], dtype=np.float32).reshape(-1,1)
        self.advantages = np.asarray(batch['advantages'], dtype=np.float32).reshape(-1,1)
        self.generator_ready, self.full = True, True
        self.native, self.owner, self.mode = native, owner, mode
        self.previous_state = previous_state
        self.cursor = 0
        self.epoch = 0
        self.field_hashes = {key: hashlib.sha256(getattr(self,key).tobytes()).hexdigest()
                             for key in ['observations','actions','values','log_probs','returns','advantages']}

    def get(self, batch_size=None):
        size = self.buffer_size*self.n_envs
        count = math.ceil(size/batch_size)
        chunks = self.native['minibatches'][self.epoch*count:(self.epoch+1)*count]
        assert len(chunks) == count
        all_indices = np.concatenate([np.asarray(m['indices'], dtype=np.int64) for m in chunks])
        assert np.array_equal(np.sort(all_indices), np.arange(size)), 'incorrect epoch traversal'
        self.epoch += 1
        for step in chunks:
            if self.mode == 'conditional':
                load_state(self.owner, self.previous_state)
            indices = np.asarray(step['indices'], dtype=np.int64)
            sample = self._get_samples(indices)
            for field, array in [('observations',self.observations), ('actions',self.actions),
                                 ('old_values',self.values), ('old_log_prob',self.log_probs),
                                 ('advantages',self.advantages), ('returns',self.returns)]:
                tensor = getattr(sample, field).detach().cpu().numpy()
                expected = array[indices]
                if field not in ['observations','actions']:
                    expected = expected.ravel()
                exact(tensor,expected)
            self.current_indices = indices
            self.current_native = step
            self.current_sample = sample
            yield sample
            self.previous_state = step['state']
            self.cursor += 1


def run_case(case, adam, mode, prior, output):
    cfg=case['config']
    env=prior.ReplayEnvironment(case)
    model=PPO('MlpPolicy',env,seed=case['seed'],device='cpu',verbose=0,
              learning_rate=cfg['learning_rate'], n_steps=cfg['rollout_steps'], batch_size=cfg['mini_batch_size'],
              n_epochs=cfg['epochs'], gamma=float(np.float32(cfg['gamma'])), gae_lambda=float(np.float32(cfg['gae_lambda'])),
              clip_range=float(np.float32(cfg['clip_epsilon'])), vf_coef=cfg['value_loss_coef'], ent_coef=0,
              normalize_advantage=False,max_grad_norm=float('inf'),rollout_buffer_class=FrozenBuffer,
              policy_kwargs=dict(net_arch=dict(pi=[cfg['hidden_dim']]*2,vf=[cfg['hidden_dim']]*2),
                                 activation_fn=torch.nn.ReLU,ortho_init=False,
                                 log_std_init=math.log(float(np.float32(cfg['action_std'])/np.float32(cfg['max_force']))),
                                 optimizer_kwargs=dict(eps=1e-8 if mode=='wrong-epsilon' else adam['epsilon'],
                                                       betas=(adam['beta_1'],adam['beta_2']))))
    model.policy.log_std.requires_grad_(False)
    model.policy.action_dist=SquashedLatentDensity(np.float32(cfg['action_std'])/np.float32(cfg['max_force']),cfg['max_force'])
    load_state(model,case['initial'])
    model.set_logger(configure(folder=None,format_strings=[]))
    _,cb=model._setup_learn(len(case['updates'])*cfg['rollout_steps']*env.num_envs,progress_bar=False)
    assert model.train.__func__ is PPO.train
    assert model.policy.evaluate_actions.__func__ is ActorCriticPolicy.evaluate_actions
    assert type(model.policy.optimizer) is torch.optim.Adam
    assert model.clip_range_vf is None and model.target_kl is None
    params=list(model.policy.parameters())
    identities=tuple(id(p) for p in params)
    previous=case['initial']
    steps=[]; tensor_hashes=[]; states=[]; gradient_states=[]
    total=0
    for ui,native in enumerate(case['updates']):
        buffer=model.rollout_buffer
        buffer.install(native,model,mode,previous)
        tensor_hashes.append(buffer.field_hashes)
        local_count=0
        pending={}
        def before_step(optimizer,args,kwargs):
            sample=buffer.current_sample
            expected=buffer.current_native
            with torch.no_grad():
                values,lp,_=model.policy.evaluate_actions(sample.observations,sample.actions)
                ratio=torch.exp(lp-sample.old_log_prob)
                eps=float(np.float32(cfg['clip_epsilon']))
                policy_loss=-torch.minimum(sample.advantages*ratio,sample.advantages*torch.clamp(ratio,1-eps,1+eps)).mean()
                value_loss=(values.flatten()-sample.returns).square().mean()
            pending.clear()
            pending.update(policy_loss=compare(policy_loss.item(),expected['policy_loss'],'loss'),
                           value_loss=compare(value_loss.item(),expected['value_loss'],'loss'))
            gradient_states.append(np.stack([export(model,False,True),export(model,True,True)]))
        def after_step(optimizer,args,kwargs):
            nonlocal local_count,total
            expected=buffer.current_native['state']
            actual_a,actual_c=export(model),export(model,True)
            record={'update':ui+1,'minibatch':local_count+1,**pending,
                    'actor':compare(actual_a,expected['actor'],'parameters'),
                    'critic':compare(actual_c,expected['critic'],'parameters')}
            states.append(np.stack([actual_a,actual_c]))
            total+=1;local_count+=1
            optimizer_steps=[int(optimizer.state[p]['step'].item()) for p in params if p.requires_grad]
            required_step=local_count if mode=='reset-adam' else total
            assert set(optimizer_steps)=={required_step}, 'optimizer step history changed'
            assert tuple(id(p) for p in model.policy.parameters())==identities
            record['optimizer_step']=required_step
            steps.append(record)
        pre=model.policy.optimizer.register_step_pre_hook(before_step)
        post=model.policy.optimizer.register_step_post_hook(after_step)
        try:
            model.train()
        finally:
            pre.remove();post.remove()
        assert local_count==len(native['minibatches']) and buffer.cursor==local_count and buffer.epoch==cfg['epochs']
        assert buffer.field_hashes=={key:hashlib.sha256(getattr(buffer,key).tobytes()).hexdigest() for key in buffer.field_hashes}
        previous=native['minibatches'][-1]['state']
        if mode=='reset-adam':
            model.policy.optimizer.state.clear()
    env.close()
    keys=['actor','critic','policy_loss','value_loss']
    failures=[{'update':s['update'],'minibatch':s['minibatch'],'field':key,'error':s[key]} for s in steps for key in keys if not s[key]['pass']]
    result={'case':case['name'],'mode':mode,'pass':not failures,'first_failure':failures[0] if failures else None,
            'maximum_errors':{key:max(s[key]['max_abs'] for s in steps) for key in keys},
            'transactions':total,'source_transitions':case['updates'][-1]['total_env_steps'],'new_simulator_transitions':0,
            'input_field_hashes':tensor_hashes,'steps':steps}
    stem=f"{case['name']}-{mode}"
    (output/(stem+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/(stem+'.npz'),states=np.asarray(states),gradients=np.asarray(gradient_states))
    print(stem,result['pass'],result['maximum_errors'],flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert stable_baselines3.__version__=='2.9.0' and torch.__version__=='2.8.0+cpu'
    assert np.__version__=='2.2.6' and gymnasium.__version__=='1.3.0'
    sys.path.insert(0,str(args.native/'sources/audits/ppo_boundary'))
    import compare as prior
    adam=json.loads((args.native/'adam-config.json').read_text())
    assert adam['epsilon']==1e-5 and adam['weight_decay'] is None and adam['grad_clipping'] is None
    controls=check_density()
    reports=[]
    for name in CASE_NAMES:
        case=json.loads((args.native/(name+'.json')).read_text())
        assert case['name']==name
        for mode in ['free','conditional']:
            reports.append(run_case(case,adam,mode,prior,args.output))
    first=json.loads((args.native/'default-201.json').read_text())
    for mode in ['wrong-epsilon','reset-adam']:
        reports.append(run_case(first,adam,mode,prior,args.output))
    summary={'schema':1,'limits':LIMITS,'adam':adam,'adapter_controls':controls,
             'versions':{'sb3':stable_baselines3.__version__,'torch':torch.__version__,'numpy':np.__version__,
                         'gymnasium':gymnasium.__version__,'python':sys.version},
             'source_hashes':{'PPO.train':hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest(),
                              'evaluate_actions':hashlib.sha256(inspect.getsource(ActorCriticPolicy.evaluate_actions).encode()).hexdigest()},
             'results':[{k:r[k] for k in ['case','mode','pass','first_failure','maximum_errors','transactions']} for r in reports],
             'negative_controls_detected':all(not r['pass'] for r in reports if r['mode'] in ['wrong-epsilon','reset-adam']),
             'scope':'Fixed native tensors; no simulator interactions, online policy comparison, or learning-quality qualification.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for module in [sys.modules[PPO.__module__],sys.modules[ActorCriticPolicy.__module__]]:
        (args.output/Path(module.__file__).name).write_text(inspect.getsource(module))
    print(json.dumps(summary,indent=2),flush=True)
    if not all(r['pass'] for r in reports if r['mode']=='free') or not summary['negative_controls_detected']:
        sys.exit(1)


if __name__=='__main__':
    main()
