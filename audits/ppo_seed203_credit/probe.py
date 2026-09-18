"""Frozen-state first-action comparisons. No trainer or optimization is invoked."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ARMS = ('baseline', 'extra-frozen', 'extra-refresh')
GAMMA = float(np.float32(.99))
LAMBDA = float(np.float32(.95))
HORIZON = 1500
REPEATS = 256
STATES = 16
SHIFT = .02

class Step(C.Structure):
    _fields_ = [('observation', C.c_float*4), ('state', C.c_float*4),
                ('reward', C.c_float), ('force', C.c_float), ('terminated', C.c_uint32),
                ('truncated', C.c_uint32), ('steps', C.c_uint32), ('status', C.c_uint32)]

def library(path):
    lib = C.CDLL(str(path))
    for name, args, ret in [
        ('rr_probe_create', [C.c_float]*4+[C.c_uint64,C.c_uint32], C.c_uint64),
        ('rr_probe_step', [C.c_uint64,C.c_float], Step),
        ('rr_probe_free', [C.c_uint64], C.c_uint32),
        ('rr_probe_interactions', [], C.c_uint64),
    ]:
        fn = getattr(lib,name); fn.argtypes = args; fn.restype = ret
    assert C.sizeof(Step) == 56
    return lib

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def load_network(path):
    raw = np.fromfile(path, dtype='<f4')
    assert raw.shape == (4545,) and np.isfinite(raw).all()
    net = torch.nn.Sequential(torch.nn.Linear(4,64), torch.nn.ReLU(),
                              torch.nn.Linear(64,64), torch.nn.ReLU(), torch.nn.Linear(64,1))
    offset = 0
    for layer, ni, no in zip((net[0],net[2],net[4]),(4,64,64),(64,64,1)):
        count = ni*no
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(raw[offset:offset+count].reshape(ni,no).T.copy()))
            layer.bias.copy_(torch.from_numpy(raw[offset+count:offset+count+no].copy()))
        offset += count+no
    assert offset == len(raw)
    for parameter in net.parameters(): parameter.requires_grad_(False)
    return net.eval()

def predict(net, observation):
    with torch.no_grad():
        result = net(torch.from_numpy(np.ascontiguousarray(observation,dtype=np.float32))).numpy().reshape(-1)
    assert np.isfinite(result).all()
    return result

def independent_inference(raw, observation):
    x = np.asarray(observation, dtype=np.float64); offset = 0
    for index,(ni,no) in enumerate(zip((4,64,64),(64,64,1))):
        count = ni*no
        w = raw[offset:offset+count].astype('f8').reshape(ni,no)
        b = raw[offset+count:offset+count+no].astype('f8')
        x = x@w+b
        if index<2: x = np.maximum(x,0)
        offset += count+no
    return x.reshape(-1)

def select_panel(directory):
    with np.load(directory/'final-trajectories.npz',allow_pickle=False) as source:
        d = {k:source[k] for k in source.files}
    o = d['observation']
    eligible = ((d['horizon']==1500)&(d['ticks']>=25)&(d['ticks']<=475)&(d['ticks']%25==0)
                &(abs(o[:,0])>=.5)&(o[:,0]*o[:,1]>0))
    keys = np.unique(d['keys'][eligible])
    assert len(keys)>=STATES
    selected_keys = keys[np.linspace(0,len(keys)-1,STATES,dtype=int)]
    records = []
    for key in selected_keys:
        candidates = np.flatnonzero(eligible&(d['keys']==key))
        index = int(candidates[np.argmin(d['ticks'][candidates])])
        tick = int(d['ticks'][index])
        previous = np.flatnonzero((d['keys']==key)&(d['horizon']==1500)&(d['ticks']==tick-1))
        assert len(previous)==1 and not d['terminated'][previous[0]] and not d['truncated'][previous[0]]
        state = d['after_state'][previous[0]].copy()
        assert abs(state[0])<=np.float32(2.4) and abs(state[2])<=np.float32(.6)
        assert np.all(np.abs(o[index]-state)<=np.array([.002,.01,.002,.01])+1e-6)
        records.append(dict(key=int(key),tick=tick,index=index,physical_state=state.tolist(),observation=o[index].tolist()))
    return dict(eligible_episodes=len(keys),selection='first eligible tick per episode; 16 evenly spaced sorted keys',states=records)

def select_all(root, output):
    output.mkdir(parents=True,exist_ok=True)
    result = {}
    for arm in ARMS:
        folder = root/f'critic-{arm}-203'
        result[arm] = select_panel(folder)
        result[arm]['source_hashes'] = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for name in ['final-trajectories.npz','actor-1048576.bin','critic-1048576.bin','critic-preextra-1048576.bin']}
    write(output/'panels.json',result)
    return result

def check_inference(folder, panel, actor, critics):
    obs = np.array([r['observation'] for r in panel['states']], dtype='f4')
    errors = {}
    for name,net in [('actor-1048576.bin',actor),('critic-preextra-1048576.bin',critics[0]),('critic-1048576.bin',critics[1])]:
        raw = np.fromfile(folder/name,dtype='<f4')
        error = float(np.max(abs(predict(net,obs)-independent_inference(raw,obs))))
        assert error<1e-4
        errors[name] = error
    with np.load(folder/'final-trajectories.npz',allow_pickle=False) as d:
        indices = [r['index'] for r in panel['states']]
        for field,net in [('critic_preextra',critics[0]),('critic',critics[1])]:
            error = float(np.max(abs(predict(net,obs)-d[field][indices])))
            assert error<1e-4
            errors[field+'_archive'] = error
    return errors

def run_case(lib, folder, panel, arm, output, repeats=REPEATS, horizon=HORIZON, shift=SHIFT, control=False):
    output.mkdir(parents=True,exist_ok=True)
    actor = load_network(folder/'actor-1048576.bin')
    critics = [load_network(folder/'critic-preextra-1048576.bin'),load_network(folder/'critic-1048576.bin')]
    inference = check_inference(folder,panel,actor,critics)
    before_rng = torch.random.get_rng_state().clone()
    snapshots = [tuple(p.detach().numpy().tobytes() for p in net.parameters()) for net in [actor,*critics]]
    ns = len(panel['states']); nbase=ns*repeats; n=2*nbase
    first_obs = np.repeat(np.array([r['observation'] for r in panel['states']],dtype='f4'),repeats*2,axis=0)
    initial_state = np.repeat(np.array([r['physical_state'] for r in panel['states']],dtype='f4'),repeats*2,axis=0)
    domain = 0x203C000000000000 + (ARMS.index(arm)+1)*0x1000000 + (0x100000 if control else 0)
    environment_seeds = domain + np.arange(nbase,dtype=np.uint64)
    action_seed = domain ^ 0x71F493E1
    rng = np.random.default_rng(action_seed)
    handles = []
    before_count = int(lib.rr_probe_interactions())
    returns = np.zeros(n); weighted_rewards = np.zeros(n); gae = np.zeros((n,2)); gae1 = np.zeros((n,2))
    td = np.zeros((n,2)); lengths = np.zeros(n,dtype=np.int32); endings = np.full(n,-1,dtype=np.int8)
    current_obs = first_obs.copy(); current_state=initial_state.copy()
    values = np.column_stack([predict(net,current_obs) for net in critics]).astype('f8')
    initial_value = values.copy(); tail=np.zeros((n,2)); active = np.ones(n,dtype=bool)
    first_actions = np.zeros(n,dtype='f4'); first_forces = np.zeros(n,dtype='f4')
    trace=[]
    # Match the frozen Gaussian scale used by the previous Torch policy holder.
    sigma = np.float32(torch.exp(torch.tensor(np.log(.1),dtype=torch.float32)).item())
    signs = np.tile(np.array([-1.,1.],dtype='f4'),nbase)
    try:
        for i,state in enumerate(initial_state):
            handle = lib.rr_probe_create(*map(float,state),int(environment_seeds[i//2]),horizon)
            assert handle;handles.append(handle)
        for tick in range(horizon):
            indices = np.flatnonzero(active)
            innovations = np.repeat(rng.standard_normal(nbase).astype('f4'),2)
            mean = predict(actor,current_obs[indices])
            actions = mean + sigma*innovations[indices]
            if tick==0:
                actions = actions + np.float32(shift)*signs[indices]
                first_actions[indices] = actions
            assert np.isfinite(actions).all() and abs(actions).max(initial=0)<1e5
            rows=[lib.rr_probe_step(handles[i],float(a)) for i,a in zip(indices,actions)]
            assert all(r.status==0 and r.steps==tick+1 for r in rows)
            reward=np.array([r.reward for r in rows],dtype='f8')
            observation=np.array([r.observation[:] for r in rows],dtype='f4')
            state=np.array([r.state[:] for r in rows],dtype='f4')
            force=np.array([r.force for r in rows],dtype='f4')
            terminal=np.array([r.terminated for r in rows],dtype=bool)
            truncated=np.array([r.truncated for r in rows],dtype=bool)
            assert np.isfinite(observation).all() and np.isfinite(state).all() and np.isfinite(reward).all()
            assert (abs(force)<=20).all()
            angle=abs(state[:,2])>np.float32(.6);position=abs(state[:,0])>np.float32(2.4)
            assert np.array_equal(terminal,angle|position)
            assert np.array_equal(truncated,(~terminal)&(tick+1==horizon))
            next_value=np.column_stack([predict(net,observation) for net in critics]).astype('f8')
            delta=reward[:,None]+GAMMA*next_value*(~terminal)[:,None]-values[indices]
            if tick==0:td[indices]=delta;first_forces[indices]=force
            returns[indices]+=GAMMA**tick*reward
            weighted_rewards[indices]+=(GAMMA*LAMBDA)**tick*reward
            gae[indices]+=(GAMMA*LAMBDA)**tick*delta
            gae1[indices]+=GAMMA**tick*delta
            for j,i in enumerate(indices):
                if (i//2)%repeats==0:
                    trace.append((i//(2*repeats),i%2,tick,*current_obs[i],*observation[j],*current_state[i],*state[j],
                        actions[j],force[j],reward[j],*values[i],*next_value[j],int(terminal[j]),int(truncated[j])))
            lengths[indices]=tick+1
            finished=terminal|truncated
            endcode=np.where(angle&position,3,np.where(angle,1,np.where(position,2,0)))
            endings[indices[finished]]=endcode[finished]
            tail[indices[finished]]=GAMMA**(tick+1)*next_value[finished]*(~terminal[finished])[:,None]
            active[indices[finished]]=False
            current_obs[indices]=observation;current_state[indices]=state;values[indices]=next_value
            if tick in [0,249,499,999,1499]:
                print('CREDIT PROGRESS',arm,'control' if control else 'main',tick+1,'active',int(active.sum()),flush=True)
            if not active.any():break
        assert not active.any() and (endings>=0).all()
        telescoping=float(np.max(abs(gae1+initial_value-tail-returns[:,None])))
        assert telescoping<2e-8
        arrays=dict(returns=returns.reshape(ns,repeats,2),gae=gae.reshape(ns,repeats,2,2),
            td=td.reshape(ns,repeats,2,2),lambda_rewards=weighted_rewards.reshape(ns,repeats,2),
            lambda1=gae1.reshape(ns,repeats,2,2),initial_values=initial_value.reshape(ns,repeats,2,2),
            tails=tail.reshape(ns,repeats,2,2),lengths=lengths.reshape(ns,repeats,2),endings=endings.reshape(ns,repeats,2),
            first_actions=first_actions.reshape(ns,repeats,2),first_forces=first_forces.reshape(ns,repeats,2),
            environment_seeds=environment_seeds.reshape(ns,repeats))
        if control:
            for name in ['returns','lambda_rewards','lengths','endings','first_actions','first_forces']:
                assert np.array_equal(arrays[name][...,0],arrays[name][...,1]),name
            for name in ['gae','td','lambda1','tails','initial_values']:
                assert np.array_equal(arrays[name][:,:,0,:],arrays[name][:,:,1,:]),name
        np.savez_compressed(output/'outcomes.npz',**arrays)
        # Columns are documented; all retained numeric trace fields are exact
        # conversions of the actual native/Torch outputs, not resimulated data.
        columns=['state_index','branch','tick','obs_x','obs_v','obs_a','obs_w','next_x','next_v','next_a','next_w',
            'before_x','before_v','before_a','before_w','after_x','after_v','after_a','after_w',
            'latent','force','reward','v_pre','v_post','next_v_pre','next_v_post','terminated','truncated']
        np.savez_compressed(output/'replicate0-traces.npz',rows=np.array(trace,dtype='f8'),columns=np.array(columns))
        actual_count=int(lib.rr_probe_interactions())-before_count
        assert actual_count==int(lengths.sum())
        for snapshot,net in zip(snapshots,[actor,*critics]):
            assert snapshot==tuple(p.detach().numpy().tobytes() for p in net.parameters())
        assert torch.equal(before_rng,torch.random.get_rng_state())
        result=dict(arm=arm,seed=203,states=ns,repeats=repeats,branches=['minus','plus'],critics=['pre','post'],
            shift=shift,sigma=float(sigma),gamma=GAMMA,lambda_=LAMBDA,horizon=horizon,action_seed=int(action_seed),
            control=control,physical_interactions=actual_count,new_training_interactions=0,
            inference_checks=inference,lambda1_telescoping_max_error=telescoping,
            weights_unchanged=True,torch_rng_unchanged=True,source_hashes=panel['source_hashes'],
            versions=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__))
        write(output/'metadata.json',result)
        print('CREDIT COMPLETE',arm,'control' if control else 'main',actual_count,flush=True)
        return result
    finally:
        for handle in handles:assert lib.rr_probe_free(handle)==0

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','controls','measure'])
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--panels',type=Path);p.add_argument('--library',type=Path);p.add_argument('--arm',choices=ARMS)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6'
    if a.mode=='select':select_all(a.inputs,a.output);return
    panels=json.loads(a.panels.read_text());lib=library(a.library)
    (a.output/'torch-config.txt').write_text(torch.__config__.show())
    try:
        if a.mode=='controls':
            for arm in ARMS:
                panel=dict(panels[arm]);panel['states']=panel['states'][:2]
                run_case(lib,a.inputs/f'critic-{arm}-203',panel,arm,a.output/arm,repeats=4,horizon=60,shift=0,control=True)
            print('CREDIT CONTROLS PASS',flush=True)
        else:
            assert a.arm is not None
            panel=panels[a.arm]
            for n,h in panel['source_hashes'].items():assert hashlib.sha256((a.inputs/f'critic-{a.arm}-203'/n).read_bytes()).hexdigest()==h
            run_case(lib,a.inputs/f'critic-{a.arm}-203',panel,a.arm,a.output)
    except BaseException as error:
        write(a.output/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
