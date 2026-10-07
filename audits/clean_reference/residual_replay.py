#!/usr/bin/env python3
"""Read-only exact historical replay; issue35/comment6028527470."""
from __future__ import annotations
import argparse, ctypes as C, json, types, copy, time, os
from pathlib import Path
import numpy as np
import torch
import physical_horizon_compare as horizon
pool=horizon.pool
need=pool.need
write=pool.write

class PhysicalTap:
    """Copy native result packets for the one live training handle. No native calls added."""
    def __init__(self,ref,env):
        self.ref=ref; self.env=env; self.rows=[]; self.last=None
        self.original_reset=ref.LIB.rr_env_reset; self.original_step=ref.LIB.rr_env_step
        ref.LIB.rr_env_reset=self.reset;ref.LIB.rr_env_step=self.step
    @staticmethod
    def packet(r):return np.array(r.observation[:],np.float32),np.array(r.state[:],np.float32),int(r.steps)
    def reset(self,handle):
        r=self.original_reset(handle)
        if handle==self.env.handle:self.last=self.packet(r)
        return r
    def step(self,handle,a,transform):
        r=self.original_step(handle,a,transform)
        if handle==self.env.handle:
            need(self.last is not None and r.status==0,'physical pre-state exists')
            obs,state,age=self.last; no,ns,na=self.packet(r)
            need(na==age+1,'native age continuity')
            self.rows.append((obs,state,no,ns,float(r.reward),float(r.force),int(r.terminated),int(r.truncated),age))
            self.last=(no,ns,na)
        return r
    def close(self):self.ref.LIB.rr_env_reset=self.original_reset;self.ref.LIB.rr_env_step=self.original_step

def arrays(m,tap):
    rb=m.rollout_buffer;need(len(tap.rows)==2048,'one native packet per fitting row')
    names=('obs','state','next_obs','next_state','physical_reward','command','terminated','truncated','age')
    d={n:np.array([r[i] for r in tap.rows],dtype=np.int32 if i>=6 else np.float32) for i,n in enumerate(names)}
    for n in ('observations','actions','rewards','values','log_probs','advantages','returns','episode_starts'):
        d['rb_'+n]=getattr(rb,n).copy()
    need(np.array_equal(d['obs'],rb.observations[:,0]),'exact observed states bind to rollout')
    need(np.array_equal(d['age']==0,rb.episode_starts[:,0].astype(bool)),'ages bind to episode starts')
    need(np.array_equal(d['command'],20*np.clip(rb.actions[:,0,0],-1,1)),'exact clipped command')
    live=(d['terminated']+d['truncated']==0)[:-1]
    need(np.array_equal(d['next_state'][:-1][live],d['state'][1:][live]),'physical state continuity')
    need(np.array_equal(d['next_obs'][:-1][live],d['obs'][1:][live]),'observation continuity')
    d['timeout_values']=np.zeros(2048,np.float32)
    with torch.no_grad():
        for i in np.flatnonzero(d['truncated']):
            d['timeout_values'][i]=m.policy.predict_values(m.policy.obs_to_tensor(d['next_obs'][i])[0])[0].item()
        d['last_value']=m.policy.predict_values(torch.as_tensor(m._last_obs)).numpy().copy()
    d['last_done']=m._last_episode_starts.copy()
    expected=d['physical_reward'].copy()
    for i in np.flatnonzero(d['truncated']):
        expected[i]+=m.gamma*torch.as_tensor(d['timeout_values'][i])
    need(np.array_equal(expected,rb.rewards[:,0]),'exact timeout-corrected rewards')
    return d

class Capture(pool.Recorder):
    def __init__(self,m,e,tap,out,historical=None,keep_from=129):
        super().__init__(m,e,out/'coverage.jsonl'); self.tap=tap;self.out=out;self.historical=historical;self.keep_from=keep_from
        (out/'policies').mkdir();(out/'batches').mkdir();(out/'warm').mkdir()
        self.save_policy(0)
    def save_policy(self,u):np.savez_compressed(self.out/'policies'/f'{u:04}.npz',**{k:v.numpy() for k,v in pool.weights(self.m).items()})
    def train(self):
        before=pool.rng();u=self.m.num_timesteps//2048
        d=arrays(self.m,self.tap)
        need(pool.eq(before,pool.rng()),'packet recording consumes no RNG')
        if u>=self.keep_from:
            np.savez_compressed(self.out/'batches'/f'{u:04}.npz',**d)
            torch.save(horizon.snapshot(self.m,self.e),self.out/'warm'/f'{u:04}.pt')
        self.tap.rows.clear()
        super().train()
        if self.historical is not None:need(self.rows[-1]==self.historical[u-1],f'exact original coverage update {u}')
        self.save_policy(u)
        if u%32==0: print('REPLAY',u,flush=True)

def original_weights(folder,cp):
    with np.load(folder/f'policy-{cp}.npz',allow_pickle=False) as z:return {k:torch.from_numpy(z[k].copy()) for k in z.files}

def preflight(ref,out,archive_root):
    rows=[]
    for seed in (203,204):
        m,e=horizon.build(ref,seed,'physical')
        exact=pool.eq(pool.weights(m),original_weights(archive_root/str(seed)/'pair/physical',0))
        rows.append(dict(seed=seed,initial_exact=exact));e.close()
    write(out/'initial-identities.json',rows)
    # The per-seed replay gate remains strict; a failed seed cannot be replayed.
    need(any(r['initial_exact'] for r in rows),'no seed eligible for historical replay')
    a,e=horizon.build(ref,775,'physical');a.learn(4096,reset_num_timesteps=False,log_interval=None);expected=horizon.snapshot(a,e);e.close()
    m,e=horizon.build(ref,775,'physical');tap=PhysicalTap(ref,e.envs[0]);cap=Capture(m,e,tap,out,keep_from=1)
    m.train=types.MethodType(lambda self:cap.train(),m)
    try:
        m.learn(4096,reset_num_timesteps=False,log_interval=None)
        need(pool.eq(expected,horizon.snapshot(m,e)),'plain/captured policy Adam RNG episodes observations exact')
        write(out/'PREFLIGHT.json',dict(initial_identities=rows,observer_exact=True,training_transitions=8192,captured_rows=4096))
    finally:tap.close();e.close()

def replay(ref,seed,out,archive_root):
    hist=archive_root/str(seed)/'pair/physical'
    old_cov=[json.loads(x) for x in (hist/'coverage.jsonl').read_text().splitlines()]
    old_eps=json.loads((hist/'episodes.json').read_text())
    m,e=horizon.build(ref,seed,'physical');need(pool.eq(pool.weights(m),original_weights(hist,0)),'initial tensors exact')
    tap=PhysicalTap(ref,e.envs[0]);cap=Capture(m,e,tap,out,old_cov);m.train=types.MethodType(lambda self:cap.train(),m)
    results=[]; cps=[]; start=time.time()
    try:
        for cp in horizon.CPS:
            if cp:m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
            need(pool.eq(pool.weights(m),original_weights(hist,cp)),f'exact original tensors {cp}')
            got=horizon.evaluate_checkpoint(ref,m,e,950000+seed,cp)
            need(got==[r for r in old_eps if r['checkpoint']==cp],f'exact historical evaluation {cp}')
            cps.append(dict(checkpoint=cp,policy_exact=True,evaluation_exact=True));results+=got
            write(out/'checkpoint-identities.json',cps);print('MATCHED',seed,cp,flush=True)
        need([x.episodes for x in e.envs]==[list(map(tuple,json.loads((hist/'training-episodes.json').read_text())[0]))],'exact original episode ledger')
        # Original snapshots are hash-verified trusted artifacts, never arbitrary user pickles.
        original=torch.load(hist/'final-state.pt',weights_only=False,map_location='cpu')
        need(pool.eq(original,horizon.snapshot(m,e)),'exact original final policy/Adam/RNG/live observations/episodes')
        torch.save(horizon.snapshot(m,e),out/'final-state.pt')
        write(out/'episodes.json',results);write(out/'training-episodes.json',[x.episodes for x in e.envs])
        write(out/'COMPLETE.json',dict(seed=seed,historical_checkpoints_exact=cps,coverage_records=len(cap.rows),final_full_snapshot_exact=True,training_transitions=m.num_timesteps,evaluation_transitions=sum(r['steps'] for r in results),seconds=time.time()-start))
    finally:tap.close();e.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','replay']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);write(a.out/'runtime.json',horizon.runtime())
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ref,native=pool.load(a.root/'203/inputs/native.zip',a.out/'native');write(a.out/'native-receipt.json',native)
    if a.stage=='preflight':preflight(ref,a.out,a.root)
    else:replay(ref,a.seed,a.out,a.root)
if __name__=='__main__':main()
