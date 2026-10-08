#!/usr/bin/env python3
"""Reconstruct exact seed201 PPO and capture contemporaneous incoming actors.
No learning change. Protocol: RustRobotics issue #35 comment 6061891252.
"""
from pathlib import Path
import argparse, copy, hashlib, json, sys, types, time
import numpy as np
import torch
import stable_baselines3 as sb3

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import value_batch_compare_20261007 as vb
p = vb.pool
from ledger import assert_episode_ledger
SEL = json.loads((HERE / 'SELECTION.json').read_text())
PAIRS = SEL['pairs']
SELECTED_UPDATES = {v[side+'_update'] for v in PAIRS for side in ('failure','timeout')}
FIELDS = ('observations','actions','rewards','values','log_probs','advantages','returns','episode_starts','packets')
assert len(PAIRS)==16 and len(SELECTED_UPDATES)==32

def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)+'\n')

class Tap:
    def __init__(self, ref, env):
        self.ref=ref; self.env=env; self.last=None; self.rows=[]; self.total=0
        self.original_reset=ref.LIB.rr_env_reset
        self.original_step=ref.LIB.rr_env_step
        ref.LIB.rr_env_reset=self.reset
        ref.LIB.rr_env_step=self.step
    def reset(self,h):
        r=self.original_reset(h)
        if h == self.env.handle:
            assert r.status==0
            self.last=(list(r.state),list(r.observation),int(r.steps))
        return r
    def step(self,h,a,transform):
        r=self.original_step(h,a,transform)
        if h == self.env.handle:
            assert self.last is not None and r.status==0 and r.steps==self.last[2]+1
            state,obs,age=self.last
            self.rows.append([*state,*obs,*r.state,*r.observation,float(a),r.force,r.reward,
                              age,r.terminated,r.truncated])
            self.last=(list(r.state),list(r.observation),int(r.steps))
            self.total+=1
        return r
    def close(self):
        self.ref.LIB.rr_env_reset=self.original_reset
        self.ref.LIB.rr_env_step=self.original_step

class Capture(vb.Recorder):
    def __init__(self,model,env,tap,out,history=None,selected=None):
        super().__init__(model,env,out/'diagnostics.jsonl')
        self.tap=tap
        self.out=out
        self.history=history
        self.selected=selected or set()
        self.full_sha=hashlib.sha256()
        self.selected_packets={}
        (out/'actors').mkdir(exist_ok=True)
    def train(self):
        u=self.m.num_timesteps//2048
        rb=self.m.rollout_buffer
        packet=np.asarray(self.tap.rows,np.float32)
        assert packet.shape==(2048,22),(u,packet.shape)
        assert np.array_equal(packet[:,4:8],rb.observations[:,0]),('observation',u)
        assert np.array_equal(packet[:,19]==0,rb.episode_starts[:,0].astype(bool)),('episode starts',u)
        assert np.array_equal(packet[:,17],np.float32(20)*np.clip(rb.actions[:,0,0],-1,1)),('force',u)
        arrs={field:(packet if field=='packets' else getattr(rb,field)) for field in FIELDS}
        for field,arr in arrs.items():
            arr=np.asarray(arr)
            self.full_sha.update(f'{u:04d}:{field}:{arr.dtype}:{arr.shape}\n'.encode())
            self.full_sha.update(arr.tobytes(order='C'))
        for pair in (PAIRS if self.history is not None else []):
            for side in ('failure','timeout'):
                st=pair[side+'_start']
                if st//2048+1==u:
                    index=st%2048
                    row=packet[index].copy()
                    assert row[19]==0,('not reset',st)
                    self.selected_packets[(pair['pair_index'],side)]=dict(
                        state=row[:4].astype(float).tolist(),
                        observation=row[4:8].astype(float).tolist(),
                        start=st, update=u)
        if u in self.selected:
            weights=p.weights(self.m)
            np.savez_compressed(self.out/'actors'/f'incoming-{u:04d}.npz',
                                **{k:v.numpy() for k,v in weights.items()})
        self.tap.rows.clear()
        super().train()
        if self.history is not None:
            assert self.rows[-1]==self.history[u-1],('archived diagnostic differs',u)
        if u%16==0:print('HISTORICAL UPDATE',u,'steps',self.m.num_timesteps,flush=True)

def install(model,env,ref,out,history=None,selected=None):
    tap=Tap(ref,env.envs[0])
    cap=Capture(model,env,tap,out,history,selected)
    model.train=types.MethodType(lambda self:cap.train(),model)
    return tap,cap

def load_native(raw,out):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert sys.version_info[:2]==(3,11)
    assert torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6' and sb3.__version__=='2.9.0'
    ref,receipt=p.load(raw,out/'native')
    write(out/'native-receipt.json',receipt)
    return ref

def compare_policy(model,path):
    with np.load(path,allow_pickle=False) as z:
        weights=model.policy.state_dict()
        return set(z.files)==set(weights) and all(
            np.array_equal(z[k],v.detach().cpu().numpy()) for k,v in weights.items())

def check_preflight(ref,out):
    plain,env=vb.build(ref,775,2048)
    try:
        plain.learn(4096,reset_num_timesteps=False,log_interval=None)
        reference=vb.snapshot(plain,env)
    finally: env.close()
    model,env=vb.build(ref,775,2048)
    tap,cap=install(model,env,ref,out)
    try:
        model.learn(4096,reset_num_timesteps=False,log_interval=None)
        assert p.eq(reference,vb.snapshot(model,env)),'recorder changes PPO, Adam or RNG'
        assert tap.total==4096
        assert_episode_ledger([v.episodes for v in env.envs], json.loads(json.dumps(reference['episodes'])))
        assert not cap.selected_packets, 'generic preflight must not capture selected seed201 states'
        write(out/'PREFLIGHT.json',{'noninterference':True,'steps_each':4096,'optimizer_transactions_each':640})
        print('RECORDER PREFLIGHT PASS',flush=True)
    finally: tap.close();env.close()

def exact_replay(ref,out,control):
    model,env=vb.build(ref,201,2048)
    past_diagnostics=[json.loads(line) for line in (control/'diagnostics.jsonl').read_text().splitlines()]
    past_eval=json.loads((control/'episodes.json').read_text())
    tap,cap=install(model,env,ref,out,past_diagnostics,SELECTED_UPDATES)
    matches=[]
    try:
        for cp in vb.CPS:
            if cp:model.learn(cp-model.num_timesteps,reset_num_timesteps=False,log_interval=None)
            assert compare_policy(model,control/f'policy-{cp}.npz'),('checkpoint actor/critic mismatch',cp)
            rows=vb.evaluate(ref,model,env,201,cp)
            assert rows==[r for r in past_eval if r['checkpoint']==cp],('original episode/reward mismatch',cp)
            matches.append({'checkpoint':cp,'model_arrays_exact':True,'reward_traces_exact':True})
            write(out/'checkpoint-gates.json',matches)
            print('EXACT CHECKPOINT',cp,flush=True)
        assert len(cap.rows)==512 and tap.total==1048576
        digest=cap.full_sha.hexdigest()
        assert digest==SEL['original_archived_packet_digest'],('all 512 batch contents differ',digest)
        live_ledger = [v.episodes for v in env.envs]
        write(out/'training-episodes.json', live_ledger)
        assert_episode_ledger(live_ledger, json.loads((control/'training-episodes.json').read_text()))
        assert len(cap.selected_packets)==32
        selected_digest=hashlib.sha256()
        linked=[]
        for pair in PAIRS:
            row=dict(pair)
            for side in ('failure','timeout'):
                x=cap.selected_packets[(pair['pair_index'],side)]
                row[side]=x
                selected_digest.update(f"{pair['pair_index']}:{side}:{x['start']}\n".encode())
                selected_digest.update(np.array(x['state']+x['observation'],np.float32).tobytes())
            linked.append(row)
        assert selected_digest.hexdigest()==SEL['selected_packet_digest'],'selected physical/noisy reset packets differ'
        snaps=sorted((out/'actors').glob('incoming-*.npz'))
        assert len(snaps)==32 and {int(s.stem.split('-')[1]) for s in snaps}==SELECTED_UPDATES
        write(out/'LINKED.json',linked)
        hashes={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in snaps}
        write(out/'ACTOR_HASHES.json',hashes)
        write(out/'COMPLETE.json',{
            'historical_checkpoints_exact':len(matches),'historical_diagnostics_exact':len(cap.rows),
            'historical_episode_ledger_exact':True,'historical_physical_packets_sha256':digest,
            'selected_packet_sha256':selected_digest.hexdigest(),'actors_exact_incoming':len(snaps),
            'historical_training_interactions':tap.total,'new_production_training':0,
            'selected_indices':[pair['pair_index'] for pair in PAIRS]})
        print('HISTORICAL REPLAY AND ACTOR EXTRACTION PASS',len(snaps),'snapshots',flush=True)
    finally: tap.close();env.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['preflight','replay'])
    parser.add_argument('--raw',required=True,type=Path)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    ref=load_native(args.raw,args.out)
    if args.stage=='preflight':check_preflight(ref,args.out)
    else:
        assert (args.control/'policy-1048576.npz').is_file()
        exact_replay(ref,args.out,args.control)

if __name__=='__main__':main()
