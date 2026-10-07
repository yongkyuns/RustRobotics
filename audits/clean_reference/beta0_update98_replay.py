#!/usr/bin/env python3
"""Exact candidate-beta0 replay and native packet capture; issue35/comment6035029961."""
from __future__ import annotations
import argparse,copy,json,types,time
from pathlib import Path
import numpy as np
import torch
import actor_beta0_compare as beta
import residual_replay as rr
h=beta.h;p=beta.p
CPS=(0,65536,131072,262144)

class Capture(rr.Capture):
    def __init__(self,m,e,tap,out,historical=None,keep_from=65):
        super().__init__(m,e,tap,out,historical,keep_from)
        (out/'after').mkdir()
        self.moment_checks=0
        def hook(opt,args,kwargs):
            for n,q in m.policy.named_parameters():
                if beta.actor_name(n):p.need(torch.equal(opt.state[q]['exp_avg'],q.grad),'beta0 moment=current gradient')
            self.moment_checks+=1
        self.hook=m.policy.optimizer.register_step_post_hook(hook)
    def train(self):
        before=p.rng();u=self.m.num_timesteps//2048
        super().train()
        if u>=self.keep_from:torch.save(h.snapshot(self.m,self.e),self.out/'after'/f'{u:04}.pt')
    def close(self):self.hook.remove()

def preflight(ref,out,counts):
    m,e=beta.build(ref,204,'actor_beta0');m.learn(4096,reset_num_timesteps=False,log_interval=None)
    expected=h.snapshot(m,e);e.close()
    m,e=beta.build(ref,204,'actor_beta0');tap=rr.PhysicalTap(ref,e.envs[0]);cap=Capture(m,e,tap,out,keep_from=1)
    m.train=types.MethodType(lambda self:cap.train(),m)
    try:
        m.learn(4096,reset_num_timesteps=False,log_interval=None)
        p.need(p.eq(expected,h.snapshot(m,e)),'plain/packet observed policy Adam RNG episodes exact')
        p.need(cap.moment_checks==640,'all640 actor moment checks')
        p.write(out/'PREFLIGHT.json',dict(observer_exact=True,moment_checks=640,training_steps=8192,native_steps=counts['native_steps']))
    finally:cap.close();tap.close();e.close()

def replay(ref,out,counts,prior):
    hist=prior/'pair-204/actor_beta0'
    oldcov=[json.loads(x) for x in (hist/'coverage.jsonl').read_text().splitlines()]
    oldeval=json.loads((hist/'episodes.json').read_text())
    m,e=beta.build(ref,204,'actor_beta0');p.need(p.eq(p.weights(m),rr.original_weights(hist,0)),'original initialization exact')
    tap=rr.PhysicalTap(ref,e.envs[0]);cap=Capture(m,e,tap,out,oldcov);m.train=types.MethodType(lambda self:cap.train(),m)
    checked=[];rows=[];eval_steps=0
    try:
        for cp in CPS:
            if cp:m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
            beta.check_config(m,'actor_beta0')
            p.need(p.eq(p.weights(m),rr.original_weights(hist,cp)),'exact original policy '+str(cp))
            # Match the original checkpoint exporter, including empty Adam state entries at cp0.
            beta.export_state(m,e,out,cp)
            # Original snapshot files are hash-verified retained experiment outputs.
            old=torch.load(hist/f'snapshot-{cp}.pt',map_location='cpu',weights_only=False)
            p.need(p.eq(old,h.snapshot(m,e)),'exact full checkpoint '+str(cp))
            n=counts['native_steps'];got=h.evaluate_checkpoint(ref,m,e,beta.EVAL_BASE+204,cp)
            p.need(got==[r for r in oldeval if r['checkpoint']==cp],'exact original evaluator '+str(cp))
            used=counts['native_steps']-n;p.need(used==sum(r['steps'] for r in got),'evaluator-call count');eval_steps+=used;rows+=got
            checked.append(dict(checkpoint=cp,policy_exact=True,optimizer_rng_visible_episode_exact=True,evaluation_exact=True))
            p.write(out/'checkpoint-identities.json',checked);print('EXACT',cp,flush=True)
        p.need(len(cap.rows)==128 and cap.moment_checks==40960,'128 exact original updates')
        p.write(out/'episodes.json',rows);p.write(out/'training-episodes.json',[x.episodes for x in e.envs])
        torch.save(h.snapshot(m,e),out/'final-state.pt')
        p.need(counts['native_steps']==262144+eval_steps,'all replay calls')
        p.write(out/'COMPLETE.json',dict(training_steps=262144,completed_updates=128,actual_beta0_checks=40960,evaluation_steps=eval_steps,native_steps=counts['native_steps'],all_checkpoints_exact=True,captured_batches=64))
    finally:cap.close();tap.close();e.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','replay']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    ref,c=beta.setup(a.root/'prior/inputs/native.zip',a.out)
    try:
        if a.stage=='preflight':preflight(ref,a.out,c)
        else:replay(ref,a.out,c,a.root/'prior')
    finally:p.write(a.out/'ACTUAL_CALLS.json',c);beta.manifest(a.out)
if __name__=='__main__':main()
