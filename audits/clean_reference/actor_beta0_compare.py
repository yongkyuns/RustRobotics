#!/usr/bin/env python3
"""Fixed from-scratch actor beta1=0 test. Issue35/comment6034283959.
Only standard Adam parameter groups differ. No PPO/GAE/gradient code replaced.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, inspect, os, time
from pathlib import Path
import numpy as np
import torch
import physical_horizon_compare as h
p=h.pool
CPS=(0,65536,131072,262144,524288,786432,1048576)
ARMS={"reference":.9,"actor_beta0":0.}
SEEDS=(201,202,203,204)
EVAL_BASE=11960000

def actor_name(name):
    if name=="log_std" or name.startswith(("mlp_extractor.policy_net.","action_net.")):return True
    if name.startswith(("mlp_extractor.value_net.","value_net.")):return False
    raise AssertionError("Unclassified/shared trainable parameter: "+name)

def named_optimizer(m):
    return {n:copy.deepcopy(m.policy.optimizer.state[q]) for n,q in m.policy.named_parameters()}

def named_snapshot(m,e):
    s=h.snapshot(m,e);s["adam"]=named_optimizer(m);return s

def group_options(m):
    return [{**{k:v for k,v in g.items() if k!="params"},
             "names":[n for n,q in m.policy.named_parameters() if any(q is x for x in g["params"])]}
             for g in m.policy.optimizer.param_groups]

def build(ref,seed,arm,**kwargs):
    m,e=h.build(ref,seed,"physical",**kwargs)
    if arm=="stock":return m,e
    beta=ARMS[arm]
    named=list(m.policy.named_parameters())
    p.need(len(named)==13 and len({q.data_ptr() for _,q in named})==13,"disjoint parameter storage")
    groups=[dict(params=[q for n,q in named if actor_name(n)],betas=(beta,.999)),
            dict(params=[q for n,q in named if not actor_name(n)],betas=(.9,.999))]
    before=p.weights(m);rng=p.rng()
    m.policy.optimizer=torch.optim.Adam(groups,**copy.deepcopy(m.policy.optimizer.defaults))
    p.need(p.eq(before,p.weights(m)) and p.eq(rng,p.rng()),"grouping leaves initialization/RNG unchanged")
    check_config(m,arm)
    return m,e

def check_config(m,arm):
    opt=m.policy.optimizer
    p.need(type(opt) is torch.optim.Adam and opt.step.__func__ is torch.optim.Adam.step,"stock Adam")
    expected=ARMS[arm]
    for g in opt.param_groups:
        for q in g["params"]:
            name=next(n for n,v in m.policy.named_parameters() if q is v)
            p.need(g["betas"]==((expected if actor_name(name) else .9),.999),"only actor beta1 differs")
        p.need(g["lr"]==3e-4 and g["eps"]==1e-5 and g["weight_decay"]==0 and
               g["amsgrad"] is False and g["maximize"] is False,"frozen Adam options")
    p.need((m.n_steps,m.batch_size,m.n_epochs,m.gamma,m.gae_lambda,m.max_grad_norm)==
           (2048,64,10,.999,1.,.5),"frozen PPO options")

def export_state(m,e,folder,cp):
    arrays={}
    for n,q in m.policy.named_parameters():
        for key,value in m.policy.optimizer.state[q].items():
            p.need(torch.is_tensor(value),"tensor optimizer state")
            arrays[n+"/"+key]=value.detach().cpu().numpy().copy()
    np.savez_compressed(folder/f"adam-{cp}.npz",**arrays)
    # This snapshot omits the native environment's private RNG/physics slots; not standalone exact resume.
    torch.save(h.snapshot(m,e),folder/f"snapshot-{cp}.pt")
    p.write(folder/f"optimizer-{cp}.json",group_options(m))

def setup(raw,out):
    out.mkdir(parents=True,exist_ok=False)
    p.write(out/"runtime.json",h.runtime())
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ref,receipt=p.load(raw,out/"native");p.write(out/"native-receipt.json",receipt)
    original=ref.LIB.rr_env_step
    counts={"native_steps":0}
    def counted(*args):
        value=original(*args);counts["native_steps"]+=1;return value
    ref.LIB.rr_env_step=counted
    return ref,counts

def preflight(ref,out,counts):
    tests=[]
    # Original bootstrap / independent GAE / observer / evaluation controls, no source modifications.
    base=out/"base";base.mkdir();h.preflight(ref,base)
    tests.extend(json.loads((base/"PREFLIGHT.json").read_text())["passed"])
    train_cost=json.loads((base/"PREFLIGHT.json").read_text())["training_environment_steps"]
    states=[];initial=[]
    for arm in ("stock","reference"):
        m,e=build(ref,181201,arm);initial.append(p.weights(m))
        m.learn(4096,reset_num_timesteps=False,log_interval=None)
        states.append(named_snapshot(m,e));e.close();train_cost+=4096
    p.need(p.eq(initial[0],initial[1]) and p.eq(states[0],states[1]),"grouped reference exactly matches stock")
    tests.append("grouped_reference_stock_policy_named_Adam_RNG_environment_exact")
    physical=[];initial=[];rr=[]
    for arm in ARMS:
        m,e=build(ref,181202,arm,capture=True);initial.append(p.weights(m))
        h.collect(m,2048);train_cost+=2048
        physical.append(copy.deepcopy(e.envs[0].raw));rr.append(p.rng());e.close()
    p.need(p.eq(initial[0],initial[1]) and p.eq(physical[0],physical[1]) and p.eq(rr[0],rr[1]),"equal first2048 trajectories")
    tests.append("initial_parameters_first2048_trajectory_and_RNG_pair_exact")
    # Actual beta0 gradient and moment verification through stock Adam.
    m,e=build(ref,181203,"actor_beta0")
    checked=[0]
    def hook(opt,args,kwargs):
        for n,q in m.policy.named_parameters():
            if actor_name(n):
                p.need(torch.equal(opt.state[q]["exp_avg"],q.grad),"beta0 first moment equals current clipped gradient")
        checked[0]+=1
    handle=m.policy.optimizer.register_step_post_hook(hook)
    m.learn(4096,reset_num_timesteps=False,log_interval=None);train_cost+=4096
    reference=named_snapshot(m,e);handle.remove();e.close()
    p.need(checked[0]==640,"actual beta0 step checks")
    tests.append("640_actual_beta0_moment_equals_current_gradient_checks")
    m,e=build(ref,181203,"actor_beta0");record=p.observe(m,e)
    m.learn(4096,reset_num_timesteps=False,log_interval=None);train_cost+=4096
    p.need(p.eq(reference,named_snapshot(m,e)) and len(record.rows)==2,"beta0 observation has no training effect")
    tests.append("candidate_observer_policy_Adam_RNG_episode_identity")
    before=h.snapshot(m,e);one=h.evaluate_checkpoint(ref,m,e,EVAL_BASE+181203,4096)
    two=h.evaluate_checkpoint(ref,m,e,EVAL_BASE+181203,4096)
    p.need(one==two and p.eq(before,h.snapshot(m,e)),"candidate eval no effect")
    e.close();tests.append("candidate_repeat_evaluator_and_state_identity")
    m,e=build(ref,181204,"actor_beta0")
    rows=h.evaluate_checkpoint(ref,m,e,EVAL_BASE+181204,0)
    p.need(m.num_timesteps==0 and m._last_obs is None and len(rows)==96,"candidate checkpoint0")
    # Disjoint loss gradients on actual-sized inputs (normal synthetic diagnostic observations).
    o=torch.zeros((64,4));a=torch.zeros((64,1))
    value,lp,entropy=m.policy.evaluate_actions(o,a)
    params=list(m.policy.named_parameters())
    ag=torch.autograd.grad(lp.mean(),[q for _,q in params],allow_unused=True,retain_graph=True)
    vg=torch.autograd.grad(value.mean(),[q for _,q in params],allow_unused=True)
    p.need(all(v is None if actor_name(n) else u is None for (n,_),u,v in zip(params,ag,vg)),"no cross network gradients")
    e.close();tests.extend(["candidate_uninitialized_checkpoint0","disjoint_actor_critic_loss_graphs"])
    p.write(out/"PREFLIGHT.json",dict(passed=tests,training_transitions=train_cost,
        native_steps=counts["native_steps"],evaluation_steps=counts["native_steps"]-train_cost,
        full_training_results=False))
    print("PREFLIGHT PASS",len(tests),flush=True)

def measure(ref,out,seed,counts):
    p.need(seed in SEEDS,"fixed seed")
    starts=[];initial_rows=[]
    for arm in ARMS:
        folder=out/arm;folder.mkdir();m,e=build(ref,seed,arm)
        starts.append(p.weights(m))
        if len(starts)==2:p.need(p.eq(starts[0],starts[1]),"paired random initialization exact")
        p.write(folder/"config.json",dict(seed=seed,arm=arm,actor_beta1=ARMS[arm],critic_beta1=.9,
            gamma=m.gamma,gae_lambda=m.gae_lambda,training_steps=1048576,n_steps=2048,
            batch_size=64,n_epochs=10,training_timeout=256,evaluation_domain=EVAL_BASE+seed,
            optimizer=group_options(m),checkpoints=CPS))
        rec=p.observe(m,e,folder/"coverage.jsonl");rows=[];checkpoints=[];t0=time.monotonic()
        arm_calls=counts["native_steps"];eval_calls=0;train_seconds=0.
        try:
            for cp in CPS:
                if cp:
                    begin=time.monotonic()
                    m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
                    train_seconds+=time.monotonic()-begin
                check_config(m,arm)
                p.need(m.num_timesteps==cp,"exact checkpoint")
                w=p.weights(m);p.need(all(bool(torch.isfinite(v).all()) for v in w.values()),"finite checkpoint")
                np.savez_compressed(folder/f"policy-{cp}.npz",**{k:v.numpy() for k,v in w.items()})
                export_state(m,e,folder,cp)
                before=counts["native_steps"];panel=h.evaluate_checkpoint(ref,m,e,EVAL_BASE+seed,cp)
                used=counts["native_steps"]-before
                p.need(sum(r["steps"] for r in panel)==used,"exact evaluation step count")
                eval_calls+=used;rows+=panel
                if cp==0:
                    initial_rows.append(panel)
                    if len(initial_rows)==2:p.need(initial_rows[0]==panel,"paired initial evaluation exact")
                checkpoints.append(dict(checkpoint=cp,policy_sha256=p.model_digest(w),
                                        eval_native_calls=used,train_seconds=train_seconds))
                p.write(folder/"episodes.json",rows);p.write(folder/"checkpoints.json",checkpoints)
                p.write(folder/"training-episodes.json",[x.episodes for x in e.envs])
                det=sum(x["ending"]=="timeout" for x in panel if x["mode"]=="deterministic")
                stoch=sum(x["ending"]=="timeout" for x in panel if x["mode"]=="stochastic")
                print("CHECKPOINT",seed,arm,cp,det,stoch,flush=True)
            p.need(len(rec.rows)==512 and all(r["episode_starts"][0]>=8 for r in rec.rows),"512 covered updates")
            p.need(all(int(v["step"])==163840 for v in m.policy.optimizer.state.values()),"equal optimizer budgets")
            p.need(counts["native_steps"]-arm_calls==1048576+eval_calls,"all native calls accounted")
            p.write(folder/"receipt.json",dict(seed=seed,arm=arm,actor_beta1=ARMS[arm],
                training_steps=m.num_timesteps,rollout_updates=len(rec.rows),optimizer_steps=163840,
                initial_sha256=p.model_digest(starts[-1]),final_sha256=p.model_digest(w),
                evaluation_records=len(rows),evaluation_steps=eval_calls,wall_seconds=time.monotonic()-t0,
                train_seconds=train_seconds))
        finally:e.close()
    p.write(out/"COMPLETE.json",dict(seed=seed,arms=list(ARMS),paired_initial_exact=True,
        training_steps=2097152,native_steps=counts["native_steps"],
        evaluation_steps=counts["native_steps"]-2097152,production_changed=False))

def manifest(out):
    p.write(out/"MANIFEST.json",{str(q.relative_to(out)):p.sha(q.read_bytes())
              for q in sorted(out.rglob("*")) if q.is_file() and q.name!="MANIFEST.json"})

def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["preflight","measure"])
    ap.add_argument("--raw",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int);a=ap.parse_args()
    ref,c=setup(a.raw,a.out)
    try:
        if a.stage=="preflight":preflight(ref,a.out,c)
        else:measure(ref,a.out,a.seed,c)
    finally:
        p.write(a.out/"ACTUAL_CALLS.json",c)
        src=a.out/"executed-source";src.mkdir(exist_ok=True)
        for q in (Path(__file__),Path(h.__file__),Path(p.__file__)):
            (src/q.name).write_bytes(q.read_bytes())
        manifest(a.out)
if __name__=="__main__":main()
