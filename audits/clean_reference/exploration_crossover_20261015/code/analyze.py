"""Independent recorded-path reduction; no Torch, native calls or optimization.

Protocol reduction refuses incomplete runs and checks every original diagonal.
The separate controls mode never emits a scientific crossover result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from crossover_core import *
from physical_reader import read_group
from run_crossover import check_input_bundle

METRICS=('completions','margin50mm_completions','raw_return_mean','discounted_return_mean',
         'mean_trajectory_force_rms','mean_effort_penalty_nonterminal',
         'mean_discounted_effort_nonterminal','mean_state_penalty_nonterminal',
         'valid_steps','nonterminal_steps')
CONTRASTS={
 'sigma_S10_minus_S4_at_M4':('M4S4','M4S10'),
 'sigma_S10_minus_S4_at_M10':('M10S4','M10S10'),
 'mean_M10_minus_M4_at_S4':('M4S4','M10S4'),
 'mean_M10_minus_M4_at_S10':('M4S10','M10S10'),
}


def reduce_group(path,weights,norm,mode):
    a=load_npz(path)
    primary,error=read_group(a,weights,norm,mode)
    costs=effort_metrics(a)
    require([r['key'] for r in primary]==[r['key'] for r in costs],'Cost identities')
    rows=[dict(p,**{k:v for k,v in c.items() if k!='key'}) for p,c in zip(primary,costs)]
    receipt=json.loads(path.with_suffix('.json').read_text())
    require(receipt['trace_sha256']==digest(path),'Changed trace hash')
    require(receipt['native_transitions']==sum(r['steps'] for r in rows),'Changed transition count')
    require(receipt['cases']==rows,'Recorded case reduction mismatch')
    return a,rows,error


def group_summary(rows):
    require(len(rows)==CASES,'Complete case group required')
    return dict(completions=sum(r['completed'] for r in rows),
        margin50mm_completions=sum(r['margin50mm_complete'] for r in rows),
        raw_return_mean=float(np.mean([r['raw_return'] for r in rows])),
        discounted_return_mean=float(np.mean([r['discounted_return'] for r in rows])),
        mean_trajectory_force_rms=float(np.mean([r['force_rms_all_steps'] for r in rows])),
        mean_effort_penalty_nonterminal=float(np.mean([r['effort_penalty_nonterminal'] for r in rows])),
        mean_discounted_effort_nonterminal=float(np.mean([r['discounted_effort_nonterminal'] for r in rows])),
        mean_state_penalty_nonterminal=float(np.mean([r['state_penalty_nonterminal'] for r in rows])),
        valid_steps=sum(r['valid_steps'] for r in rows),
        nonterminal_steps=sum(r['nonterminal_steps'] for r in rows))


def summarize_factorial(groups):
    expected={(s,a,p,m) for s in SEEDS for a in ARMS for p in PANELS for m in MODES}
    require(set(groups)==expected,'All 64 declared seed/arm/panel/mode groups required')
    summaries={k:group_summary(v) for k,v in groups.items()}
    per_seed=[];effects={}
    for s in SEEDS:
        for p in PANELS:
            for m in MODES:
                cell={a:summaries[s,a,p,m] for a in ARMS}
                f={metric:factorial({a:cell[a][metric] for a in ARMS}) for metric in METRICS}
                casewise={name:{metric:paired_metrics(groups[s,a,p,m],groups[s,b,p,m],metric)
                    for metric in ('completed','margin50mm_complete','raw_return','discounted_return')}
                    for name,(a,b) in CONTRASTS.items()}
                per_seed.append(dict(seed=s,panel=p,mode=m,cells=cell,effects=f,paired=casewise))
                for metric,v in f.items():
                    for contrast,value in v.items():effects.setdefault((p,m,metric,contrast),{})[s]=value
    pooled=[dict(panel=p,mode=m,metric=metric,contrast=c,**four_seed_interval(v))
            for (p,m,metric,c),v in sorted(effects.items())]
    return dict(protocol=PROTOCOL,per_seed=per_seed,pooled=pooled,
        uncertainty='Pointwise 99% Student intervals across four exposed training seeds; modes are not extra agents.',
        effort_contract='RMS refers to clipped command force, including terminal commands; reward effort/state penalties exclude physical terminal transitions. Disturbed applied force is not exported.')


def analyze_protocol(inputs,result,out):
    check_input_bundle(inputs)
    complete=json.loads((result/'COMPLETE.json').read_text())
    require(complete['protocol']==PROTOCOL and complete['status']=='complete' and
            complete['groups']==64 and complete['trajectories']==4096 and
            complete['training_steps']==complete['optimizer_steps']==0,'Incomplete/changed execution')
    diagonal=json.loads((result/'DIAGONALS_EXACT.json').read_text())
    require(diagonal['groups']==32 and diagonal['cases']==2048 and
            diagonal['original_inputs_manifest']==digest(inputs/'INPUT_MANIFEST.json'),'Missing diagonal qualification')
    groups={};records=[];total=0;err=0.
    for s in SEEDS:
        for arm,(mi,si) in ARMS.items():
            mp=inputs/f'results/{s}-decay{mi}/{CHECKPOINT:07d}'
            sp=inputs/f'results/{s}-decay{si}/{CHECKPOINT:07d}'
            w=crossed_policy(load_npz(mp/'policy.npz'),load_npz(sp/'policy.npz'))
            n=json.loads((mp/'NORMALIZER.json').read_text())
            dest=result/f'results/{s}/{arm}'
            require(equal_arrays(w,load_npz(dest/'policy.npz')),'Changed crossed policy')
            require(n==json.loads((dest/'NORMALIZER.json').read_text()),'Wrong mean/normalizer pair')
            binding=json.loads((dest/'BINDING.json').read_text())
            require(binding==dict(seed=s,arm=arm,checkpoint=CHECKPOINT,
                mean_policy=digest(mp/'policy.npz'),sigma_policy=digest(sp/'policy.npz'),
                normalizer=digest(mp/'NORMALIZER.json'),policy_sha256=digest(dest/'policy.npz')),'Wrong source binding')
            for p in PANELS:
                sel=load_npz(inputs/f'parent/results/selection/{s}-{p}.npz')
                chosen=sel['accepted']
                require(chosen.dtype==bool and int(chosen.sum())==CASES,'Invalid original selection')
                for mode in MODES:
                    file=dest/f'{p}-{mode}.npz';a,rows,e=reduce_group(file,w,n,mode)
                    # Original evaluator asserts the complete reset state. Here we
                    # additionally bind output keys to the immutable diagonal.
                    baseline=load_npz(mp/file.name)
                    require(np.array_equal(a['keys'],sel['keys'][chosen]) and
                            np.array_equal(a['initial'],sel['observations'][chosen]) and
                            np.array_equal(a['initial_state'],sel['states'][chosen]),'Original selection mismatch')
                    require(np.array_equal(a['keys'],baseline['keys']) and
                            np.array_equal(a['initial'],baseline['initial']) and
                            np.array_equal(a['initial_state'],baseline['initial_state']),'Changed reset selection')
                    if mi==si:require(equal_arrays(a,baseline),'Archived diagonal mismatch')
                    elif mode=='deterministic':
                        same=result/f'results/{s}/M{mi}S{mi}/{file.name}'
                        require(equal_arrays(a,load_npz(same)),'Deterministic sigma dependence')
                    groups[s,arm,p,mode]=rows;total+=sum(r['steps'] for r in rows);err=max(err,e)
                    records.extend(dict(r,seed=s,arm=arm,panel=p,mode=mode) for r in rows)
    require(total==complete['native_transitions'],'Total transition mismatch')
    report=summarize_factorial(groups)
    report.update(registered_crossover_complete=True,native_transitions=total,
                  max_independent_inference_error=err,training_steps=0,optimizer_steps=0)
    write_json(out/'RESULT.json',report);write_json(out/'CASES.json',records)
    return report


def analyze_controls(reference,result,out):
    original=json.loads((reference/'analysis/CASES.json').read_text())
    allrows=[];summaries=[];total=0;err=0.
    for s in SEEDS:
        p=reference/f'results/{s}-decay/{CHECKPOINT:07d}'
        w=load_npz(p/'policy.npz');n=json.loads((p/'NORMALIZER.json').read_text())
        for panel in PANELS:
            for mode in MODES:
                file=result/f'{s}/{panel}-{mode}.npz';a,rows,e=reduce_group(file,w,n,mode)
                prior=[r for r in original if r['seed']==s and r['arm']=='decay' and
                       r['checkpoint']==CHECKPOINT and r['panel']==panel and r['mode']==mode]
                require(len(prior)==CASES,'Missing historical summaries')
                for r,z in zip(rows,prior):
                    for k in ('key','steps','ending','completed','raw_return','discounted_return','margin','minimum_tick','margin50mm_complete'):
                        require(r[k]==z[k],'Prior summary mismatch')
                total+=sum(r['steps'] for r in rows);err=max(err,e)
                allrows.extend(dict(r,seed=s,panel=panel,mode=mode) for r in rows)
                summaries.append(dict(seed=s,panel=panel,mode=mode,**group_summary(rows)))
        donor=load_npz(reference/f'results/{s}-constant/{CHECKPOINT:07d}/policy.npz')
        changed=crossed_policy(w,donor)
        a,rows,e=reduce_group(result/f'{s}/sigma_changed_deterministic_control.npz',changed,n,'deterministic')
        require(equal_arrays(a,load_npz(result/f'{s}/broad-deterministic.npz')),'Sigma control mismatch')
        total+=sum(r['steps'] for r in rows);err=max(err,e)
    require(allrows==json.loads((result/'CASES.json').read_text()),'Control case report changed')
    receipt=json.loads((result/'CONTROL_VERIFIED.json').read_text())
    require(receipt==dict(status='software_controls_only',registered_crossover_complete=False,
        reference_summaries_exact=1024,deterministic_sigma_control_trajectories=256,
        total_trajectories=1280,native_transitions=total,training_steps=0,optimizer_steps=0,
        four_pass_checkpoint_available=False),'Changed control completion receipt')
    report=dict(**receipt,max_independent_inference_error=err,groups=summaries,
                force_contract=FORCE_CONTRACT,
                causal_interpretation='None: genuine ten-pass software controls, not the missing four-pass crossover.')
    write_json(out/'CONTROL_ANALYSIS.json',report)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['protocol','controls'])
    ap.add_argument('--inputs',type=Path);ap.add_argument('--reference',type=Path)
    ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.mode=='protocol':
        require(a.inputs is not None,'--inputs required');analyze_protocol(a.inputs,a.results,a.output)
    else:
        require(a.reference is not None,'--reference required');analyze_controls(a.reference,a.results,a.output)
