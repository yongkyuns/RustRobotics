#!/usr/bin/env python3
"""Complete replay ledger, retaining failures instead of stopping at one panel.
No tolerance changes: all_exact remains false on any historical mismatch, and
such traces use an UNQUALIFIED prefix. Instrumented-vs-uninstrumented equality
and all trace/policy/RNG controls remain mandatory. No training is performed.
"""
import argparse
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
import replay_checkpoints_20261006 as r
from replay_original_setup_20261006 import original_policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);inputs=out/'inputs'
    r.need(sys.version_info[:2]==(3,11),'Python version')
    r.need(torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6','numeric versions')
    r.need(sb3.__version__=='2.9.0' and gym.__version__=='1.3.0','reference versions')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    r.write(out/'verified_inputs.json',r.unpack(args.raw,inputs))
    native=inputs/'clean-ppo-native'
    r.need((native/'commit.txt').read_text().strip()=='da6623ae4b5878ae5079bf05c18c3d327570bc16','native source')
    os.environ['PPO_REF_LIBRARY']=str((native/'library/librust_robotics_train.so').resolve())
    source=native/'sources/audits/clean_reference/reference.py'
    spec=importlib.util.spec_from_file_location('archived_eval',source);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    import stable_baselines3.ppo.ppo as ppo
    import stable_baselines3.common.buffers as buffers
    import stable_baselines3.common.policies as policies
    import stable_baselines3.common.on_policy_algorithm as on_policy
    for module in (ppo,buffers,policies,on_policy):
        r.need(inspect.getsource(module).encode()==(native/'python-controls/sources'/Path(module.__file__).name).read_bytes(),'reference source identity')
    original=ref.LIB;ledger=[];all_records=[];steps=0;negative=[]
    for seed in (201,202,203,204):
        folder=inputs/f'clean-ppo-sb3-defaults-{seed}'
        r.need((folder/'reference.py').read_bytes()==source.read_bytes(),'driver identity')
        for cp in r.CHECKPOINTS:
            model=original_policy(folder,cp);before=r.policy_hash(model);wanted=r.expected(folder,cp)
            plain=ref.evaluate(model,2,seed,cp);ordered=r.ordered(plain)
            exact=ordered==wanted
            if not exact:r.write(out/f'MISMATCH-plain-{seed}-{cp}.json',{'expected':wanted,'actual':ordered})
            trace=r.TraceABI(original);ref.LIB=trace
            try:traced=ref.evaluate(model,2,seed,cp)
            finally:ref.LIB=original
            # This gate is not relaxed even for a historical mismatch.
            r.compare(traced,ordered,f'instrumentation-{seed}-{cp}',out)
            r.need(not trace.live and len(trace.finished)==96,'complete handles')
            r.need(r.policy_hash(model)==before,'read-only model')
            for row in traced:r.validate_trace(trace.finished[row['seed']],row)
            if seed==201 and cp==65536:
                row=next(a for a in traced if a['mode']=='deterministic' and a['episode']==0)
                negative=r.negative_controls(trace.finished[row['seed']],row)
            if cp:
                prefix='' if exact else 'UNQUALIFIED-'
                np.savez_compressed(out/f'{prefix}traces-{seed}-{cp}.npz',**{str(k):v for k,v in trace.finished.items()})
            diffs=[]
            for actual,old in zip(ordered,wanted):
                fields={k:[old[k],actual[k]] for k in old if old[k]!=actual[k]}
                if fields:diffs.append({'mode':old['mode'],'episode':old['episode'],'fields':fields})
            ledger.append(dict(seed=seed,checkpoint=cp,exact=exact,records=96,
                same_endings_and_lengths=all(a['ending']==b['ending'] and a['steps']==b['steps'] for a,b in zip(ordered,wanted)),
                mismatched_records=len(diffs),differences=diffs,policy_sha256=before))
            all_records.extend(traced);steps+=sum(a['steps'] for a in traced)*2
            r.write(out/'checkpoint_ledger.json',ledger)
            print('PANEL',seed,cp,'exact=',exact,'mismatched=',len(diffs),flush=True)
    all_exact=all(x['exact'] for x in ledger)
    r.write(out/'records.json',all_records)
    r.write(out/'replay_summary.json',dict(all_exact=all_exact,original_records=1536,traced_records=1536,
       exact_panels=sum(x['exact'] for x in ledger),total_panels=16,
       exact_nonzero_panels=sum(x['exact'] and x['checkpoint']>0 for x in ledger),
       simulation_steps=steps,training_updates=0,trace_columns=r.COLUMNS,mutation_controls=negative,
       python=sys.version,torch=torch.__version__,numpy=np.__version__,sb3=sb3.__version__,gymnasium=gym.__version__,
       limitation='Historical bitwise gate remains failed if all_exact=false. UNQUALIFIED traces must not be substituted for historical exact matches.'))
    print('COMPLETE REPLAY LEDGER; all_exact=',all_exact,'simulation_steps=',steps,flush=True)
    r.need(all_exact,'Historical exact-replay acceptance remains FAILED; see complete ledger, no tolerance changes')

if __name__=='__main__':main()
