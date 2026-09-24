#!/usr/bin/env python3
"""Strict complete-cohort accounting. No absent job becomes a zero score."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

SEEDS = (41001,41002,41004,41006)
ARMS = ('extra-ordinary','extra-transient')
CPS = (0,1024,4096,4608)
SHORT = ('reset-det','reset-stoch','outward-stoch','stress-det','stress-stoch')
LONG = ('long-det','long-stoch','outward-long')
CAPS = dict(zip(SHORT+LONG,(2048,2048,2048,2048,2048,30000,30000,6000)))
PROTOCOL = 5816968405


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def interval(values: list[float], factor: float=2.586) -> dict:
    require(len(values)>1 and all(math.isfinite(x) for x in values),'bad interval data')
    mean = sum(values)/len(values)
    se = statistics.stdev(values)/math.sqrt(len(values))
    return {'n':len(values),'mean':mean,'interval_99':[mean-factor*se,mean+factor*se]}


def paired(candidate: list[dict], control: list[dict]) -> dict:
    require(len(candidate)==len(control)>1,'unpaired sizes')
    require(all(a['rep']==b['rep'] and a['key']==b['key'] for a,b in zip(candidate,control)),'unpaired identities')
    gains = sum(a['ending']=='timeout' and b['ending']!='timeout' for a,b in zip(candidate,control))
    losses = sum(a['ending']!='timeout' and b['ending']=='timeout' for a,b in zip(candidate,control))
    return {'gained':gains,'lost':losses,'net':gains-losses,
            **{metric:interval([float(a[metric])-float(b[metric]) for a,b in zip(candidate,control)]) for metric in ('discounted','total')}}


def f32(value: float) -> float:
    return struct.unpack('<f', struct.pack('<f', value))[0]


def float_bits(values: list) -> bytes:
    flat=[]
    for v in values:
        flat.extend(v if isinstance(v,list) else [v])
    require(all(math.isfinite(float(v)) for v in flat),'nonfinite array')
    return struct.pack('<'+'f'*len(flat),*flat)


def normalize(raw: list[float]) -> list[float]:
    require(bool(raw),'empty advantages')
    raw=[f32(x) for x in raw]
    mean=f32(sequential_sum(raw)/len(raw))
    variance=f32(sequential_sum([f32(f32(x-mean)*f32(x-mean)) for x in raw])/len(raw))
    scale=max(f32(math.sqrt(variance)),f32(1e-6))
    return [f32(f32(x-mean)/scale) for x in raw]


def sequential_sum(values: list[float]) -> float:
    result=0.0
    for value in values: result=f32(result+value)
    return result


def check_targets(batch: dict) -> None:
    n=len(batch['rewards'])
    require(n>0 and batch['ends'][-1],'incomplete stream')
    for field in ('values','terminals','timeouts','ends','bootstraps','raw1','returns1','remaining'):
        require(len(batch[field])==n,'inconsistent stream shapes')
    start=0;gamma=f32(.995)
    for end in range(n):
        require(type(batch['terminals'][end]) is bool and type(batch['ends'][end]) is bool,'bad terminal flag')
        require(not batch['terminals'][end] or batch['ends'][end],'unmarked terminal')
        if not batch['ends'][end]:continue
        nv=f32(batch['bootstraps'][end]);gae=0.0
        require(not batch['terminals'][end] or nv==0.0,'terminal bootstrap')
        for i in range(end,start-1,-1):
            nt=0.0 if batch['terminals'][i] else 1.0
            value=f32(batch['values'][i]);reward=f32(batch['rewards'][i])
            delta=f32(f32(reward+f32(f32(gamma*nv)*nt))-value)
            gae=f32(delta+f32(f32(gamma*nt)*gae))
            require(float_bits([gae])==float_bits([batch['raw1'][i]]),'wrong raw GAE')
            require(float_bits([f32(gae+value)])==float_bits([batch['returns1'][i]]),'wrong return')
            require(batch['remaining'][i]==end-i+1,'wrong remaining support')
            nv=value
        start=end+1
    require(start==n,'unclosed stream')


def read_cases(path: Path, seed: int, arm: str) -> dict:
    raw = rows(path)
    require(len(raw)==2336,'incomplete evaluation: expected2336 records')
    result = {}
    for r in raw:
        require(int(r['seed'])==seed and r['arm']==arm,'wrong seed/arm')
        cp,panel,rep = int(r['checkpoint']),r['panel'],int(r['rep'])
        require(cp in CPS and panel in CAPS,'unknown checkpoint/panel')
        require(panel not in LONG or cp==4608,'unexpected long panel')
        n = 32 if panel in LONG else 256 if cp==4608 else 64
        require(0<=rep<n,'wrong replicate')
        key=(cp,panel,rep)
        require(key not in result,'duplicate case')
        cap,steps = int(r['cap']),int(r['steps'])
        require(cap==CAPS[panel] and 0<steps<=cap,'bad horizon')
        require(r['ending'] in ('timeout','position','angle','both'),'bad ending')
        require(r['ending']!='timeout' or steps==cap,'false timeout')
        position=f32(float(r['max_position']))>f32(2.4)
        angle=f32(float(r['max_angle']))>f32(.6)
        ending='both' if position and angle else 'position' if position else 'angle' if angle else 'timeout'
        require(r['ending']==ending,'incorrect boundary classification')
        require(r['centered'] in ('true','false'),'bad centered flag')
        require(r['ending']=='timeout' or r['centered']=='false','centered terminal')
        require(all(math.isfinite(float(r[x])) for x in ('total','discounted','max_position','max_angle','force_rms')),'nonfinite outcome')
        require(float(r['max_position'])>=0 and float(r['max_angle'])>=0 and float(r['force_rms'])>=0,'negative diagnostic')
        result[key] = r
    expected = {(cp,p,i) for cp in CPS for p in SHORT+(LONG if cp==4608 else ()) for i in range(32 if p in LONG else 256 if cp==4608 else 64)}
    require(set(result)==expected,'missing case')
    return result


def verify_single(path: Path) -> dict:
    complete=json.loads((path/'complete.json').read_text())
    seed,arm = complete['seed'],complete['arm']
    require(seed in SEEDS and arm in ARMS,'unregistered run')
    for name,value in {'execution':'complete','protocol':PROTOCOL,'from_scratch':True,'updates':4608,
                       'added_transitions':21233664,'actor_steps':110592,'critic_steps':110592,
                       'sample_visits':28311552,'evaluation_records':2336}.items():
        require(complete.get(name)==value,f'bad completion field {name}')
    cases=read_cases(path/'evaluation.csv',seed,arm)
    require(sum(int(r['steps']) for r in cases.values())==complete['evaluation_steps'],'evaluation cost mismatch')
    updates=rows(path/'updates.csv')
    require(len(updates)==4608,'incomplete updates')
    for i,r in enumerate(updates,1):
        for name,value in {'update':i,'primary':512,'supplement':8704,'added':4608,'actor_steps':24,'critic_steps':24,'sample_visits':6144}.items():
            require(int(r[name])==value,f'bad update field {name}')
        require(0<=int(r['tail'])<=4096,'bad tail cost')
        require(all(math.isfinite(float(r[k])) for k in ('policy_loss','value_loss')),'nonfinite loss')
    for cp in CPS:
        for kind in ('actor','critic'):
            require((path/f'{kind}-{cp}.bin').is_file(),'missing checkpoint')
    detailed={}
    for u in (1,4097,4608):
        d=path/f'update-{u}'
        original=json.loads((d/'original-union.json').read_text())
        augmented=json.loads((d/'augmented-union.json').read_text())
        require(len(original['raw'])==1024 and len(augmented['raw'])==1536,'wrong union size')
        for field in ('observations','latents','old_log_probs','returns','raw'):
            require(float_bits(augmented[field][:1024])==float_bits(original[field]),'original prefix changed')
        require(float_bits(normalize(augmented['raw']))==float_bits(augmented['normalized']),'wrong global normalization')
        for stream in range(4):
            sd=d/'added'/f'stream-{stream}'
            identity=json.loads((sd/'identity.json').read_text())
            require(identity['seed']==seed and identity['update']==u and identity['stream']==stream,'bad stream identity')
            require(identity['key']==(0x5243545241490001^(seed<<32)^(u<<8)^stream),'bad training key')
            require(identity['stress']==(arm=='extra-transient'),'wrong start stratum')
            x,v,theta,omega=identity['initial']
            require(abs(x)<=.2 and abs(v)<=.4 and abs(theta)<=.25 and abs(omega)<=.5,'invalid initial range')
            if arm=='extra-transient': require(abs(theta)>=.125 and abs(omega)>=.25 and theta*omega>0,'not a stress start')
            data=json.loads((sd/'batch.json').read_text());check_targets(data)
            require(len(data['rewards'])==1152,'bad added collection budget')
            begin=1024+stream*128
            for dest,source in [('observations','observations'),('latents','latents'),('old_log_probs','old_log_probs'),('returns','returns1'),('raw','raw1')]:
                require(float_bits(augmented[dest][begin:begin+128])==float_bits(data[source][:128]),'added rows do not bind to streams')
            for i,left in enumerate(data['remaining'][:128]):
                require(left>1024 or data['terminals'][i+left-1],'insufficient return support')
        records=[json.loads((d/'optimizer'/f'step-{i}.json').read_text()) for i in range(1,25)]
        # The inherited observer writes indices in its existing JSON schema.
        indices=[]
        for i,r in enumerate(records,1):
            require(len(r['indices'])==256,'wrong minibatch size')
            indices.extend(r['indices'])
            for kind in ('actor','critic'):
                require((d/'optimizer'/f'{kind}-{i}.bin').is_file(),'missing optimizer weights')
        for offset in range(0,len(indices),1536):
            require(sorted(indices[offset:offset+1536])==list(range(1536)),'incomplete fitting epoch')
        detailed[str(u)]={'rows':1536,'steps':24,'original_prefix_preserved':True}
    summary={}
    for cp in CPS:
        summary[str(cp)]={}
        for panel in SHORT+(LONG if cp==4608 else ()):
            items=[r for (c,p,_),r in cases.items() if c==cp and p==panel]
            summary[str(cp)][panel]={'n':len(items),'completed':sum(r['ending']=='timeout' for r in items),
                'mean_discounted':sum(float(r['discounted']) for r in items)/len(items),
                'mean_total':sum(float(r['total']) for r in items)/len(items)}
    return {'seed':seed,'arm':arm,'complete':complete,'summary':summary,'detailed':detailed,
            'scope':'complete native run accounting; not independent reconstruction of every unexported training batch or optimizer moment'}


def combine(paths: list[Path]) -> dict:
    require(len(paths)==8,'all eight runs are required')
    runs={}
    for path in paths:
        r=verify_single(path);key=(r['seed'],r['arm'])
        require(key not in runs,'duplicate run')
        runs[key]=(path,r)
    require(set(runs)=={(s,a) for s in SEEDS for a in ARMS},'missing cohort run')
    per_seed={};pooled_stress_gain=0;screen=True
    for seed in SEEDS:
        control=runs[seed,ARMS[0]][0];candidate=runs[seed,ARMS[1]][0]
        for kind in ('actor','critic'):
            require((control/f'{kind}-0.bin').read_bytes()==(candidate/f'{kind}-0.bin').read_bytes(),'initialization mismatch')
        # Original initial union must be identical before arm-specific extra rows.
        require((control/'update-1/original-union.json').read_bytes()==(candidate/'update-1/original-union.json').read_bytes(),'first original union differs')
        c=read_cases(candidate/'evaluation.csv',seed,ARMS[1]);b=read_cases(control/'evaluation.csv',seed,ARMS[0])
        per_seed[str(seed)]={}
        for panel in SHORT+LONG:
            n=256 if panel in SHORT else 32
            contrast=paired([c[4608,panel,i] for i in range(n)],[b[4608,panel,i] for i in range(n)])
            per_seed[str(seed)][panel]=contrast
            if panel in SHORT[:3]:
                screen &= contrast['net']>=0 and contrast['discounted']['interval_99'][1]>=0
            if panel.startswith('stress-'): pooled_stress_gain += contrast['net']
    effects={}
    for panel in SHORT+LONG:
        effects[panel]={metric:interval([per_seed[str(s)][panel][metric]['mean'] for s in SEEDS],5.840909309733352) for metric in ('discounted','total')}
        n=256 if panel in SHORT else 32
        effects[panel]['completion_rate']=interval([per_seed[str(s)][panel]['net']/n for s in SEEDS],5.840909309733352)
    return {'protocol':PROTOCOL,'runs':8,'training_seeds':4,'per_seed':per_seed,'paired_training_seed_effects':effects,
            'pooled_stress_gain':pooled_stress_gain,'local_component_screen':bool(screen and pooled_stress_gain>0),
            'note':'Unadjusted descriptive intervals. No independent-training-seed claim from pooling episodes; aggregate preservation is not casewise safety. No adoption.'}


def manifest(root: Path) -> None:
    data={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('single','combine','manifest'));p.add_argument('paths',type=Path,nargs='+');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode=='manifest':
        require(len(a.paths)==1,'one manifest root');manifest(a.paths[0]);return
    report=verify_single(a.paths[0]) if a.mode=='single' else combine(a.paths)
    text=json.dumps(report,sort_keys=True,indent=2)+'\n'
    if a.output: a.output.write_text(text)
    print(text)


if __name__=='__main__':main()
