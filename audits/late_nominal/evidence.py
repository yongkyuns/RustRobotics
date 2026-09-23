#!/usr/bin/env python3
"""Immutable input preparation and outcome accounting. Never picks a trained policy."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import struct
import zipfile

SOURCE_SHA = 'b4e3364d19ea79f334e4a6e9734e562cbd02cbba09e3e0deec15436dfdc02687'
HISTORY_SHA = 'd8a533e2c3ccc20f49cc65b536ebd18e00e9a1f6d2aace406f83986bcb983348'
PANELS = ('reset-det','reset-stoch','outward-stoch')


def require(condition: bool, message: str) -> None:
    if not condition: raise ValueError(message)


def ordered_sum(values) -> float:
    total = 0.0
    for value in values: total += value
    return total


def verified_zip(raw: bytes, expected: str) -> dict[str, bytes]:
    require(hashlib.sha256(raw).hexdigest() == expected, 'outer digest')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)), 'duplicate archive member')
        manifest=json.loads(archive.read('manifest.json'))
        require(set(names)==set(manifest)|{'manifest.json'}, 'manifest membership')
        result={}
        for name,digest in manifest.items():
            path=PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in name, 'unsafe path')
            value=archive.read(name)
            require(hashlib.sha256(value).hexdigest()==digest, f'member digest: {name}')
            result[name]=value
    return result


def prepare_input(source: Path, prior: Path) -> dict:
    members=verified_zip(source.read_bytes(), SOURCE_SHA)
    history=verified_zip(members['historical-source.zip'], HISTORY_SHA)
    for name,raw in history.items():
        if name.startswith('reference/'):
            target=prior/'history'/name.removeprefix('reference/')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    for name,raw in members.items():
        if name.startswith('extra-fresh/') or name=='evaluation.csv':
            target=prior/'online'/name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    require(len((prior/'history/updates.csv').read_text().splitlines())==4225, 'history rows')
    require(len((prior/'online/extra-fresh/updates.csv').read_text().splitlines())==129, 'online rows')
    for cp in (0,1024,4096):
        for kind in ('actor','critic'): require((prior/f'history/{kind}-{cp}.bin').is_file(), 'missing historical checkpoint')
    for cp in (4097,4128,4137,4224):
        for kind in ('actor','critic'): require((prior/f'online/extra-fresh/{kind}-{cp}.bin').is_file(), 'missing online checkpoint')
    return {'artifact':10752868007,'sha256':SOURCE_SHA,'member_checks':len(members),'historical_member_checks':len(history)}


def select(curve: list[dict]) -> int | None:
    require([int(r['update']) for r in curve] == list(range(4128,4225)), 'window sequence')
    previous=None; best=None
    for row in curve:
        update=int(row['update']); value=float(row['nominal_det_mean'])
        require(math.isfinite(value), 'nonfinite score')
        drop=0.0 if previous is None else previous-value
        require(float(row['previous_minus_current']) == drop, 'drop calculation')
        if drop>0 and (best is None or drop>best[1]): best=(update,drop)
        previous=value
    return None if best is None else best[0]


def outcomes(path: Path, arms: tuple[str,...], checkpoints: dict[str,set[int]], reps: int) -> dict:
    with path.open() as stream: rows=list(csv.DictReader(stream))
    expected={(a,c,p,r) for a in arms for c in checkpoints[a] for p in PANELS for r in range(reps)}
    result={}
    for row in rows:
        k=(row['arm'],int(row['checkpoint']),row['panel'],int(row['rep']))
        require(k in expected and k not in result,'unexpected/duplicate case')
        require(int(row['seed'])==41004,'seed')
        cap=int(row['cap']); steps=int(row['steps']); require(cap==2048 and 0<steps<=cap,'horizon')
        for field in ('discounted','total','max_position','max_angle','force_rms'):
            row[field]=float(row[field]);require(math.isfinite(row[field]),'nonfinite result')
        f32=lambda v: struct.unpack('<f',struct.pack('<f',v))[0]
        position=f32(row['max_position'])>f32(2.4);angle=f32(row['max_angle'])>f32(0.6)
        ending='both' if position and angle else 'position' if position else 'angle' if angle else 'timeout'
        require(row['ending']==ending,'boundary classification')
        require(row['ending']!='timeout' or steps==cap,'false timeout')
        row['steps']=steps;row['key']=int(row['key']);result[k]=row
    require(set(result)==expected,'missing cases')
    return result


def contrast(a: list[dict], b: list[dict]) -> dict:
    require(len(a)==len(b) and len(a)>1,'pair count')
    require(all(x['key']==y['key'] for x,y in zip(a,b)),'paired keys')
    answer={}
    for name in ('discounted','total'):
        differences=[x[name]-y[name] for x,y in zip(a,b)]
        require(all(math.isfinite(v) for v in differences),'nonfinite contrast')
        mean=ordered_sum(differences)/len(differences)
        half=2.586*(statistics.stdev(differences)/math.sqrt(len(differences)))
        answer[name]={'mean':mean,'descriptive_99_interval':[mean-half,mean+half]}
    answer['gained']=sum(x['ending']=='timeout' and y['ending']!='timeout' for x,y in zip(a,b))
    answer['lost']=sum(x['ending']!='timeout' and y['ending']=='timeout' for x,y in zip(a,b))
    return answer


def summarize(root: Path) -> dict:
    complete=json.loads((root/'complete.json').read_text())
    require(complete['history_exact'] is True,'history gate')
    with (root/'curve.csv').open() as f: curve=list(csv.DictReader(f))
    target=select(curve);require(target==complete['selected_update'],'selection')
    window=outcomes(root/'window.csv',('extra-fresh',),{'extra-fresh':set(range(4128,4225))},64)
    require(sum(r['steps'] for r in window.values())==complete['window_steps'],'window cost')
    for row in curve:
        cp=int(row['update']);mean=ordered_sum(window['extra-fresh',cp,'reset-det',i]['discounted'] for i in range(64))/64
        require(mean==float(row['nominal_det_mean']),'curve from outcomes')
    report={'classification':'diagnostic_witness_not_training_qualification','selected_update':target,'complete':complete}
    if target is None: return report
    require(complete['sham_exact'] is True and complete['pair_records']==8192,'replay gate')
    arms=('incoming','original','baseline-only')
    cps={a:{target-1 if a=='incoming' else target} for a in arms}
    by=outcomes(root/'selected/evaluation.csv',arms,cps,512)
    require(sum(r['steps'] for r in by.values())==complete['confirmation_steps'],'confirmation cost')
    panels={};contrasts={}
    for p in PANELS:
        selected={a:[by[a,next(iter(cps[a])),p,i] for i in range(512)] for a in arms}
        panels[p]={a:{'completed':sum(r['ending']=='timeout' for r in rs),'n':512,'mean_discounted':ordered_sum(r['discounted'] for r in rs)/512,'mean_total':ordered_sum(r['total'] for r in rs)/512} for a,rs in selected.items()}
        contrasts[p]={a+'-minus-'+b:contrast(selected[a],selected[b]) for a,b in (('original','incoming'),('baseline-only','incoming'),('baseline-only','original'))}
    report['panels']=panels;report['contrasts']=contrasts
    report['note']='Selection is exposed; confirmation does not reselect. Intervals describe paired cases, not independent training seeds. Retained credit labels have a far bootstrap.'
    return report


def main() -> None:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('archive',type=Path);a.add_argument('prior',type=Path)
    b=sub.add_parser('summarize');b.add_argument('output',type=Path)
    args=p.parse_args()
    if args.command=='prepare': report=prepare_input(args.archive,args.prior)
    else:
        report=summarize(args.output)
        (args.output/'summary.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True,indent=2))

if __name__=='__main__': main()
