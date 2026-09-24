#!/usr/bin/env python3
"""Read-only cohort accounting and loss-exhaustive native replay requests.

No learning, checkpoint choice, or intervention tuning. All replay requests bind
to already-recorded evaluation cases; none is a held-out qualification sample.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

SEEDS = (41001, 41002, 41004, 41006)
ARMS = ('extra-ordinary', 'extra-transient')
SHORT = ('reset-det', 'reset-stoch', 'outward-stoch', 'stress-det', 'stress-stoch')
LONG = ('long-det', 'long-stoch', 'outward-long')
CPS = (0, 1024, 4096, 4608)
SOURCE = 'c634e052e19dd48b1c72627f54e4a9a14ad36cf1'
FLOATS = ('total', 'discounted', 'max_position', 'max_angle', 'force_rms')
KEY_FIELDS = ('seed', 'arm', 'checkpoint', 'panel', 'rep')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def case_key(row: dict) -> tuple:
    return (int(row['seed']), row['arm'], int(row['checkpoint']), row['panel'], int(row['rep']))


def case_id(key: tuple) -> str:
    return '_'.join(map(str, key))


def f32(x: str | float) -> float:
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def manifest(root: Path) -> int:
    entries = json.loads((root / 'manifest.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(entries) | {'manifest.json'}, f'manifest membership: {root}')
    for name, digest in entries.items():
        p = root / name
        require(p.resolve().is_relative_to(root.resolve()), 'manifest path escape')
        require(hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'payload hash: {p}')
    return len(entries)


def expected_keys(seed: int, arm: str) -> set[tuple]:
    return {(seed, arm, cp, panel, rep)
            for cp in CPS for panel in SHORT + (LONG if cp == 4608 else ())
            for rep in range(32 if panel in LONG else 256 if cp == 4608 else 64)}


def validate_cases(items: list[dict], seed: int, arm: str) -> dict:
    records = {}
    expected = expected_keys(seed, arm)
    for r in items:
        key = case_key(r)
        require(key not in records, f'duplicate case: {key}')
        require(key in expected, f'unexpected case: {key}')
        cp, panel, rep = key[2:]
        cap = 30000 if panel in LONG[:2] else 6000 if panel == 'outward-long' else 2048
        require(int(r['cap']) == cap and 0 < int(r['steps']) <= cap, 'invalid duration')
        require(r['ending'] in ('timeout', 'angle', 'position', 'both'), 'invalid ending')
        require(r['ending'] != 'timeout' or int(r['steps']) == cap, 'false completion')
        require(all(math.isfinite(float(r[n])) for n in FLOATS), 'nonfinite outcome')
        pos, ang = f32(r['max_position']) > f32(2.4), f32(r['max_angle']) > f32(.6)
        ending = 'both' if pos and ang else 'position' if pos else 'angle' if ang else 'timeout'
        require(r['ending'] == ending, 'boundary mismatch')
        records[key] = r
    require(set(records) == expected, 'incomplete case set')
    return records


def read_cohort(root: Path) -> tuple[dict, dict]:
    build = root / 'reset-coverage-build'
    members = {'build': manifest(build)}
    cases = {}
    for seed in SEEDS:
        prior = None
        for arm in ARMS:
            name = f'reset-coverage-run-{seed}-{arm}'
            path = root / name
            members[name] = manifest(path)
            require((path/'build-commit.txt').read_text().strip() == SOURCE, 'wrong executable source')
            require((path/'build-manifest.json').read_bytes() == (build/'manifest.json').read_bytes(), 'wrong build manifest')
            current = validate_cases(load_csv(path/'evaluation.csv'), seed, arm)
            cases.update(current)
            if prior:
                for kind in ('actor', 'critic'):
                    require((path/f'{kind}-0.bin').read_bytes() == (prior/f'{kind}-0.bin').read_bytes(), 'different random initialization')
                require((path/'update-1/original-union.json').read_bytes() == (prior/'update-1/original-union.json').read_bytes(), 'different first original batch')
            prior = path
    require(len(cases) == 18688, 'incomplete cohort')
    return cases, members


def paired(before: dict, after: dict) -> tuple[bool, bool]:
    require(before['key'] == after['key'], 'evaluation random key changed')
    require(before['panel'] == after['panel'] and before['rep'] == after['rep'] and before['seed'] == after['seed'], 'case identity changed')
    return before['ending'] != 'timeout' and after['ending'] == 'timeout', before['ending'] == 'timeout' and after['ending'] != 'timeout'


def interval(values: list[float]) -> dict:
    require(len(values) == 4, 'training-seed interval requires four pairs')
    mean = statistics.mean(values)
    half = 5.840909309733352 * statistics.stdev(values) / math.sqrt(len(values))
    return {'mean': mean, 'interval_99': [mean-half, mean+half], 'n': 4}


def analyze(cases: dict) -> tuple[dict, dict]:
    require(set(cases) == set().union(*(expected_keys(s,a) for s in SEEDS for a in ARMS)), 'incomplete cohort keys')
    reasons: dict[tuple, set[str]] = {}
    def request(key: tuple, reason: str) -> None:
        require(key in cases, 'request is not an existing measurement')
        reasons.setdefault(key, set()).add(reason)
    final, final_losses, late, late_losses = {}, [], {}, []
    for seed in SEEDS:
        final[str(seed)] = {}
        for panel in SHORT + LONG:
            n = 32 if panel in LONG else 256
            pairs = [(cases[seed, ARMS[0], 4608, panel, i], cases[seed, ARMS[1], 4608, panel, i]) for i in range(n)]
            statuses = [paired(a,b) for a,b in pairs]
            gains = sum(g for g,_ in statuses); losses = sum(l for _,l in statuses)
            final[str(seed)][panel] = {
                'n': n, 'ordinary': sum(a['ending']=='timeout' for a,b in pairs),
                'transient': sum(b['ending']=='timeout' for a,b in pairs),
                'gained': gains, 'lost': losses, 'net': gains-losses,
                'mean_discounted_difference': statistics.mean(float(b['discounted'])-float(a['discounted']) for a,b in pairs),
                'mean_total_difference': statistics.mean(float(b['total'])-float(a['total']) for a,b in pairs)}
            for (a,b), (_,lost) in zip(pairs,statuses):
                if lost:
                    final_losses.append({'before': a, 'after': b})
                    request(case_key(a), 'final-arm-loss-control')
                    request(case_key(b), 'final-arm-loss-treatment')
            for arm in ARMS:
                request((seed, arm, 4608, panel, 0), 'existing-full-trace-control')
        for arm in ARMS:
            late[f'{seed}-{arm}'] = {}
            for panel in SHORT:
                pairs = [(cases[seed,arm,4096,panel,i], cases[seed,arm,4608,panel,i]) for i in range(64)]
                statuses = [paired(a,b) for a,b in pairs]
                late[f'{seed}-{arm}'][panel] = {
                    'n':64,'before':sum(a['ending']=='timeout' for a,b in pairs),
                    'after':sum(b['ending']=='timeout' for a,b in pairs),
                    'gained':sum(g for g,_ in statuses),'lost':sum(l for _,l in statuses)}
                for (a,b),(_,lost) in zip(pairs,statuses):
                    if lost:
                        late_losses.append({'before':a,'after':b})
                        request(case_key(a),'late-loss-before')
                        request(case_key(b),'late-loss-after')
    effects = {}
    for panel in SHORT + LONG:
        effects[panel] = {'completion_rate':interval([final[str(s)][panel]['net']/final[str(s)][panel]['n'] for s in SEEDS])}
        for field,name in [('mean_discounted_difference','discounted'),('mean_total_difference','total')]:
            effects[panel][name] = interval([final[str(s)][panel][field] for s in SEEDS])
    endings = {name:sum(x['after']['ending']==name for x in final_losses) for name in ('position','angle','both')}
    plan = {case_id(k): {'expected':cases[k], 'reasons':sorted(v)} for k,v in sorted(reasons.items())}
    result = {'source':SOURCE, 'training_seeds':4, 'runs':8, 'evaluation_rows':len(cases),
              'final':final,'late_same_64':late,'final_losses':final_losses,'late_losses':late_losses,
              'paired_training_seed_effects':effects,'final_loss_endings':endings,
              'final_lost_cases':len(final_losses),'late_lost_cases':len(late_losses),
              'requests':len(plan),'replay_cap_budget':sum(int(v['expected']['cap']) for v in plan.values()),
              'note':'Post-hoc diagnostic. All requested cases already exposed and recorded. No training or held-out qualification.'}
    return result, plan


def compare_original(result: dict, original: dict) -> None:
    for s in SEEDS:
        for p in SHORT + LONG:
            ours, theirs = result['final'][str(s)][p], original['per_seed'][str(s)][p]
            for field in ('gained','lost','net'):
                require(ours[field] == theirs[field], f'original case contrast {s}/{p}/{field}')
    for p in SHORT + LONG:
        for metric in ('completion_rate','discounted','total'):
            a=result['paired_training_seed_effects'][p][metric]
            b=original['paired_training_seed_effects'][p][metric]
            require(math.isclose(a['mean'],b['mean'],abs_tol=1e-10,rel_tol=1e-12),'original cohort mean')
            require(all(math.isclose(x,y,abs_tol=1e-10,rel_tol=1e-12) for x,y in zip(a['interval_99'],b['interval_99'])),'original cohort interval')


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def prepare(root: Path, out: Path) -> dict:
    cases,members=read_cohort(root)
    result,plan=analyze(cases)
    original=json.loads((root/'reset-coverage-comparison/cohort-result.json').read_text())
    compare_original(result,original)
    result['verified_members']=members
    result['registered_screen']=original['local_component_screen']
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'cohort.json',result);write_json(out/'requests.json',plan)
    with (out/'requests.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(KEY_FIELDS)
        w.writerows(case_key(v['expected']) for v in plan.values())
    return result


def verify_replay(root: Path, out: Path) -> dict:
    plan=json.loads((out/'requests.json').read_text())
    records=load_csv(out/'replayed.csv')
    actual={case_id(case_key(r)):r for r in records}
    require(len(actual)==len(records) and set(actual)==set(plan),'replay request coverage')
    traces=[];controls=0;steps=0
    for identity,item in plan.items():
        row=actual[identity];expected=item['expected']
        require(row.keys()==expected.keys(),'replay schema')
        for name in row:
            if name in FLOATS:
                require(float(row[name])==float(expected[name]),f'native outcome mismatch {identity}/{name}')
            else:
                require(row[name]==expected[name],f'native outcome mismatch {identity}/{name}')
        path=out/f'trace-{identity}.csv'
        if 'existing-full-trace-control' in item['reasons']:
            prior=root/f"reset-coverage-run-{row['seed']}-{row['arm']}"/f"trace-4608-{row['panel']}.csv"
            require(path.read_bytes()==prior.read_bytes(),f'existing full trace mismatch {identity}')
            controls+=1
        trace=load_csv(path);require(len(trace)==int(row['steps']),'trace length')
        total=discounted=force2=0.;power=1.;maxp=maxa=0.;maxv=maxw=0.
        previous=None
        for i,t in enumerate(trace):
            require(int(t['t'])==i,'trace time order')
            require(all(math.isfinite(float(v)) for k,v in t.items() if k not in ('terminal','truncated')),'nonfinite trace')
            if previous:
                for a,b in [('x','nx'),('v','nv'),('theta','ntheta'),('omega','nomega'),('o0','no0'),('o1','no1'),('o2','no2'),('o3','no3')]:
                    require(f32(t[a])==f32(previous[b]),'trace state/observation continuity')
            require(i==len(trace)-1 or (t['terminal']=='false' and t['truncated']=='false'),'early trace end')
            reward=f32(t['reward']);total+=reward;discounted+=power*reward;power*=f32(.99)
            force2+=f32(t['applied'])**2
            maxp=max(maxp,abs(f32(t['x'])),abs(f32(t['nx'])))
            maxa=max(maxa,abs(f32(t['theta'])),abs(f32(t['ntheta'])))
            maxv=max(maxv,abs(f32(t['v'])),abs(f32(t['nv'])))
            maxw=max(maxw,abs(f32(t['omega'])),abs(f32(t['nomega'])))
            previous=t
        require(total==float(row['total']) and discounted==float(row['discounted']),'trace reward reconstruction')
        require(maxp==f32(row['max_position']) and maxa==f32(row['max_angle']),'trace maxima')
        require(math.isclose(math.sqrt(force2/len(trace)),float(row['force_rms']),rel_tol=1e-13,abs_tol=1e-13),'trace applied force')
        require((trace[-1]['terminal']=='false')==(row['ending']=='timeout'),'trace final terminal')
        steps+=len(trace)
        traces.append({'case':identity,'reasons':item['reasons'],'steps':len(trace),'ending':row['ending'],
                       'max_abs_velocity':maxv,'max_abs_angular_velocity':maxw,
                       'initial':[f32(trace[0][n]) for n in ('x','v','theta','omega')],
                       'final':[f32(trace[-1][n]) for n in ('nx','nv','ntheta','nomega')]})
    complete=json.loads((out/'native-complete.json').read_text())
    require(complete=={'cases':len(plan),'simulation_steps':steps,'training_updates':0},'native completion receipt')
    result={'all_native_outcomes_exact':True,'all_existing_control_traces_byte_identical':True,
            'existing_trace_controls':controls,'cases':len(plan),'simulation_steps':steps,'training_updates':0,'traces':traces,
            'scope':'Exact reconstruction of exposed outcomes, not new learning evidence or independent dynamics verification.'}
    write_json(out/'replay-verification.json',result)
    return result


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','verify'));p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();result=prepare(args.root,args.out) if args.mode=='prepare' else verify_replay(args.root,args.out)
    print(json.dumps({k:v for k,v in result.items() if k not in ('final','late_same_64','final_losses','late_losses','traces','paired_training_seed_effects')},indent=2,sort_keys=True))

if __name__=='__main__':main()
