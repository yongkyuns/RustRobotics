#!/usr/bin/env python3
"""Frozen #33 inference-only projection, not a learned-policy acceptance gate."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def key(row):
    return (row['controller'], int(row['training_seed']), int(row['updates']), int(row['episode']))


def verify(directory, frozen):
    expected = {('ppo', s, 2048, i) for s in range(201,205) for i in range(32)}
    old = {key(r): r for r in rows(frozen/'episodes.tsv') if key(r) in expected}
    all_panels = {}
    summaries = []
    for mode in ('original','projected','projected_repeat'):
        panel = rows(directory/mode/'episodes.tsv')
        assert len(panel) == 128 and {key(r) for r in panel} == expected
        for r in panel:
            assert int(r['evaluation_seed']) == 0x33000000 + int(r['training_seed'])*1024 + int(r['episode'])
            assert r['initial_observation'] == old[key(r)]['initial_observation']
            assert 1 <= int(r['steps']) <= 1000
            assert r['initial_value'] == ('none' if mode != 'original' else old[key(r)]['initial_value'])
        all_panels[mode] = {key(r):r for r in panel}
        for seed in (None,201,202,203,204):
            group = [r for r in panel if seed is None or int(r['training_seed']) == seed]
            summaries.append({'controller':mode,'training_seed':seed,'episodes':len(group),
                'mean_return':statistics.mean(float(r['return']) for r in group),
                'mean_steps':statistics.mean(int(r['steps']) for r in group),
                'median_steps':statistics.median(int(r['steps']) for r in group),
                'endings':{e:sum(r['ending']==e for r in group) for e in ('Angle','Position','Both','Timeout')}})
    assert all_panels['original'] == old, 'ALL original final episode fields must replay'
    old_traces = [r for r in rows(frozen/'traces.tsv') if key(r) in expected]
    new_traces = [r for r in rows(directory/'original/traces.tsv') if int(r['episode']) < 2]
    assert new_traces == old_traces, 'all previously retained original traces must match'
    for name in ('episodes.tsv','traces.tsv'):
        assert (directory/'projected'/name).read_bytes() == (directory/'projected_repeat'/name).read_bytes(), 'projected same-build replay'
    return {'development_only':True,'baseline_matches_all_128_records':True,'projected_replay_exact':True,'groups':summaries}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    frozen=args.input.resolve()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report={'development_only':True,'commands':[],'completed':False}
    target=Path('rust_robotics_train/tests/ppo_symmetry_experiment.rs')
    original=Path('rust_robotics_train/tests/ppo_balancing_diagnostics.rs')
    extension=Path('scripts/ppo_symmetry_extension.rs')
    assert not target.exists(), 'do not overwrite an existing test'
    env={**os.environ,'PPO_SYMMETRY_INPUT':str(frozen),'PPO_SYMMETRY_OUTPUT':str(out),'CARGO_TERM_COLOR':'never'}

    def run(label,command,count=None,failure=None,marker=None):
        result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
        (out/(label+'.log')).write_text(result.stdout)
        print(result.stdout,flush=True)
        record={'label':label,'command':command,'returncode':result.returncode,'expected_count':count,'failure':failure,'marker':marker}
        report['commands'].append(record)
        if failure:
            assert result.returncode == 101 and 'running 1 test\n' in result.stdout, record
            assert re.search(r'^test '+re.escape(failure)+r' \.\.\. FAILED$',result.stdout,re.MULTILINE),record
            assert 'test result: FAILED. 0 passed; 1 failed; 0 ignored;' in result.stdout,record
            assert 'panicked at' in result.stdout and marker in result.stdout,record
        else:
            assert result.returncode == 0, record
            if count is not None:
                assert f'test result: ok. {count} passed; 0 failed; 0 ignored;' in result.stdout,record
        record['verified']=True

    controls=['cargo','test','--locked','-p','rust_robotics_train','--test','ppo_symmetry_experiment','--','--skip','measure_default_balancing_development_panel','--skip','measure_frozen_reflection_projection','--test-threads=1','--show-output']
    try:
        report['commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        report['rustc']=subprocess.check_output(['rustc','-Vv'],text=True)
        for name,sha in {'snapshots.tsv':'3db6e300f195e81db4eef98472260055fbf104ec85d44bd2ecc9e9f68916777a','episodes.tsv':'ca906062d58c687474a310687db5b214229ab0805afa581adfcf50a564f59675','traces.tsv':'3f95d334cabf68fe1f40dda4ddc1fff57ff3468aca703c782583eb9a5071cd63','config.txt':'9afb5c511a23339dfde7bef121df5f33ccb82d974d0cc87e8bc2402a681ec03f'}.items():
            assert digest(frozen/name)==sha,(name,'wrong frozen input')
        provenance=json.loads((frozen/'results.json').read_text())
        report['source_sha256']={}
        for name,sha in provenance['source_sha256'].items():
            archived=frozen/'sources'/name
            assert digest(archived)==sha,name
            if name not in ('scripts/diagnose_ppo_balancing.py','rust_robotics_train/tests/ppo_balancing_diagnostics.rs'):
                assert digest(Path(name))==sha,('production/lockfile changed',name)
            current=Path(name)
            target_copy=out/'sources'/name
            target_copy.parent.mkdir(parents=True,exist_ok=True)
            target_copy.write_bytes(current.read_bytes())
            report['source_sha256'][name]=digest(current)
        data=original.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()=='db3d8b8c1b892c33829e17654c4388780d39b55d'
        shutil.copytree(frozen,out/'frozen-input')
        target.write_bytes(data+b'\n'+extension.read_bytes())
        run('format',['rustfmt','--edition','2021',str(target)])
        fixed=target.read_bytes()
        (out/'ppo_symmetry_experiment.rs').write_bytes(fixed)
        run('projection-controls',controls,14)
        test='projection_reflects_all_coordinates_and_averages_force'
        mutations=[('missing-coordinate-reflection','let mirrored = policy(o.map(|x| -x));','let mirrored = policy([-o[0], o[1], o[2], o[3]]);','exact observation reflection and two deterministic calls'),('wrong-projection-sign','let projected = 0.5 * (forward - mirrored);','let projected = 0.5 * (forward + mirrored);','force-space difference and half scaling'),('wrong-projection-scale','let projected = 0.5 * (forward - mirrored);','let projected = 0.25 * (forward - mirrored);','force-space difference and half scaling')]
        for label,old,new,marker in mutations:
            text=fixed.decode()
            assert text.count(old)==1,label
            try:
                target.write_text(text.replace(old,new))
                run(label,['cargo','test','--locked','-p','rust_robotics_train','--test','ppo_symmetry_experiment',test,'--','--exact','--test-threads=1','--show-output'],1,test,marker)
            finally:
                target.write_bytes(fixed)
        run('restored-controls',controls,14)
        run('unchanged-unit-tests',['cargo','test','--locked','-p','rust_robotics_train','--lib'],55)
        run('strict-clippy',['cargo','clippy','--locked','-p','rust_robotics_train','--all-targets','--','-D','warnings'])
        run('format-check',['cargo','fmt','--all','--','--check'])
        run('measurements',['cargo','test','--locked','--release','-p','rust_robotics_train','--test','ppo_symmetry_experiment','measure_frozen_reflection_projection','--','--ignored','--exact','--test-threads=1','--show-output'],1)
        report['summary']=verify(out,frozen)
        assert target.read_bytes()==fixed
        report['completed']=True
    except Exception as exc:
        report['error']=repr(exc)
        raise
    finally:
        if target.exists():
            (out/'final-generated-test.rs').write_bytes(target.read_bytes())
            target.unlink()
        for source in (Path(__file__),extension,Path('.github/workflows/ppo-symmetry-experiment.yml')):
            if source.exists():shutil.copy2(source,out/source.name)
        status=subprocess.run(['git','diff','--exit-code'],text=True,capture_output=True)
        report['restored_clean']=status.returncode==0
        report['restored_diff']=status.stdout
        report['file_sha256']={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p!=out/'results.json'}
        (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        assert report['restored_clean'],'tracked files not restored'


if __name__=='__main__':
    main()
