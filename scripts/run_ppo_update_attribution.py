#!/usr/bin/env python3
"""Temporary audit driver. Does not replace any training equation."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(os.environ['PPO_ATTR_DIR'])
ROOT.mkdir(parents=True, exist_ok=True)
TRAIN = Path('rust_robotics_train/src/trainer.rs')
ENV = Path('rust_robotics_train/src/env.rs')
MODULE = Path('rust_robotics_train/src/ppo_update_attribution.rs')

def blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)

def run(label, args, passed=None):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=2400)
    (ROOT/(label+'.log')).write_text(result.stdout)
    print(result.stdout, flush=True)
    assert result.returncode == 0, (label,result.returncode)
    if passed is not None:
        assert f'test result: ok. {passed} passed; 0 failed;' in result.stdout, label

if sys.argv[1] == 'prepare':
    assert blob(TRAIN.read_bytes()) == '515bf5e3f6f2875234d7711200e6f7019aad5123'
    assert blob(ENV.read_bytes()) == 'c87d55f9affc0685e24dc3c6fc77718da4f0c4e5'
    s=TRAIN.read_text()
    s=once(s,'            let step = self\n                .env\n                .step_with_rng(sample.action, &mut self.environment_rng);',
        '            #[cfg(test)]\n            update_attribution::before(&self.env, observation, sample.latent, value);\n'
        '            let step = self\n                .env\n                .step_with_rng(sample.action, &mut self.environment_rng);\n'
        '            #[cfg(test)]\n            update_attribution::after(&step);')
    s+='\n#[cfg(test)]\n#[path = "ppo_update_attribution.rs"]\nmod update_attribution;\n'
    TRAIN.write_text(s)
    # Test-only cloning resets only the external clock, never physical state.
    helper='''#[cfg(test)]
impl PendulumEnv {
    pub(crate) fn attribution_branch(&self, max_steps: usize) -> Self {
        assert!(max_steps > 0);
        let mut branch = self.clone();
        branch.steps = 0;
        branch.config.max_steps = max_steps;
        branch
    }
    pub(crate) fn attribution_age(&self) -> usize { self.steps }
}

'''
    s=once(ENV.read_text(),'#[cfg(test)]\nmod tests {',helper+'#[cfg(test)]\nmod tests {')
    ENV.write_text(s)
    s=MODULE.read_text()
    # Source-review parenthesis repair, before first execution; no numeric change.
    s=once(s,'unwrap())))}else{None};','unwrap()))}else{None};')
    s=once(s,'    fs::write(root().join("config.txt"),format!("{:?}\\n",s.config())).unwrap();',
        '    for kind in ["actor", "critic"] {\n'
        '        assert_eq!(fs::read(root().join(format!("{kind}-512.bin"))).unwrap(), fs::read(root().join(format!("expected-{kind}-512.bin"))).unwrap(), "exact prior baseline weights at update512");\n'
        '    }\n'
        '    fs::write(root().join("config.txt"),format!("{:?}\\n",s.config())).unwrap();')
    MODULE.write_text(s)
    run('format', ['cargo','fmt','--all'])
    provenance=ROOT/'sources';provenance.mkdir(exist_ok=True)
    for p in [TRAIN,ENV,MODULE,Path(__file__),Path('Cargo.lock'),Path('rust_robotics_train/src/model.rs'),Path('rust_robotics_train/src/ppo_distribution.rs'),Path('rust_robotics_train/src/algorithm.rs'),Path('.github/workflows/ppo-update-attribution.yml')]:
        target=provenance/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    (ROOT/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
    (ROOT/'rustc.txt').write_text(subprocess.check_output(['rustc','-Vv'],text=True))
    prior=os.environ.get('PPO_ATTR_PRIOR')
    if prior:
        seed=os.environ['PPO_ATTR_SEED']
        rows=list(csv.DictReader(Path(prior).open(),delimiter='\t'))
        for kind in ['actor','critic']:
            found=[r for r in rows if r['training_seed']==seed and r['updates']=='512' and r['kind']==kind]
            assert len(found)==1
            text=found[0]['f32_hex'];words=[text[i:i+8] for i in range(0,len(text),8)]
            if kind=='actor':
                assert words[:2]==['41a00000','40000000'];words=words[2:]
            assert len(words)==4545
            (ROOT/f'expected-{kind}-512.bin').write_bytes(b''.join(int(w,16).to_bytes(4,'little') for w in words))
        (ROOT/'prior-snapshots.tsv').write_bytes(Path(prior).read_bytes())
elif sys.argv[1]=='controls':
    run('clippy',['cargo','clippy','--locked','-p','rust_robotics_train','--all-targets','--','-D','warnings'])
    run('unit-tests',['cargo','test','--locked','-p','rust_robotics_train','--lib'],60)
    run('format-check',['cargo','fmt','--all','--','--check'])
    run('diff-check',['git','diff','--check'])
elif sys.argv[1]=='measure':
    run('measurement',['cargo','test','--locked','--release','-p','rust_robotics_train','--lib','trainer::update_attribution::attribute_from_scratch_update','--','--ignored','--exact','--test-threads=1','--show-output'],1)
    assert 'ATTRIBUTION COMPLETE' in (ROOT/'measurement.log').read_text()
elif sys.argv[1]=='preserve':
    (ROOT/'source-diff.patch').write_text(subprocess.check_output(['git','diff'],text=True))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (ROOT/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
else:
    raise ValueError('prepare, controls, measure or preserve')
