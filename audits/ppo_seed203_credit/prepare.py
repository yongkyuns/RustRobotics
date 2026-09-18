"""Disposable build preparation. Only a child ABI and cdylib target are added."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import zipfile

BASE='d3f59f9bf38d4038ba7c5008e3a3b41c15e99835'
ARTIFACTS={
 'critic-native':(10566149387,'3c3976e2704cecf0b6f716babf8ee08fced49acf748193f6f9c6e03e1f728a68'),
 'critic-baseline-203':(10566074594,'c04b9646f4b54b367737aef3566475b95c7b9cdbe2bc898c16633574106443ce'),
 'critic-extra-frozen-203':(10565249386,'9921dc56da26efede1e7b9d0075b2d945fb29902c90795aee928d36812336f47'),
 'critic-extra-refresh-203':(10565354343,'83a7db66c03d75a355eabf1508951fe3ee74ffd4fe3272202c7588faf1ec4b63'),
}
PROTECTED=['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
out=Path(os.environ['CREDIT_BUILD']);out.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','diff','--exit-code',BASE,'--',*PROTECTED],check=True)
(out/'raw-inputs').mkdir(exist_ok=True)
provenance={}
for name,(artifact,digest) in ARTIFACTS.items():
    path=out/'raw-inputs'/f'{name}.zip'
    with path.open('wb') as f:
        subprocess.run(['gh','api',f'repos/yongkyuns/RustRobotics/actions/artifacts/{artifact}/zip'],stdout=f,check=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    target=out/'inputs'/name;target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
        manifest=json.loads(z.read('manifest.json'))
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,(name,n)
        z.extractall(target)
    provenance[name]={'artifact':artifact,'sha256':digest,'verified_members':len(manifest)}
(out/'input-artifacts.json').write_text(json.dumps(provenance,indent=2)+'\n')
# The env body is byte-identical; the child can restore diagnostic physical
# states but calls the exact ordinary step_with_rng for every transition.
p=Path('rust_robotics_train/src/env.rs');original=p.read_text()
p.write_text(original+'\n#[path = "../../audits/ppo_seed203_credit/native.rs"]\nmod seed203_credit;\n')
p=Path('rust_robotics_train/Cargo.toml');text=p.read_text();assert '[lib]' not in text
p.write_text(text+'\n[lib]\ncrate-type = ["rlib", "cdylib"]\n')
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
assert Path('rust_robotics_train/src/env.rs').read_text().startswith(original)
changed=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changed)==['rust_robotics_train/Cargo.toml','rust_robotics_train/src/env.rs'],changed
paths=list(Path('audits/ppo_seed203_credit').glob('*'))+[Path('Cargo.toml'),Path('Cargo.lock'),Path('AGENTS.md')]
for crate in ['rust_robotics_train','rust_robotics_core','rust_robotics_algo']:
    paths+=list(Path(crate).rglob('*.rs'))+list(Path(crate).rglob('Cargo.toml'))
for path in paths:
    if path.is_file():
        dest=out/'sources'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
shutil.copyfile('.github/workflows/ppo-seed203-credit.yml',out/'workflow-source.yml')
(out/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(out/'baseline.txt').write_text(BASE+'\n')
(out/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
(out/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
print('CREDIT PREPARATION PASS: every original input hash verified; no production algorithm edits',flush=True)
