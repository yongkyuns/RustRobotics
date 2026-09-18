"""Prepare an isolated checkout, reusing the verified real Rust ABI/evaluator."""
from pathlib import Path
import os,hashlib,zipfile,json,io,shutil,subprocess
BASE='d3f59f9bf38d4038ba7c5008e3a3b41c15e99835'
SHA='17ebe5551c5cb1d610580775d3a9831423e9070f6de4257f5dc2c8135b983cd4'
OUT=Path(os.environ['PPO_BATCH_OUT']);OUT.mkdir(exist_ok=True,parents=True)
ROOT=Path('audits/ppo_recovery_batches')
PROTECTED=['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
subprocess.run(['git','diff','--exit-code',BASE,'--',*PROTECTED],check=True)
raw=Path(os.environ['PPO_BATCH_PRIOR']).read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    manifest=json.loads(z.read('manifest.json'))
    for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    original=z.read('c3-native-original.zip')
    reference=z.read('sources/audits/ppo_epsilon/c3_reference.py')
assert hashlib.sha256(original).hexdigest()=='615ac5ef521e7d28b3f8e3c19b4b5070949d79489b1ea256bcea7c7265dff100'
with zipfile.ZipFile(io.BytesIO(original)) as z:
    m=json.loads(z.read('manifest.json'))
    for n,h in m.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    bridge=z.read('sources/audits/ppo_crosscheck/bridge.rs').decode()
addition='''
#[no_mangle]
pub extern "C" fn rr_trainer_create_batch(seed: u64, arm: u32) -> u64 {
    if arm > 2 { return 0; }
    TRAINERS.with(|v| {
        let mut v=v.borrow_mut();
        v.push(Some(PpoTrainerSession::audit_new_batch(seed,arm)));
        v.len() as u64
    })
}
'''
anchor='#[cfg(test)]\nmod tests {';assert bridge.count(anchor)==1
(ROOT/'bridge.rs').write_text(bridge.replace(anchor,addition+'\n'+anchor))
(ROOT/'reference.py').write_bytes(reference)
p=Path('rust_robotics_train/Cargo.toml');assert '[lib]' not in p.read_text();p.write_text(p.read_text()+'\n[lib]\ncrate-type = ["rlib", "cdylib"]\n')
p=Path('rust_robotics_train/src/lib.rs');p.write_text(p.read_text()+'\n#[path = "../../audits/ppo_recovery_batches/bridge.rs"]\nmod batch_bridge;\n')
p=Path('rust_robotics_train/src/trainer.rs');s=p.read_text()
a='        for _ in 0..self.config.ppo.epochs_per_update {\n            indices.shuffle(rng);'
b='        batch_audit::record_epoch(self, rollout, 0);\n        for epoch in 0..self.config.ppo.epochs_per_update {\n            indices.shuffle(rng);'
assert s.count(a)==1;s=s.replace(a,b)
a='                last_value_loss = value_loss_scalar;\n            }\n        }'
b='                last_value_loss = value_loss_scalar;\n            }\n            batch_audit::record_epoch(self, rollout, epoch + 1);\n        }'
assert s.count(a)==1;s=s.replace(a,b)
s+='\n#[path = "../../audits/ppo_recovery_batches/native.rs"]\nmod batch_audit;\n';p.write_text(s)
subprocess.run(['cargo','fmt','--all'],check=True);subprocess.run(['git','diff','--check'],check=True)
changes=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changes)==sorted(['rust_robotics_train/Cargo.toml','rust_robotics_train/src/lib.rs','rust_robotics_train/src/trainer.rs']),changes
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'baseline.txt').write_text(BASE+'\n');(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
(OUT/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
files=list(ROOT.glob('*'))+[Path('Cargo.lock'),Path('Cargo.toml'),Path('AGENTS.md')]
for crate in ['rust_robotics_train','rust_robotics_core','rust_robotics_algo']:
    files+=list(Path(crate).rglob('*.rs'))+list(Path(crate).rglob('Cargo.toml'))
for p in files:
    if p.is_file():
        dst=OUT/'sources'/p;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(p,dst)
shutil.copyfile('.github/workflows/ppo-recovery-batches.yml',OUT/'workflow-source.yml')
print('BATCH PREPARATION PASS: verified inputs; only audit constructors and observation hooks')
