#!/usr/bin/env python3
"""Prepare private diagnostic hooks around the unchanged native PPO optimizer."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,zipfile
BASE='4739f370558b9443708c920ac30614f86e3c07bb'
DIGEST='ab471447c13df0ca7db3ecc01e09cb3ee705cd8e8b1c759512adde4a0dc25af6'
OUT=Path(os.environ['DECAY_BUILD']);OUT.mkdir(parents=True,exist_ok=True)
ROOT=Path('audits/late_actor_decay')
PROTECTED=['rust_robotics_algo','rust_robotics_core','rust_robotics_train','rust_robotics_sim','docs','Cargo.lock']
subprocess.run(['git','diff','--exit-code',BASE,'--',*PROTECTED],check=True)
archive=Path(os.environ['DECAY_PRIOR_BUILD']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST
prior=OUT/'prior-build'
with zipfile.ZipFile(archive) as z:
    assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
    for n,h in json.loads(z.read('manifest.json')).items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    z.extractall(prior)
shutil.copyfile(archive,OUT/'prior-build.zip')
for old,new in [('original/reference.py','reference.py'),('robustness','robustness'),('robustness.rs','robustness.rs')]:
    shutil.copyfile(prior/old,OUT/new)
(OUT/'robustness').chmod(0o755)
bridge=(prior/'original/bridge.rs').read_text()
addition='''
thread_local! {
    static MODES: RefCell<std::collections::BTreeMap<u64,u32>> = const { RefCell::new(std::collections::BTreeMap::new()) };
}
#[no_mangle]
pub extern "C" fn rr_trainer_create_decay(seed: u64, mode: u32) -> u64 {
    if mode > 1 { return 0; }
    let handle=rr_trainer_create(seed);
    MODES.with(|m| m.borrow_mut().insert(handle,mode));
    handle
}
#[no_mangle]
pub extern "C" fn rr_trainer_update_decay(handle: u64, updates: u32) -> CMetrics {
    let mode=MODES.with(|m| m.borrow().get(&handle).copied());
    let Some(mode)=mode else { return CMetrics{status:1,..Default::default()}; };
    TRAINERS.with(|v| {
        let mut v=v.borrow_mut();
        let Some(Some(t))=v.get_mut(handle.wrapping_sub(1) as usize) else {return CMetrics{status:1,..Default::default()};};
        for _ in 0..updates {t.audit_decay_update(mode);}
        let m=t.metrics();
        CMetrics{updates:m.total_updates as u64,steps:m.total_env_steps as u64,episodes:m.total_episodes as u64,
            policy_loss:m.last_policy_loss,value_loss:m.last_value_loss,status:0}
    })
}
#[no_mangle]
pub extern "C" fn rr_trainer_free_decay(handle: u64) -> u32 {
    MODES.with(|m| m.borrow_mut().remove(&handle));
    rr_trainer_free(handle)
}
'''
marker='#[cfg(test)]\nmod tests {';assert bridge.count(marker)==1
(ROOT/'bridge.rs').write_text(bridge.replace(marker,addition+'\n'+marker))
p=Path('rust_robotics_train/Cargo.toml');original=p.read_text();assert '[lib]' not in original
p.write_text(original+'\n[lib]\ncrate-type = ["rlib", "cdylib"]\n')
p=Path('rust_robotics_train/src/lib.rs');p.write_text(p.read_text()+'\n#[path = "../../audits/late_actor_decay/bridge.rs"]\nmod decay_bridge;\n')
p=Path('rust_robotics_train/src/trainer.rs');original=p.read_text()
old='''                self.actor = self.actor_optimizer.step(
                    self.config.ppo.learning_rate,'''
new='''                self.actor = self.actor_optimizer.step(
                    decay_audit::actor_rate(self.config.ppo.learning_rate),'''
assert original.count(old)==1
s=original.replace(old,new)
anchor='                last_value_loss = value_loss_scalar;'
hook='                decay_audit::record_indices(self, chunk);\n'
assert s.count(anchor)==1;s=s.replace(anchor,hook+anchor)
s+='\n#[path = "../../audits/late_actor_decay/native.rs"]\nmod decay_audit;\n'
p.write_text(s)
assert s.replace(new,old).replace(hook,'').startswith(original)
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changed=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changed)==sorted(['rust_robotics_train/Cargo.toml','rust_robotics_train/src/lib.rs','rust_robotics_train/src/trainer.rs']),changed
for path in [Path('Cargo.lock'),Path('Cargo.toml'),Path('AGENTS.md')]+list(ROOT.glob('*'))+list(Path('rust_robotics_train').rglob('*.rs'))+[Path('rust_robotics_train/Cargo.toml')]:
    if path.is_file():
        dest=OUT/'sources'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
shutil.copyfile(ROOT/'run.py',OUT/'run.py')
shutil.copyfile('.github/workflows/ppo-late-actor-decay.yml',OUT/'workflow.yml')
(OUT/'patch.diff').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'audit-commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
(OUT/'baseline.txt').write_text(BASE+'\n')
(OUT/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
print('ACTOR DECAY PREPARATION PASS',flush=True)
