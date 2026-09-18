"""Prepare disposable native build; original production algorithms are retained."""
from pathlib import Path
import hashlib,io,json,os,re,shutil,subprocess,zipfile
BASE='d3f59f9bf38d4038ba7c5008e3a3b41c15e99835'
SHA='555d222dfbcaa79ee60b0b2c5740dad5e1f9836439f4e36eda368faa87cdc03a'
OUT=Path(os.environ['PPO_CRITIC_OUT']);OUT.mkdir(parents=True,exist_ok=True)
ROOT=Path('audits/ppo_critic_cadence')
PROTECTED=['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
subprocess.run(['git','diff','--exit-code',BASE,'--',*PROTECTED],check=True)
raw=Path(os.environ['PPO_CRITIC_PRIOR']).read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    manifest=json.loads(z.read('manifest.json'))
    for n,h in manifest.items(): assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    bridge=z.read('sources/audits/ppo_recovery_batches/bridge.rs').decode()
    reference=z.read('sources/audits/ppo_recovery_batches/reference.py')
    evaluator=z.read('sources/audits/ppo_recovery_batches/measure.py').decode()
(OUT/'prior-native.zip').write_bytes(raw)
bridge,n=re.subn(r'\n#\[no_mangle\]\npub extern "C" fn rr_trainer_create_batch\(.*?\n}\n','\n',bridge,flags=re.S)
assert n==1
addition='''
thread_local! {
    static EXTRA: RefCell<std::collections::BTreeMap<u64,(u32,StdRng)>> = const { RefCell::new(std::collections::BTreeMap::new()) };
}
#[no_mangle]
pub extern "C" fn rr_trainer_create_critic(seed: u64, mode: u32) -> u64 {
    if mode > 2 { return 0; }
    let handle=rr_trainer_create(seed);
    EXTRA.with(|m| m.borrow_mut().insert(handle,(mode,StdRng::seed_from_u64(seed ^ 0x4352_4954_4943_0001))));
    handle
}
#[no_mangle]
pub extern "C" fn rr_trainer_update_critic(handle: u64, updates: u32) -> CMetrics {
    EXTRA.with(|extra| {
        let mut extra=extra.borrow_mut();
        let Some((mode,rng))=extra.get_mut(&handle) else {return CMetrics{status:1,..Default::default()};};
        TRAINERS.with(|v| {
            let mut v=v.borrow_mut();
            let Some(Some(t))=v.get_mut(handle.wrapping_sub(1) as usize) else {return CMetrics{status:1,..Default::default()};};
            for _ in 0..updates { t.audit_critic_update(*mode,rng); }
            let m=t.metrics();
            CMetrics{updates:m.total_updates as u64,steps:m.total_env_steps as u64,episodes:m.total_episodes as u64,
                policy_loss:m.last_policy_loss,value_loss:m.last_value_loss,status:0}
        })
    })
}
#[no_mangle]
pub extern "C" fn rr_trainer_free_critic(handle: u64) -> u32 {
    EXTRA.with(|m| m.borrow_mut().remove(&handle));
    rr_trainer_free(handle)
}
'''
anchor='#[cfg(test)]\nmod tests {';assert bridge.count(anchor)==1
(ROOT/'bridge.rs').write_text(bridge.replace(anchor,addition+'\n'+anchor))
(ROOT/'reference.py').write_bytes(reference)
evaluator=evaluator.replace("NATIVE/'sources/audits/ppo_recovery_batches/reference.py'","NATIVE/'sources/audits/ppo_critic_cadence/reference.py'")
evaluator=evaluator.replace("NATIVE=Path(os.environ['PPO_BATCH_NATIVE'])","NATIVE=Path(os.environ['PPO_CRITIC_NATIVE'])")
evaluator=evaluator.replace('r.LIB.rr_trainer_create_batch.argtypes=[C.c_uint64,C.c_uint32];r.LIB.rr_trainer_create_batch.restype=C.c_uint64','')
(ROOT/'evaluation_wrapper.py').write_text(evaluator)
p=Path('rust_robotics_train/Cargo.toml');assert '[lib]' not in p.read_text();p.write_text(p.read_text()+'\n[lib]\ncrate-type = ["rlib", "cdylib"]\n')
p=Path('rust_robotics_train/src/lib.rs');p.write_text(p.read_text()+'\n#[path = "../../audits/ppo_critic_cadence/bridge.rs"]\nmod critic_bridge;\n')
p=Path('rust_robotics_train/src/trainer.rs');s=p.read_text()
a='                last_value_loss = value_loss_scalar;'
hook='                critic_audit::ordinary_minibatch(self, chunk);\n'
assert s.count(a)==1;s=s.replace(a,hook+a)
p.write_text(s+'\n#[path = "../../audits/ppo_critic_cadence/native.rs"]\nmod critic_audit;\n')
p=Path('rust_robotics_train/src/ppo_rollout_pool.rs');s=p.read_text()
a='            observations.push(observation);'
b='''            critic_audit::observe(critic_audit::Row {
                reward: step.reward, value, terminal: step.terminated(),
                path_end: step.done || index + 1 == rollout_steps,
                final_observation: step.observation, bootstrap: 0.0,
            });
            observations.push(observation);'''
assert s.count(a)==1;s=s.replace(a,b)
a='                let (path_returns, path_advantages) = compute_gae('
b='                critic_audit::bootstrap(bootstrap_value);\n'+a
assert s.count(a)==1;p.write_text(s.replace(a,b))
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changed=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
expected=['rust_robotics_train/Cargo.toml','rust_robotics_train/src/lib.rs','rust_robotics_train/src/trainer.rs','rust_robotics_train/src/ppo_rollout_pool.rs']
assert sorted(changed)==sorted(expected),changed
for n in ['trainer.rs','lib.rs']:
    p=Path('rust_robotics_train/src')/n
    original=subprocess.check_output(['git','show',BASE+':'+str(p)]).decode()
    assert p.read_text().replace(hook,'').startswith(original),n
files=list(ROOT.glob('*'))+[Path('Cargo.lock'),Path('Cargo.toml'),Path('AGENTS.md')]
for crate in ['rust_robotics_train','rust_robotics_core','rust_robotics_algo']:
    files+=list(Path(crate).rglob('*.rs'))+list(Path(crate).rglob('Cargo.toml'))
for p in files:
    if p.is_file():
        dst=OUT/'sources'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
shutil.copyfile('.github/workflows/ppo-critic-cadence.yml',OUT/'workflow-source.yml')
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'baseline.txt').write_text(BASE+'\n');(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
(OUT/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
print('CRITIC PREPARATION PASS: source protected, exact old evidence verified')
