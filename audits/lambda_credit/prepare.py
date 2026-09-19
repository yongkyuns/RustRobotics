"""Disposable build. No production algorithm is replaced or retuned."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,zipfile
BASE='4739f370558b9443708c920ac30614f86e3c07bb'
OUT=Path(os.environ['LAMBDA_BUILD']);OUT.mkdir(parents=True,exist_ok=True)
ROOT=Path('audits/lambda_credit')
PROTECTED=['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','docs','Cargo.lock']
subprocess.run(['git','diff','--exit-code',BASE,'--',*PROTECTED],check=True)

def unpack(path,digest,destination):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path
    with zipfile.ZipFile(path) as z:
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
        for n,h in json.loads(z.read('manifest.json')).items():
            assert hashlib.sha256(z.read(n)).hexdigest()==h,n
        z.extractall(destination)

unpack(Path('/tmp/witness.zip'),'9b29ad8d9fc34f6c6eecc6141bae66ca31acd9eaaf5863a2d5e0459414d6b295',OUT/'witness')
unpack(Path('/tmp/long-build.zip'),'ab471447c13df0ca7db3ecc01e09cb3ee705cd8e8b1c759512adde4a0dc25af6',OUT/'previous-build')
prior=OUT/'previous-build'
for old,new in [('original/reference.py','reference.py'),('robustness','robustness'),('robustness.rs','robustness.rs'),('run.py','previous_run.py')]:
    shutil.copyfile(prior/old,OUT/new)
(OUT/'robustness').chmod(0o755)
# The only executable change in the witness probe is lambda for its measured
# Trial. Controls keep their own explicit lambda values and all original tests.
probe=(OUT/'witness/probe-executed.rs').read_text()
anchor='lambda: f64::from(0.95_f32),'
assert probe.count(anchor)==2
probe=probe.replace(anchor,'lambda: 1.0,',1)
(ROOT/'probe.rs').write_text(probe)
bridge=(prior/'original/bridge.rs').read_text()
addition='''
#[no_mangle]
pub extern "C" fn rr_trainer_create_lambda(seed: u64, mode: u32) -> u64 {
    if mode > 1 { return 0; }
    let mut config=PpoTrainerConfig::default();
    if mode == 1 { config.ppo.gae_lambda=1.0; }
    TRAINERS.with(|v| {
        let mut v=v.borrow_mut();
        v.push(Some(PpoTrainerSession::new_seeded(config,seed)));
        v.len() as u64
    })
}
#[no_mangle]
pub extern "C" fn rr_trainer_lambda(handle: u64) -> f32 {
    TRAINERS.with(|v| match v.borrow().get(handle.wrapping_sub(1) as usize) {
        Some(Some(t)) => t.config().ppo.gae_lambda,
        _ => f32::NAN,
    })
}
'''
anchor='#[cfg(test)]\nmod tests {'
assert bridge.count(anchor)==1
(ROOT/'bridge.rs').write_text(bridge.replace(anchor,addition+'\n'+anchor))
p=Path('rust_robotics_train/Cargo.toml');assert '[lib]' not in p.read_text()
p.write_text(p.read_text()+'\n[lib]\ncrate-type = ["rlib", "cdylib"]\n\n[[bin]]\nname = "lambda-credit-probe"\npath = "../audits/lambda_credit/probe.rs"\n')
p=Path('rust_robotics_train/src/lib.rs')
p.write_text(p.read_text()+'\n#[path = "../../audits/lambda_credit/bridge.rs"]\nmod lambda_bridge;\n')
p=Path('rust_robotics_train/src/trainer.rs');original=p.read_text()
p.write_text(original+'\n#[cfg(test)]\n#[path = "../../audits/lambda_credit/native_tests.rs"]\nmod lambda_tests;\n')
subprocess.run(['cargo','fmt','--all'],check=True)
assert p.read_text().startswith(original)
subprocess.run(['git','diff','--check'],check=True)
changed=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changed)==sorted(['rust_robotics_train/Cargo.toml','rust_robotics_train/src/lib.rs','rust_robotics_train/src/trainer.rs']),changed
# Keep the validated prior raw archives separate, not silently repackaged.
shutil.copyfile('/tmp/witness.zip',OUT/'witness-original.zip')
shutil.copyfile('/tmp/long-build.zip',OUT/'long-build-original.zip')
for path in list(ROOT.glob('*'))+[Path('Cargo.lock'),Path('Cargo.toml'),Path('AGENTS.md')]+list(Path('rust_robotics_train').rglob('*.rs'))+[Path('rust_robotics_train/Cargo.toml')]:
    if path.is_file():
        dest=OUT/'sources'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
shutil.copyfile('.github/workflows/ppo-lambda-credit.yml',OUT/'workflow.yml')
(OUT/'patch.diff').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
(OUT/'baseline.txt').write_text(BASE+'\n')
(OUT/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
print('LAMBDA PREPARATION PASS',flush=True)
