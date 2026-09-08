"""Prepare a disposable audit checkout; no production equations are changed."""
from pathlib import Path
import hashlib

def blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

trainer = Path('rust_robotics_train/src/trainer.rs')
audit = Path('rust_robotics_train/src/ppo_critic_audit.rs')
assert blob(trainer.read_bytes()) == '515bf5e3f6f2875234d7711200e6f7019aad5123'
assert blob(audit.read_bytes()) == '46342b0f788053ca6f2052cf8e4b441c8318b732'
text = audit.read_text()
# Two source-review corrections, made before any experiment execution: remove
# an accidental redundant write and use the iterator form in a test control.
old = '    write_f32(root.join("prefitted-critic.bin")), &[]);\n'
assert text.count(old) == 1
text = text.replace(old, '')
old = '    for i in 0..4 { for v in [-0.5_f32,0.0,0.5] {\n        let mut x=[0.0;4];x[i]=v;\n        assert!((p.act(x)-20.0*(-k[i]*v/20.0).tanh()).abs()<5e-5,"constructive teacher");'
new = '    for (i, gain) in k.iter().enumerate() { for v in [-0.5_f32,0.0,0.5] {\n        let mut x=[0.0;4];x[i]=v;\n        assert!((p.act(x)-20.0*(-gain*v/20.0).tanh()).abs()<5e-5,"constructive teacher");'
assert text.count(old) == 1
text = text.replace(old,new)
audit.write_text(text)
trainer.write_bytes(trainer.read_bytes()+b'\n#[cfg(test)]\n#[path = "ppo_critic_audit.rs"]\nmod critic_audit;\n')
