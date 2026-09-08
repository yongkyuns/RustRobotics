#!/usr/bin/env python3
"""Apply only observational hooks in the disposable Actions checkout."""
from pathlib import Path
import hashlib

p = Path('rust_robotics_train/src/trainer.rs')
data = p.read_bytes()
assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == '515bf5e3f6f2875234d7711200e6f7019aad5123'
s = data.decode()
def change(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
change('            for chunk in indices.chunks(batch_size) {', '            for chunk in indices.chunks(batch_size) {\n                #[cfg(test)]\n                reference_audit::before(self, chunk);')
change('                    let noise = scalar_tensor::<AutodiffBackend>(&self.device, &noise);', '                    #[cfg(test)]\n                    reference_audit::entropy_noise(&noise);\n                    let noise = scalar_tensor::<AutodiffBackend>(&self.device, &noise);')
change('                let actor_grads = GradientsParams::from_grads(actor_loss.backward(), &self.actor);', '''                #[cfg(test)]
                let audit_total = tensor_scalar(&actor_loss);
                let raw_gradients = actor_loss.backward();
                #[cfg(test)]
                reference_audit::actor(self, &raw_gradients, policy_loss_scalar, audit_total);
                let actor_grads = GradientsParams::from_grads(raw_gradients, &self.actor);''')
change('''                    let critic_grads =
                        GradientsParams::from_grads(objective.backward(), &self.critic);''', '''                    #[cfg(test)]
                    let audit_total = tensor_scalar(&objective);
                    let raw_gradients = objective.backward();
                    #[cfg(test)]
                    reference_audit::critic(self, &raw_gradients, value_loss_scalar, audit_total);
                    let critic_grads = GradientsParams::from_grads(raw_gradients, &self.critic);''')
change('                last_policy_loss = policy_loss_scalar;', '                #[cfg(test)]\n                reference_audit::after(self);\n                last_policy_loss = policy_loss_scalar;')
s += '\n#[cfg(test)]\n#[path = "ppo_reference_audit.rs"]\nmod reference_audit;\n'
p.write_text(s)
p = Path('rust_robotics_train/src/ppo_reference_audit.rs')
s = p.read_text().replace('use burn::tensor::{backend::AutodiffBackend as AD, Tensor};', 'use burn::tensor::backend::AutodiffBackend as AD;').replace('use rust_robotics_core::{LinearSnapshot, ValueSnapshot};', 'use rust_robotics_core::LinearSnapshot;')
# Strict-Clippy setup repair: enumerate the same four coefficients, unchanged order/math.
assert s.count('for i in 0..4 {') == 1
s = s.replace('for i in 0..4 {', 'for (i, coefficient) in gain.into_iter().enumerate() {')
s = s.replace('-gain[i]/20.0', '-coefficient/20.0').replace('gain[i]/20.0', 'coefficient/20.0')
p.write_text(s)
