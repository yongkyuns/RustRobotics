#!/usr/bin/env python3
"""Compile original coordinator sources and a one-guard candidate as test modules.

The only additions to original sources are cfg(test) seeded constructors/readouts.
Production coordinator/native files are not edited.
"""
from pathlib import Path
import difflib
import hashlib
import os

out=Path(os.environ['PPO_COORD_OUT']); out.mkdir(parents=True,exist_ok=True)
base=Path('rust_robotics_sim/src/simulator/ppo_trainer')
mod=(base/'mod.rs').read_text(); native=(base/'native.rs').read_text()
def blob(s):
    b=s.encode(); return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
assert blob(mod)=='7ac4aa49ebe15ea5a29cd79fd2865f059ac62f0d'
assert blob(native)=='52cf70d26f95623449c78ff7e612d601bbfb10af'
anchor='        if let Some(shared_state) = average_shared_states(\n'
assert mod.count(anchor)==1
fixed=mod.replace(anchor,'        if self.executors.len() == 1 {\n            self.refresh_summary();\n            return;\n        }\n\n'+anchor)
(out/'single-replica-guard.patch').write_text(''.join(difflib.unified_diff(mod.splitlines(True),fixed.splitlines(True),
    fromfile='a/'+str(base/'mod.rs'),tofile='b/'+str(base/'mod.rs'))))
helper='''
#[cfg(test)]
impl PpoTrainerCoordinator {
    pub fn audit_seeded(config: PpoTrainerConfig, seed: u64) -> Self {
        let mut result = Self {
            executors: vec![PlatformPpoReplicaExecutor::audit_seeded(config, seed)],
            ..Self::default()
        };
        result.refresh_summary();
        result
    }
    pub fn audit_shared(&self) -> PpoSharedState {
        self.executors[0].shared_state().unwrap()
    }
}
#[cfg(test)]
pub fn audit_average(states: &[PpoSharedState]) -> PpoSharedState {
    average_shared_states(states).unwrap()
}
'''
native_helper='''
#[cfg(test)]
impl NativePpoReplicaExecutor {
    pub fn audit_seeded(config: PpoTrainerConfig, seed: u64) -> Self {
        Self { session: PpoTrainerSession::new_seeded(config, seed) }
    }
}
'''
for kind,s in [('original',mod),('guard',fixed)]:
    p=out/'sources'/kind; p.mkdir(parents=True,exist_ok=True)
    (p/'mod.rs').write_text(s+helper); (p/'native.rs').write_text(native+native_helper)
    (p/'original-mod.rs').write_text(mod); (p/'original-native.rs').write_text(native)
header=''
for kind in ['original','guard']:
    header+=f'#[allow(dead_code)]\n#[path = "{(out/"sources"/kind/"mod.rs").resolve()}"]\nmod {kind};\n'
test=r'''
use rust_robotics_train::{PpoTrainerConfig, PpoTrainerSession, PpoSharedState, LinearSnapshot, PolicySnapshot, ValueSnapshot};
fn identical(a: &PpoSharedState, b: &PpoSharedState) -> bool { a.policy==b.policy && a.value==b.value }
fn max_actor_delta(a: &PpoSharedState,b: &PpoSharedState)->f32 {
    let mut d=0.0_f32;
    for (x,y) in [(&a.policy.input,&b.policy.input),(&a.policy.hidden,&b.policy.hidden),(&a.policy.output,&b.policy.output)] {
        for (u,v) in x.weight.iter().chain(&x.bias).zip(y.weight.iter().chain(&y.bias)) { d=d.max((u-v).abs()); }
    }
    d
}
#[test]
fn current_single_replica_exactly_matches_resetting_adam_not_persistent_training() {
    let config=PpoTrainerConfig::default(); let seed=201;
    let mut coordinator=original::PpoTrainerCoordinator::audit_seeded(config.clone(),seed);
    let mut persistent=PpoTrainerSession::new_seeded(config.clone(),seed);
    let mut self_reload=PpoTrainerSession::new_seeded(config,seed);
    assert!(identical(&coordinator.audit_shared(),&persistent.shared_state()));
    let mut report=String::from("update\tmax_actor_difference_from_persistent\tself_reload_exact\n");
    for update in 1..=4 {
        coordinator.tick(1); persistent.train_updates(1); self_reload.train_updates(1);
        self_reload.load_shared_state(&self_reload.shared_state());
        let got=coordinator.audit_shared();
        assert!(identical(&got,&self_reload.shared_state()),"actual coordinator must match explicit self-reload");
        assert_eq!(identical(&got,&persistent.shared_state()),update==1);
        report.push_str(&format!("{update}\t{}\ttrue\n",max_actor_delta(&got,&persistent.shared_state())));
    }
    println!("COORDINATOR REPRODUCTION\n{report}");
    std::fs::write(std::path::Path::new(&std::env::var("PPO_COORD_OUT").unwrap()).join("optimizer-reset.tsv"),report).unwrap();
}
#[test]
fn one_replica_guard_restores_persistent_adam_and_grouping_invariance() {
    let config=PpoTrainerConfig::default();
    let mut split=guard::PpoTrainerCoordinator::audit_seeded(config.clone(),201);
    let mut grouped=guard::PpoTrainerCoordinator::audit_seeded(config.clone(),201);
    let mut direct=PpoTrainerSession::new_seeded(config.clone(),201);
    let mut original_split=original::PpoTrainerCoordinator::audit_seeded(config.clone(),201);
    let mut original_grouped=original::PpoTrainerCoordinator::audit_seeded(config,201);
    for _ in 0..4 { split.tick(1); original_split.tick(1); }
    grouped.tick(4); direct.train_updates(4); original_grouped.tick(4);
    assert!(identical(&split.audit_shared(),&direct.shared_state()));
    assert!(identical(&grouped.audit_shared(),&direct.shared_state()));
    assert!(identical(&original_grouped.audit_shared(),&direct.shared_state()));
    assert!(!identical(&original_split.audit_shared(),&original_grouped.audit_shared()));
    println!("GUARD restores exact seeded four-update grouping invariance");
}
fn linear(ni:usize,no:usize,w:Vec<f32>)->LinearSnapshot { LinearSnapshot { in_dim:ni,out_dim:no,weight:w,bias:vec![0.0;no] } }
#[test]
fn averaging_equivalent_permuted_networks_does_not_preserve_the_policy() {
    let mut inputs=vec![0.0;8]; inputs[0]=1.0; inputs[1]=-1.0;
    let p=PolicySnapshot { input:linear(4,2,inputs),hidden:linear(2,2,vec![1.0,0.0,0.0,1.0]),
        output:linear(2,1,vec![1.0,-1.0]),action_limit:20.0,action_std:2.0 };
    let value=ValueSnapshot { input:linear(4,2,vec![0.0;8]),hidden:linear(2,2,vec![0.0;4]),output:linear(2,1,vec![0.0;2]) };
    let a=PpoSharedState { policy:p.clone(),value }; let mut b=a.clone();
    b.policy.input.weight[0]=-1.0; b.policy.input.weight[1]=1.0;
    b.policy.output.weight=vec![-1.0,1.0];
    let averaged=original::audit_average(&[a.clone(),b.clone()]);
    for x in [-1.0,-0.25,0.25,1.0] {
        let obs=[x,0.0,0.0,0.0];
        assert_eq!(a.policy.act(obs),b.policy.act(obs));
        assert_eq!(averaged.policy.act(obs),0.0);
        assert_ne!(averaged.policy.act(obs),a.policy.act(obs));
    }
    println!("PARAMETER AVERAGING COUNTEREXAMPLE: equivalent actors at x=1 command {}, average commands 0",p.act([1.0,0.0,0.0,0.0]));
}
'''
p=Path('rust_robotics_train/tests/ppo_coordinator_reference.rs'); p.write_text(header+test)
(out/'sources'/'ppo_coordinator_reference.rs').write_text(header+test)
print('Prepared exact original coordinator and one-guard candidate; no production source edited.')
