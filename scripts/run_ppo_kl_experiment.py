#!/usr/bin/env python3
"""Temporary fixed-protocol KL experiment; no production commit or default change."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BASE = 'd724c517c27b182145d4054a66d3937563381dfd'
TRAIN = Path('rust_robotics_train/src/trainer.rs')
MODULE = Path('rust_robotics_train/src/ppo_kl_experiment.rs')
EVALUATOR = Path('rust_robotics_train/src/ppo_kl_evaluation.rs')
PREFIX = 'trainer::kl_experiment::'

def change(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)

def git_blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def prepare():
    assert git_blob(TRAIN.read_bytes()) == '515bf5e3f6f2875234d7711200e6f7019aad5123'
    s = TRAIN.read_text()
    s = change(s, '    fn optimize_with_rng<R: Rng + ?Sized>(&mut self, rollout: &RolloutBatch, rng: &mut R) {', '''    fn optimize_with_rng<R: Rng + ?Sized>(&mut self, rollout: &RolloutBatch, rng: &mut R) {
        #[cfg(test)]
        let mut kl_probe = kl_experiment::begin(self, rollout);''')
    s = change(s, '                let actor_loss = if self.config.ppo.entropy_coef == 0.0 {', '''                #[cfg(test)]
                let allow_actor = kl_experiment::actor_allowed(&kl_probe);
                #[cfg(not(test))]
                let allow_actor = true;
                if allow_actor {
                let actor_loss = if self.config.ppo.entropy_coef == 0.0 {''')
    s = change(s, '                let actor_grads = GradientsParams::from_grads(actor_loss.backward(), &self.actor);', '''                let raw_actor_grads = actor_loss.backward();
                #[cfg(test)]
                kl_experiment::actor_gradient(&mut kl_probe, self, &raw_actor_grads);
                let actor_grads = GradientsParams::from_grads(raw_actor_grads, &self.actor);''')
    s = change(s, '''                    actor_grads,
                );

                let values''', '''                    actor_grads,
                );
                }
                #[cfg(test)]
                kl_experiment::after_actor(&mut kl_probe, self, allow_actor);

                let values''')
    s = change(s, '''                    let critic_grads =
                        GradientsParams::from_grads(objective.backward(), &self.critic);''', '''                    let raw_critic_grads = objective.backward();
                    #[cfg(test)]
                    kl_experiment::critic_gradient(&mut kl_probe, self, &raw_critic_grads);
                    let critic_grads = GradientsParams::from_grads(raw_critic_grads, &self.critic);''')
    s = change(s, '                last_policy_loss = policy_loss_scalar;', '''                #[cfg(test)]
                kl_experiment::after_critic(&mut kl_probe, self, self.config.ppo.value_loss_coef > 0.0);
                last_policy_loss = policy_loss_scalar;''')
    s = change(s, '        self.metrics.last_policy_loss = last_policy_loss;', '''        #[cfg(test)]
        kl_experiment::finish(kl_probe);
        self.metrics.last_policy_loss = last_policy_loss;''')
    s += '\n#[cfg(test)]\n#[path = "ppo_kl_experiment.rs"]\nmod kl_experiment;\n'
    TRAIN.write_text(s)
    source = Path('rust_robotics_train/tests/ppo_balancing_diagnostics.rs').read_bytes()
    assert git_blob(source) == 'db3d8b8c1b892c33829e17654c4388780d39b55d'
    evaluation = source.decode().replace('use rust_robotics_train::{', 'use crate::{')
    evaluation += r'''

// Reuse the unchanged independent episode evaluator, snapshot encoder and tests.
// This only exposes checkpoint output to the private experimental test module.
pub(super) fn checkpoint(session: &PpoTrainerSession, seed: u64, updates: usize, directory: &Path) -> io::Result<()> {
    if updates == 0 {
        output(directory,"episodes.tsv","controller\ttraining_seed\tupdates\tepisode\tevaluation_seed\treturn\tdiscounted_return\tsteps\tending\tabs_force_sum\tnear_limit_steps\tinitial_value\tinitial_observation\tfinal_state\tmax_abs_state")?.flush()?;
        output(directory,"traces.tsv","controller\ttraining_seed\tupdates\tepisode\tevaluation_seed\tstep\tbefore_state\tobservation\taction\treward\tafter_state\tdone\ttruncated")?.flush()?;
        output(directory,"snapshots.tsv","training_seed\tupdates\tkind\tf32_hex")?.flush()?;
        output(directory,"metrics.tsv","training_seed\tupdates\tenv_steps\tepisodes\tmean_return\tpolicy_loss\tvalue_loss")?.flush()?;
    }
    let append = |name: &str| -> io::Result<BufWriter<fs::File>> {
        Ok(BufWriter::new(fs::OpenOptions::new().append(true).open(directory.join(name))?))
    };
    let mut scores=append("episodes.tsv")?;
    let mut traces=append("traces.tsv")?;
    let mut snapshots=append("snapshots.tsv")?;
    let mut metrics=append("metrics.tsv")?;
    let m=session.metrics();
    assert_eq!(m.total_updates,updates);assert_eq!(m.total_env_steps,updates*512);
    writeln!(metrics,"{seed}\t{updates}\t{}\t{}\t{}\t{}\t{}",m.total_env_steps,m.total_episodes,m.mean_episode_return,m.last_policy_loss,m.last_value_loss)?;
    let shared=session.shared_state();check_snapshot(&shared);
    emit_snapshot(&mut snapshots,seed,updates,&shared)?;
    for index in 0..EPISODES {
        let key=format!("ppo\t{seed}\t{updates}\t{index}");
        let e_seed=episode_seed(seed,index);
        let result=episode(evaluation_config(),e_seed,|o|shared.policy.act(o),Some(&shared.value),&mut traces,(index<2).then_some(key.as_str()))?;
        emit_episode(&mut scores,&key,e_seed,&result)?;
    }
    scores.flush()?;traces.flush()?;snapshots.flush()?;metrics.flush()?;
    Ok(())
}

pub(super) fn require_prior(seed: u64, measured: &Path, prior: &Path) {
    for name in ["episodes.tsv","traces.tsv","snapshots.tsv","metrics.tsv"] {
        let original=fs::read_to_string(prior.join(name)).unwrap();
        let observed=fs::read_to_string(measured.join(name)).unwrap();
        let prefix=if name=="episodes.tsv" || name=="traces.tsv" { format!("ppo\t{seed}\t") } else { format!("{seed}\t") };
        let expected:Vec<_>=original.lines().filter(|s|s.starts_with(&prefix)).collect();
        let actual:Vec<_>=observed.lines().skip(1).collect();
        assert!(!expected.is_empty(),"prior baseline must contain this seed");
        assert_eq!(actual,expected,"monitored baseline must reproduce all prior rows in {name}");
    }
}
'''
    # Python raw literals intentionally preserve Rust backslash-t escapes.
    evaluation = evaluation.replace('\\\\t','\\t')
    EVALUATOR.write_text(evaluation)
    m=MODULE.read_text()
    m=change(m, '        let directory=root.join(arm);fs::create_dir_all(&directory).unwrap();', '''        let directory=root.join(arm);fs::create_dir_all(&directory).unwrap();
        if arm == "kl001" {
            let prior=PathBuf::from(std::env::var_os("PPO_KL_PRIOR").unwrap());
            evaluation::require_prior(seed,&root.join("baseline"),&prior);
        }''')
    MODULE.write_text(m)
    subprocess.run(['cargo','fmt','--all'],check=True)

def execute(label, args, output, report, count=None, failure=None, marker=None):
    result=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=2100)
    (output/(label+'.log')).write_text(result.stdout)
    print(result.stdout,flush=True)
    item={'label':label,'command':args,'returncode':result.returncode,'count':count,'failure':failure,'marker':marker}
    report.append(item)
    (output/'commands.json').write_text(json.dumps(report,indent=2)+'\n')
    if failure:
        assert result.returncode==101 and 'running 1 test\n' in result.stdout,item
        assert re.search(r'^test '+re.escape(PREFIX+failure)+r' \.\.\. FAILED$',result.stdout,re.M),item
        assert 'test result: FAILED. 0 passed; 1 failed; 0 ignored;' in result.stdout,item
        assert 'panicked at' in result.stdout and marker in result.stdout,item
    else:
        assert result.returncode==0,item
        if count is not None:
            assert f'test result: ok. {count} passed; 0 failed;' in result.stdout,item
    item['validated']=True
    (output/'commands.json').write_text(json.dumps(report,indent=2)+'\n')

def qualify(output):
    output.mkdir(parents=True,exist_ok=True);report=[]
    execute('strict-clippy',['cargo','clippy','--locked','-p','rust_robotics_train','--all-targets','--','-D','warnings'],output,report)
    execute('unit-baseline',['cargo','test','--locked','-p','rust_robotics_train','--lib','--','--test-threads=1'],output,report,70)
    mutants=[
        ('wrong-kl-scale',MODULE,'let k = 0.5 * d * d;','let k = d * d;','exact_kl_matches_known_gaussian_reference','analytic Gaussian KL scale'),
        ('bypassed-actor-stop',MODULE,'probe.as_ref().is_none_or(|p| !p.rule.stopped)','probe.as_ref().is_none_or(|_p| true)','stopped_actor_does_not_stop_critic_updates','actor must stop after crossing'),
        ('stopping-critic-too',TRAIN,'if self.config.ppo.value_loss_coef > 0.0 {','if allow_actor && self.config.ppo.value_loss_coef > 0.0 {','stopped_actor_does_not_stop_critic_updates','critic steps must match full reference'),
        ('nonsticky-stop',MODULE,'self.stopped |= self.target.is_some_and(|t| kl > t);','self.stopped = self.target.is_some_and(|t| kl > t);','stop_latch_is_sticky_and_new_rollout_resets_it','stop must remain latched'),
    ]
    for label,path,old,new,test,marker in mutants:
        fixed=path.read_bytes()
        try:
            path.write_text(change(fixed.decode(),old,new))
            args=['cargo','test','--locked','-p','rust_robotics_train','--lib',PREFIX+test,'--','--exact','--test-threads=1','--show-output']
            execute(label,args,output,report,failure=test,marker=marker)
        finally:
            path.write_bytes(fixed)
    execute('unit-restored',['cargo','test','--locked','-p','rust_robotics_train','--lib','--','--test-threads=1'],output,report,70)
    execute('existing-integration-controls',['cargo','test','--locked','-p','rust_robotics_train','--tests'],output,report,70)
    execute('format',['cargo','fmt','--all','--','--check'],output,report)
    execute('diff-check',['git','diff','--check'],output,report)

def preserve(output):
    source_root=output/'sources';source_root.mkdir(parents=True,exist_ok=True)
    paths=[TRAIN,MODULE,EVALUATOR,Path('Cargo.lock'),Path('scripts/run_ppo_kl_experiment.py'),Path('.github/workflows/ppo-kl-experiment.yml')]
    paths += list(Path('rust_robotics_train/src').glob('*.rs'))
    paths += [Path('rust_robotics_core/src/lib.rs'),Path('rust_robotics_algo/src/control/inverted_pendulum/mod.rs'),Path('rust_robotics_train/tests/ppo_balancing_diagnostics.rs')]
    for path in set(paths):
        if path.exists():
            dest=source_root/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    (output/'instrumentation.patch').write_text(subprocess.check_output(['git','diff','--',str(TRAIN)],text=True))
    (output/'manifest.json').write_text(json.dumps({str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')

if __name__=='__main__':
    mode=sys.argv[1];out=Path(os.environ['PPO_KL_DIR'])
    if mode=='prepare': prepare()
    elif mode=='qualify': qualify(out)
    elif mode=='preserve': preserve(out)
    elif mode=='measure':
        report=[]
        execute('paired-measurement',['cargo','test','--locked','--release','-p','rust_robotics_train','--lib',PREFIX+'measure_paired_default_kl_stopping','--','--ignored','--exact','--test-threads=1','--show-output'],out,report,1)
    else: raise ValueError(mode)
