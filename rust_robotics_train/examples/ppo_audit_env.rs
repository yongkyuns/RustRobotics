//! Temporary audit transport: the learning/evaluation plant is the REAL Rust env.
//! No policy, gradients, reward changes, or Python reimplementation lives here.
use rand::{rngs::StdRng, SeedableRng};
use rust_robotics_train::{PendulumEnv, PendulumEnvConfig};
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[1].parse().expect("environment count");
    let cap: usize = args[2].parse().expect("time limit");
    assert!(n > 0 && n <= 64 && cap > 0);
    let config = PendulumEnvConfig { max_steps: cap, ..Default::default() };
    let mut rngs: Vec<_> = (0..n).map(|i| StdRng::seed_from_u64(i as u64)).collect();
    let mut envs: Vec<_> = rngs.iter_mut().map(|r| PendulumEnv::new_with_rng(Default::default(), config, r)).collect();
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out, "READY {n} {cap}")?;
    out.flush()?;
    for line in stdin.lock().lines() {
        let line = line?;
        let tokens: Vec<_> = line.split_whitespace().collect();
        if tokens.first() == Some(&"q") { break; }
        assert_eq!(tokens.len(), n + 1, "one value per environment required");
        match tokens[0] {
            "r" => {
                write!(out, "R")?;
                for i in 0..n {
                    rngs[i] = StdRng::seed_from_u64(tokens[i+1].parse().expect("u64 seed"));
                    envs[i] = PendulumEnv::new_with_rng(Default::default(), config, &mut rngs[i]);
                    let obs = envs[i].observation_with_rng(&mut rngs[i]);
                    for value in obs { write!(out, " {value}")?; }
                    for value in envs[i].state().iter() { write!(out, " {value}")?; }
                }
                writeln!(out)?;
            }
            "s" => {
                write!(out, "S")?;
                for i in 0..n {
                    let action: f32 = tokens[i+1].parse().expect("finite force");
                    assert!(action.is_finite());
                    let step = envs[i].step_with_rng(action, &mut rngs[i]);
                    let physical = envs[i].state();
                    let obs = if step.done { envs[i].reset_with_rng(&mut rngs[i]) } else { step.observation };
                    for value in obs { write!(out, " {value}")?; }
                    write!(out, " {} {} {}", step.reward, u8::from(step.done), u8::from(step.truncated))?;
                    for value in step.observation { write!(out, " {value}")?; }
                    for value in physical.iter() { write!(out, " {value}")?; }
                }
                writeln!(out)?;
            }
            _ => panic!("unknown audit transport command"),
        }
        out.flush()?;
    }
    Ok(())
}
