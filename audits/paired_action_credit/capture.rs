//! Observation-only, thread-local capture. Disabled outside explicit audit scope.
use super::RolloutBatch;
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sample {
    pub(crate) state: [f32; 4],
    pub(crate) observation: [f32; 4],
    pub(crate) latent: f32,
}
pub(crate) struct Captured {
    pub(crate) samples: Vec<Sample>,
    pub(super) batch: Option<RolloutBatch>,
}
thread_local! {
    static ACTIVE: RefCell<Option<Captured>> = const { RefCell::new(None) };
}
pub(crate) struct Scope;
impl Scope {
    pub(crate) fn enter() -> Self {
        ACTIVE.with(|s| {
            assert!(s.borrow().is_none(), "nested fitting capture");
            *s.borrow_mut() = Some(Captured { samples: Vec::new(), batch: None });
        });
        Self
    }
    pub(crate) fn finish(self) -> Captured {
        let result = ACTIVE.with(|s| s.borrow_mut().take().expect("capture missing"));
        let batch = result.batch.as_ref().expect("fitting batch missing");
        assert_eq!(result.samples.len(), batch.observations.len());
        for (i, sample) in result.samples.iter().enumerate() {
            assert_eq!(sample.observation, batch.observations[i]);
            assert_eq!(sample.latent, batch.latent_actions[i]);
        }
        result
    }
}
impl Drop for Scope {
    fn drop(&mut self) { ACTIVE.with(|s| { s.borrow_mut().take(); }); }
}
pub(crate) fn primary(rows: impl Iterator<Item = Sample>) {
    ACTIVE.with(|s| {
        if let Some(c) = s.borrow_mut().as_mut() {
            assert!(c.samples.is_empty());
            c.samples.extend(rows);
            assert_eq!(c.samples.len(), 512);
        }
    });
}
pub(crate) fn supplemental(state: [f32; 4], observation: [f32; 4], latent: f32) {
    ACTIVE.with(|s| {
        if let Some(c) = s.borrow_mut().as_mut() {
            assert!((512..1024).contains(&c.samples.len()));
            c.samples.push(Sample { state, observation, latent });
        }
    });
}
pub(super) fn fitting(batch: &RolloutBatch) {
    ACTIVE.with(|s| {
        if let Some(c) = s.borrow_mut().as_mut() {
            assert!(c.batch.is_none());
            c.batch = Some(batch.clone());
        }
    });
}
#[test]
fn capture_scope_unwinds_and_is_thread_local() {
    let _scope = Scope::enter();
    std::thread::spawn(|| { let _other = Scope::enter(); }).join().unwrap();
    assert!(std::panic::catch_unwind(Scope::enter).is_err());
    drop(_scope);
    assert!(std::panic::catch_unwind(|| { let _s = Scope::enter(); panic!("test unwind"); }).is_err());
    let _again = Scope::enter();
}
