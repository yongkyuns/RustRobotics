//! Scoped, test-only collection length; default behavior remains historical.
use std::{cell::Cell, marker::PhantomData, rc::Rc};
thread_local! {
    static ACTIVE: Cell<Option<usize>> = const { Cell::new(None) };
}
pub(crate) fn steps(default: usize) -> usize {
    ACTIVE.with(|v| v.get().unwrap_or(default))
}
/// Must be dropped on the thread that entered it; nested modes are rejected.
pub(crate) struct Window(PhantomData<Rc<()>>);
impl Window {
    pub(crate) fn enter(n: usize) -> Self {
        assert!([512, 1024].contains(&n), "unregistered support length");
        ACTIVE.with(|v| {
            assert!(v.get().is_none(), "nested support scope");
            v.set(Some(n));
        });
        Self(PhantomData)
    }
}
impl Drop for Window {
    fn drop(&mut self) { ACTIVE.with(|v| v.set(None)); }
}
#[test]
fn scope_is_thread_local_and_restores_after_unwind() {
    assert_eq!(steps(512), 512);
    let result = std::panic::catch_unwind(|| {
        let _scope = Window::enter(1024);
        assert_eq!(steps(512), 1024);
        assert_eq!(std::thread::spawn(|| steps(512)).join().unwrap(), 512);
        panic!("intentional unwind");
    });
    assert!(result.is_err());
    assert_eq!(steps(512), 512);
}
#[test]
fn invalid_and_nested_scopes_cannot_change_the_active_length() {
    assert!(std::panic::catch_unwind(|| Window::enter(1)).is_err());
    let _scope = Window::enter(1024);
    assert!(std::panic::catch_unwind(|| Window::enter(512)).is_err());
    assert_eq!(steps(512), 1024);
}
