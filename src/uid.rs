use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_UID: AtomicU64 = AtomicU64::new(0);

pub fn global_uid() -> u64 {
    NEXT_UID.fetch_add(1, Ordering::SeqCst)
}
