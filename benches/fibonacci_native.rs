extern crate criterion;
use criterion::black_box;
use criterion::{criterion_group, criterion_main, Criterion};

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn bench_fib(c: &mut Criterion) {
    c.bench_function("fib_native 15", |b| b.iter(|| fibonacci(black_box(15))));
}

criterion_group!(benches, bench_fib);
criterion_main!(benches);
