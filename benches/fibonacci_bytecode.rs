extern crate criterion;
use criterion::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use minivm::memory::{store_code_block, RecordStorage};
use minivm::primitive_value::PrimitiveValue;
use minivm::virtual_machine::{run, Op, Operand};

fn fibonacci(n: i64) -> &'static [Op] {
    // Implementation of the following continuation passing style program in bytecode:
    // (define (fib n k)
    //  (if (< n 2)
    //      (k 1)
    //      (fib (- n 1) (lambda (f2)
    //                     (fib (- n 2) (lambda (f1)
    //                                    (k (+ f1 f2))))))))
    // The second argument k is the continuation function in closure format.
    // Closures are defined as records, where the element at index 0 is the function and the
    // remaining elements are free variables.
    // The lambdas take their closure as an implicit second argument.
    use Operand::*;

    let lambda2 = store_code_block(vec![
        // lambda (r1: f2, r2: [self-fn, f1, k]
        Op::GetRec(3, 2, I(1)), // put f1 in r3
        Op::GetRec(2, 2, I(2)), // put k in r2
        Op::GetRec(0, 2, I(0)), // put k's code in r0
        Op::Add(1, 1, R(3)),    // put n * f in r1
        Op::Jmp(R(0)),          // call k with sum in r1 and closure in r2
    ]);

    let lambda1 = store_code_block(vec![
        // lambda (r1: f1, r2: [self-fn, n-1, k]
        Op::GetRec(3, 2, I(1)), // put n-1 in r3
        // re-use the closure record
        Op::SetRec(2, I(0), I(PrimitiveValue::CodeBlock(lambda2))), // closure function
        Op::SetRec(2, I(1), R(1)),                                  // replace n-1 with f1
        Op::Copy(1, 3),                                             // put n-1 in r1
        Op::Dec(1),                                                 // put n-2 in r1
        Op::Jmp(R(5)),                                              // expect fib in r5
    ]);

    let fib = store_code_block(vec![
        // if n == 0
        Op::LessEq(0, I(2.into()), R(1)),
        Op::JmpCond(I(4), 0),
        // then k(1)
        Op::Const(1, PrimitiveValue::Integer(1)),
        Op::GetRec(0, 2, I(0)), // put k's code in r0
        Op::Jmp(R(0)),          // call k with 1 in r1 and closure in r2
        // else
        Op::Dec(1),                                                 // n -= 1
        Op::Alloc(4, 3),                                            // allocate closure
        Op::SetRec(4, I(0), I(PrimitiveValue::CodeBlock(lambda1))), // closure function
        Op::SetRec(4, I(1), R(1)),                                  // n-1
        Op::SetRec(4, I(2), R(2)),                                  // k
        Op::Copy(2, 4), // put closure in r2 as the new continuation of the recursive call to fib
        Op::Jmp(I(-11)), // call fib(n-1, lambda1)
    ]);

    let final_continuation = store_code_block(vec![Op::Copy(0, 1), Op::Term]);

    let code = store_code_block(vec![
        Op::Const(1, PrimitiveValue::Integer(n)),     // initial n
        Op::Const(5, PrimitiveValue::CodeBlock(fib)), // put fib in a register, so lambda1 can find it
        Op::Alloc(2, 1),                              // allocate closure with no variables
        Op::SetRec(2, I(0), I(PrimitiveValue::CodeBlock(final_continuation))), // put code in closure
        Op::JmpFar(fib), // call fact(n=0, final_continuation)
    ]);

    code
}

fn bench_fib(c: &mut Criterion) {
    let code = fibonacci(15);
    let storage = RecordStorage::new(0);

    c.bench_function("fib_bytecode 15", move |b| {
        b.iter(|| run(black_box(code), &storage))
    });
}

criterion_group!(benches, bench_fib);
criterion_main!(benches);
