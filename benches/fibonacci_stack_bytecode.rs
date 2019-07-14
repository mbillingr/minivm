extern crate criterion;
use criterion::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use minivm::memory::{store_code_block, RecordStorage};
use minivm::primitive_value::PrimitiveValue;
use minivm::virtual_machine::{run, Op, Operand};

fn fibonacci(n: i64) -> &'static [Op] {
    // Implementation of the following program in bytecode:
    // (define (fib n)
    //  (if (< n 2)
    //      1
    //      (+ (fib (-n 1)) (fib (- n 2)))
    use Operand::*;

    const STACK: u8 = 7;
    const SP: u8 = 6;
    const CONT: u8 = 5;
    const FIB: u8 = 4;

    macro_rules! push {
            ($r:expr) => {
                Op::SetRec(STACK, R(SP), R(r)),
                Op::Inc(SP),
            }
        }

    macro_rules! pop {
            ($r:expr) => (
                Op::Dec(SP),
                Op::GetRec(r, STACK, R(SP)),
            )
        }

    let after_fib2 = store_code_block(vec![
        //pop!(1)
        Op::Dec(SP),
        Op::GetRec(1, STACK, R(SP)),
        //pop!(CONT),
        Op::Dec(SP),
        Op::GetRec(CONT, STACK, R(SP)),
        Op::Add(0, 0, R(1)),
        Op::Jmp(R(CONT)),
    ]);

    let after_fib1 = store_code_block(vec![
        //pop!(CONT),
        Op::Dec(SP),
        Op::GetRec(CONT, STACK, R(SP)),
        //pop!(1),
        Op::Dec(SP),
        Op::GetRec(1, STACK, R(SP)),
        Op::Dec(1),
        //push!(CONT),
        Op::SetRec(STACK, R(SP), R(CONT)),
        Op::Inc(SP),
        Op::Const(CONT, PrimitiveValue::CodeBlock(after_fib2)),
        //push!(0),
        Op::SetRec(STACK, R(SP), R(0)),
        Op::Inc(SP),
        Op::Jmp(R(FIB)),
    ]);

    let fib = store_code_block(vec![
        // if n == 0
        Op::LessEq(0, I(2.into()), R(1)),
        Op::JmpCond(I(3), 0),
        // then k(1)
        Op::Const(0, PrimitiveValue::Integer(1)),
        Op::Jmp(R(CONT)),
        // else
        Op::Dec(1), // n -= 1
        //push!(1),
        Op::SetRec(STACK, R(SP), R(1)),
        Op::Inc(SP),
        //push!(CONT),
        Op::SetRec(STACK, R(SP), R(CONT)),
        Op::Inc(SP),
        Op::Const(CONT, PrimitiveValue::CodeBlock(after_fib1)),
        Op::Jmp(R(FIB)),
    ]);

    let done = store_code_block(vec![Op::Term]);

    let code = store_code_block(vec![
        Op::Alloc(STACK, 1024),
        Op::Const(SP, PrimitiveValue::Integer(0)),
        Op::Const(CONT, PrimitiveValue::CodeBlock(done)),
        Op::Const(FIB, PrimitiveValue::CodeBlock(fib)), // put fib in a register
        Op::Const(1, PrimitiveValue::Integer(n)),       // initial n
        Op::JmpFar(fib),                                // call fact(n=0, final_continuation)
    ]);

    code
}

fn bench_fib(c: &mut Criterion) {
    let code = fibonacci(15);
    let storage = RecordStorage::new(0);

    c.bench_function("fib_stack_bytecode 15", move |b| {
        b.iter(|| run(black_box(code), &storage))
    });
}

criterion_group!(benches, bench_fib);
criterion_main!(benches);
