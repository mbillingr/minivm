use crate::memory::RecordStorage;
use crate::primitive_value::PrimitiveValue;

const N_REGISTERS: usize = 8;

pub type Register = u8;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // Trivial Operations
    Term,
    Nop,

    // Debugging Helpers
    Debug(&'static str, Register),

    // Register Manipulation
    Const(Register, PrimitiveValue),
    Copy(Register, Register),
    Swap(Register, Register),

    // Arithmetic
    Inc(Register),
    Dec(Register),
    Add(Register, Register, Operand<PrimitiveValue>),
    Sub(Register, Register, Operand<PrimitiveValue>),
    Mul(Register, Register, Operand<PrimitiveValue>),
    Div(
        Register,
        Register,
        Operand<PrimitiveValue>,
        Operand<PrimitiveValue>,
    ),

    // Comparison
    Equal(Register, Register, Operand<PrimitiveValue>),
    Uneq(Register, Register, Operand<PrimitiveValue>),
    Less(Register, Operand<PrimitiveValue>, Operand<PrimitiveValue>),
    LessEq(Register, Operand<PrimitiveValue>, Operand<PrimitiveValue>),
    Not(Register, Register),

    // Branching
    Jmp(Operand<isize>),
    JmpFar(&'static [Op]),
    JmpCond(Operand<isize>, Register),
    LoadLabel(Register, isize),

    // Records
    Alloc(Register, usize),
    GetRec(Register, Register, Operand<usize>),
    SetRec(Register, Operand<usize>, Operand<PrimitiveValue>),

    // Pairs
    Cons(Register, Operand<PrimitiveValue>, Operand<PrimitiveValue>),
    Car(Register, Register),
    Cdr(Register, Register),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Operand<T> {
    R(Register),
    I(T),
}

impl<T> Operand<T>
where
    PrimitiveValue: Into<T>,
    T: std::fmt::Debug,
    T: Copy,
{
    fn eval(&self, registers: &[PrimitiveValue]) -> T {
        match self {
            Operand::R(r) => registers[*r as usize].into(),
            Operand::I(i) => *i,
        }
    }
}

pub fn eval(code: &'static [Op]) -> PrimitiveValue {
    let storage = RecordStorage::new(0);
    run(code, &storage)
}

pub fn run(mut code: &'static [Op], storage: &RecordStorage) -> PrimitiveValue {
    let mut register = [PrimitiveValue::Undefined; N_REGISTERS];
    let mut pc = 0;
    loop {
        match code[pc] {
            Op::Term => return register[0],
            Op::Nop => pc += 1,
            Op::Debug(context, r) => {
                println!("{}{:?}", context, register[r as usize]);
                pc += 1;
            }
            Op::Const(r, value) => {
                register[r as usize] = value;
                pc += 1;
            }
            Op::Copy(dst, src) => {
                register[dst as usize] = register[src as usize];
                pc += 1;
            }
            Op::Swap(a, b) => {
                let tmp = register[a as usize];
                register[a as usize] = register[b as usize];
                register[b as usize] = tmp;
                pc += 1;
            }
            Op::Inc(r) => {
                register[r as usize] = register[r as usize].add(1);
                pc += 1;
            }
            Op::Dec(r) => {
                register[r as usize] = register[r as usize].sub(1);
                pc += 1;
            }
            Op::Add(r, a, b) => {
                register[r as usize] = register[a as usize].add(b.eval(&register));
                pc += 1;
            }
            Op::Sub(r, a, b) => {
                register[r as usize] = register[a as usize].sub(b.eval(&register));
                pc += 1;
            }
            Op::Mul(r, a, b) => {
                register[r as usize] = register[a as usize].mul(b.eval(&register));
                pc += 1;
            }
            Op::Div(r, s, a, b) => {
                let (quot, rem) = a.eval(&register).div(b.eval(&register));
                register[r as usize] = quot;
                register[s as usize] = rem;
                pc += 1;
            }
            Op::Equal(r, a, b) => {
                register[r as usize] = register[a as usize].ptr_equal(b.eval(&register));
                pc += 1;
            }
            Op::Uneq(r, a, b) => {
                register[r as usize] = register[a as usize].ptr_equal(b.eval(&register)).not();
                pc += 1;
            }
            Op::Less(r, a, b) => {
                register[r as usize] = a.eval(&register).less(b.eval(&register));
                pc += 1;
            }
            Op::LessEq(r, a, b) => {
                register[r as usize] = a.eval(&register).less_eq(b.eval(&register));
                pc += 1;
            }
            Op::Not(dst, src) => {
                register[dst as usize] = register[src as usize].not();
                pc += 1;
            }
            Op::Jmp(op) => match op {
                Operand::R(r) => {
                    code = register[r as usize].as_codeblock();
                    pc = 0;
                }
                Operand::I(relative) => pc = (pc as isize + relative as isize) as usize,
            },
            Op::JmpCond(op, c) => {
                if register[c as usize].as_bool() {
                    match op {
                        Operand::R(r) => {
                            code = register[r as usize].as_codeblock();
                            pc = 0;
                        }
                        Operand::I(relative) => pc = (pc as isize + relative) as usize,
                    }
                } else {
                    pc += 1;
                }
            }
            Op::JmpFar(target) => {
                code = target;
                pc = 0;
            }
            Op::LoadLabel(r, offset) => {
                register[r as usize] =
                    PrimitiveValue::CodeBlock(&code[(pc as isize + offset) as usize..]);
                pc += 1;
            }
            Op::Alloc(r, size) => {
                let rec = storage.allocate_record(size, &mut register);
                register[r as usize] = PrimitiveValue::Record(rec);
                pc += 1;
            }
            Op::GetRec(dst, r, i) => {
                let rec = register[r as usize].as_record();
                let idx = i.eval(&register);
                register[dst as usize] = storage.get_record(rec)[idx];
                pc += 1;
            }
            Op::SetRec(r, i, v) => {
                let rec = register[r as usize].as_record();
                let idx = i.eval(&register);
                let value = v.eval(&register);
                storage.set_element(rec, idx, value);
                pc += 1;
            }
            Op::Cons(dst, car, cdr) => {
                let rec = storage.allocate_record(2, &mut register);
                storage.set_element(rec, 0, car.eval(&register));
                storage.set_element(rec, 1, cdr.eval(&register));
                register[dst as usize] = PrimitiveValue::Pair(rec.into());
                pc += 1;
            }
            Op::Car(dst, r) => {
                let rec = register[r as usize].as_record();
                register[dst as usize] = storage.get_record(rec)[0];
                pc += 1;
            }
            Op::Cdr(dst, r) => {
                let rec = register[r as usize].as_record();
                register[dst as usize] = storage.get_record(rec)[1];
                pc += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::store_code_block;
    use crate::test_utils::run_vm_test;
    use Operand::*;

    #[test]
    fn trivial_program_terminates_without_error() {
        run_vm_test(vec![Op::Term], PrimitiveValue::Undefined)
    }

    #[test]
    fn simple_block() {
        run_vm_test(vec![Op::Nop, Op::Nop, Op::Term], PrimitiveValue::Undefined)
    }

    #[test]
    fn constant() {
        run_vm_test(
            vec![Op::Const(0, PrimitiveValue::Integer(42)), Op::Term],
            42,
        )
    }

    #[test]
    fn simple_register_manipulation_runs_are_independent() {
        let code = vec![
            Op::Const(0, PrimitiveValue::Integer(0)),
            Op::Const(1, PrimitiveValue::Integer(0)),
            Op::Inc(0),
            Op::Inc(1),
            Op::Inc(0),
            Op::Cons(0, R(0), R(1)),
            Op::Term,
        ];
        run_vm_test(code.clone(), vec![2, 1]);
        run_vm_test(code.clone(), vec![2, 1]);
    }

    #[test]
    fn copy() {
        run_vm_test(
            vec![
                Op::Const(0, PrimitiveValue::Integer(1)),
                Op::Const(1, PrimitiveValue::Integer(2)),
                Op::Copy(0, 1),
                Op::Cons(0, R(0), R(1)),
                Op::Term,
            ],
            vec![2, 2],
        )
    }

    #[test]
    fn swap() {
        run_vm_test(
            vec![
                Op::Const(0, PrimitiveValue::Integer(1)),
                Op::Const(1, PrimitiveValue::Integer(2)),
                Op::Swap(0, 1),
                Op::Cons(0, R(0), R(1)),
                Op::Term,
            ],
            vec![2, 1],
        );
    }

    #[test]
    fn increment() {
        run_vm_test(
            vec![
                Op::Const(0, PrimitiveValue::Integer(1)),
                Op::Inc(0),
                Op::Term,
            ],
            2,
        )
    }

    #[test]
    fn decrement() {
        run_vm_test(
            vec![
                Op::Const(0, PrimitiveValue::Integer(0)),
                Op::Dec(0),
                Op::Term,
            ],
            -1,
        )
    }

    #[test]
    fn integer_addition() {
        run_vm_test(
            vec![
                Op::Const(1, PrimitiveValue::Integer(10)),
                Op::Const(2, PrimitiveValue::Integer(20)),
                Op::Add(0, 1, R(2)),
                Op::Term,
            ],
            30,
        )
    }

    #[test]
    fn integer_subtraction() {
        run_vm_test(
            vec![
                Op::Const(1, PrimitiveValue::Integer(10)),
                Op::Const(2, PrimitiveValue::Integer(20)),
                Op::Sub(0, 1, R(2)),
                Op::Term,
            ],
            -10,
        )
    }

    #[test]
    fn integer_multiplication() {
        run_vm_test(
            vec![
                Op::Const(1, PrimitiveValue::Integer(10)),
                Op::Const(2, PrimitiveValue::Integer(20)),
                Op::Mul(0, 1, R(2)),
                Op::Term,
            ],
            200,
        )
    }

    #[test]
    fn integer_division() {
        run_vm_test(
            vec![
                Op::Const(1, PrimitiveValue::Integer(50)),
                Op::Const(2, PrimitiveValue::Integer(20)),
                Op::Div(0, 1, R(1), R(2)),
                Op::Cons(0, R(0), R(1)),
                Op::Term,
            ],
            vec![2, 10],
        )
    }

    #[test]
    fn jump() {
        run_vm_test(
            vec![
                Op::Const(0, PrimitiveValue::Integer(0)),
                Op::Inc(0),
                Op::Jmp(I(4)),
                Op::Inc(0),
                Op::Jmp(I(3)),
                Op::Inc(0),
                Op::Jmp(I(-3)),
                Op::Term,
            ],
            2,
        );
    }

    #[test]
    fn jump_far() {
        let func = store_code_block(vec![Op::Inc(0), Op::Term]);
        let code = vec![
            Op::Const(0, PrimitiveValue::Integer(10)),
            Op::JmpFar(func),
            Op::Term,
        ];

        run_vm_test(code, 11);
    }

    #[test]
    fn jump_dynamic() {
        let func = store_code_block(vec![
            Op::Inc(0),
            Op::Const(1, PrimitiveValue::Undefined),
            Op::Term,
        ]);
        let code = vec![
            Op::Const(0, PrimitiveValue::Integer(10)),
            Op::Const(1, PrimitiveValue::CodeBlock(func)),
            Op::Jmp(R(1)),
            Op::Term,
        ];

        run_vm_test(code, 11);
    }

    #[test]
    fn record_allocation() {
        run_vm_test(
            vec![Op::Alloc(0, 1234), Op::Term],
            vec![PrimitiveValue::Undefined; 1234],
        )
    }

    #[test]
    fn record_set_element() {
        run_vm_test(
            vec![
                Op::Alloc(0, 3),
                Op::SetRec(0, I(0), I(PrimitiveValue::Integer(3))),
                Op::SetRec(0, I(1), I(PrimitiveValue::Integer(42))),
                Op::SetRec(0, I(1), I(PrimitiveValue::Integer(5))),
                Op::SetRec(0, I(2), I(PrimitiveValue::Integer(7))),
                Op::Term,
            ],
            vec![3, 5, 7],
        )
    }

    #[test]
    fn record_dynamic_set_element() {
        run_vm_test(
            vec![
                Op::Const(1, PrimitiveValue::Integer(0)),
                Op::Const(2, PrimitiveValue::Integer(100)),
                Op::Alloc(0, 3),
                Op::SetRec(0, R(1), R(2)),
                Op::Inc(1),
                Op::Dec(2),
                Op::SetRec(0, R(1), I(PrimitiveValue::Integer(99))),
                Op::Inc(1),
                Op::Dec(2),
                Op::SetRec(0, I(2), R(2)),
                Op::Inc(1),
                Op::Dec(2),
                Op::Term,
            ],
            vec![100, 99, 98],
        )
    }

    #[test]
    fn record_get_element() {
        run_vm_test(
            vec![
                Op::Alloc(1, 3),
                Op::SetRec(1, I(0), I(PrimitiveValue::Integer(3))),
                Op::SetRec(1, I(1), I(PrimitiveValue::Integer(42))),
                Op::SetRec(1, I(1), I(PrimitiveValue::Integer(5))),
                Op::SetRec(1, I(2), I(PrimitiveValue::Integer(7))),
                Op::GetRec(0, 1, I(2)),
                Op::Term,
            ],
            7,
        )
    }

    #[test]
    fn record_dynamic_get_element() {
        run_vm_test(
            vec![
                Op::Alloc(1, 3),
                Op::SetRec(1, I(0), I(PrimitiveValue::Integer(3))),
                Op::SetRec(1, I(1), I(PrimitiveValue::Integer(42))),
                Op::SetRec(1, I(1), I(PrimitiveValue::Integer(5))),
                Op::SetRec(1, I(2), I(PrimitiveValue::Integer(7))),
                Op::Const(2, PrimitiveValue::Integer(1)),
                Op::GetRec(0, 1, R(2)),
                Op::Term,
            ],
            5,
        )
    }

    #[test]
    fn cons() {
        run_vm_test(
            vec![
                Op::Const(1, PrimitiveValue::Integer(1)),
                Op::Cons(0, R(1), I(PrimitiveValue::Nil)),
                Op::Cons(0, I(PrimitiveValue::Integer(2)), R(0)),
                Op::Const(1, PrimitiveValue::Integer(3)),
                Op::Cons(0, R(1), R(0)),
                Op::Term,
            ],
            (3, (2, (1, PrimitiveValue::Nil))),
        )
    }

    #[test]
    fn car() {
        run_vm_test(
            vec![
                Op::Const(2, PrimitiveValue::Integer(2)),
                Op::Const(1, PrimitiveValue::Integer(1)),
                Op::Cons(0, R(1), R(2)),
                Op::Car(0, 0),
                Op::Term,
            ],
            1,
        )
    }

    #[test]
    fn cdr() {
        run_vm_test(
            vec![
                Op::Const(2, PrimitiveValue::Integer(2)),
                Op::Const(1, PrimitiveValue::Integer(1)),
                Op::Cons(0, R(1), R(2)),
                Op::Cdr(0, 0),
                Op::Term,
            ],
            2,
        )
    }

    #[test]
    fn factorial_cps() {
        // Implementation of the following continuation passing style program in bytecode:
        // (define (factorial n k)
        //   (if (= n 0)
        //       (k 1)
        //       (let ((n1 (- n 1)))
        //         (factorial n1
        //                    (lambda (f)
        //                      (k (* n f)))))))
        // The second argument k is the continuation function in closure format.
        // Closures are defined as records, where the element at index 0 is the function and the
        // remaining elements are free variables.
        // The lambda (called accumulate below) takes its closure as an implicit second argument.

        let accumulate = store_code_block(vec![
            // lambda (r1: f, r2: [self-fn, n, k]
            Op::GetRec(3, 2, I(1)), // put n in r3
            Op::GetRec(2, 2, I(2)), // put k in r2
            Op::GetRec(0, 2, I(0)), // put k's code in r0
            Op::Mul(1, 1, R(3)),    // put n * f in r1
            Op::Jmp(R(0)),          // call k with multiplication result in r1 and closure in r2
        ]);

        let fact = store_code_block(vec![
            // if n == 0
            Op::Uneq(0, 1, I(0.into())),
            Op::JmpCond(I(4), 0),
            // then k(1)
            Op::Const(1, PrimitiveValue::Integer(1)),
            Op::GetRec(0, 2, I(0)), // put k's code in r0
            Op::Jmp(R(0)),          // call k with 1 in r1 and closure in r2
            // else
            Op::Alloc(4, 3), // allocate closure
            Op::SetRec(4, I(0), I(PrimitiveValue::CodeBlock(accumulate))), // closure function
            Op::SetRec(4, I(1), R(1)), // n
            Op::SetRec(4, I(2), R(2)), // k
            Op::Copy(2, 4), // put closure in r2 as the new continuation of the recursive call to fact
            Op::Dec(1),     // n -= 1
            Op::Jmp(I(-11)), // call fact(n-1, accumulate)
        ]);

        let final_continuation = store_code_block(vec![Op::Copy(0, 1), Op::Term]);

        run_vm_test(
            vec![
                Op::Const(1, PrimitiveValue::Integer(0)), // initial n
                Op::Alloc(2, 1),                          // allocate closure with no variables
                Op::SetRec(2, I(0), I(PrimitiveValue::CodeBlock(final_continuation))), // put code in closure
                Op::JmpFar(fact), // call fact(n=0, final_continuation)
            ],
            1,
        );

        run_vm_test(
            vec![
                // fact(1, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(1)),
                Op::Alloc(2, 1),
                Op::SetRec(2, I(0), I(PrimitiveValue::CodeBlock(final_continuation))),
                Op::JmpFar(fact),
            ],
            1,
        );

        run_vm_test(
            vec![
                // fact(2, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(2)),
                Op::Alloc(2, 1),
                Op::SetRec(2, I(0), I(PrimitiveValue::CodeBlock(final_continuation))),
                Op::JmpFar(fact),
            ],
            2,
        );

        run_vm_test(
            vec![
                // fact(3, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(3)),
                Op::Alloc(2, 1),
                Op::SetRec(2, I(0), I(PrimitiveValue::CodeBlock(final_continuation))),
                Op::JmpFar(fact),
            ],
            6,
        );

        run_vm_test(
            vec![
                // fact(5, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(5)),
                Op::Alloc(2, 1),
                Op::SetRec(2, I(0), I(PrimitiveValue::CodeBlock(final_continuation))),
                Op::JmpFar(fact),
            ],
            120,
        );

        run_vm_test(
            vec![
                // fact(10, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(10)),
                Op::Alloc(2, 1),
                Op::SetRec(2, I(0), I(PrimitiveValue::CodeBlock(final_continuation))),
                Op::JmpFar(fact),
            ],
            3628800,
        );
    }

    #[test]
    fn fibonacci_cps() {
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
            Op::Jmp(R(5)),
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

        let test_fib = |n: i64, expect: i64| {
            run_vm_test(
                vec![
                    Op::Const(1, PrimitiveValue::Integer(n)),     // initial n
                    Op::Const(5, PrimitiveValue::CodeBlock(fib)), // put fib in a register, so lambda1 can find it
                    Op::Alloc(2, 1), // allocate closure with no variables
                    Op::SetRec(2, I(0), I(PrimitiveValue::CodeBlock(final_continuation))), // put code in closure
                    Op::JmpFar(fib), // call fact(n=0, final_continuation)
                ],
                expect,
            );
        };

        test_fib(0, 1);
        test_fib(1, 1);
        test_fib(2, 2);
        test_fib(5, 8);
        test_fib(10, 89);
        //test_fib(20, 10_946);
        //test_fib(40, 1_346_269);
    }

    #[test]
    fn fibonacci_stack() {
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

        let test_fib = |n: i64, expect: i64| {
            run_vm_test(
                vec![
                    Op::Alloc(STACK, 1024),
                    Op::Const(SP, PrimitiveValue::Integer(0)),
                    Op::Const(CONT, PrimitiveValue::CodeBlock(done)),
                    Op::Const(FIB, PrimitiveValue::CodeBlock(fib)), // put fib in a register
                    Op::Const(1, PrimitiveValue::Integer(n)),       // initial n
                    Op::JmpFar(fib), // call fact(n=0, final_continuation)
                ],
                expect,
            );
        };

        test_fib(0, 1);
        test_fib(1, 1);
        test_fib(2, 2);
        test_fib(5, 8);
        test_fib(10, 89);
        //test_fib(20, 10_946);
        //test_fib(40, 1_346_269);
    }
}
