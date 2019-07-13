use crate::memory::RecordStorage;
use crate::primitive_value::PrimitiveValue;

const N_REGISTERS: usize = 8;

type Register = u8;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Term,
    Nop,

    Debug(&'static str, Register),

    Const(Register, PrimitiveValue),
    Copy(Register, Register),
    Swap(Register, Register),

    Inc(Register),
    Dec(Register),
    Add(Register, Register, Register),
    Sub(Register, Register, Register),
    Mul(Register, Register, Register),
    Div(Register, Register, Register, Register),

    Equal(Register, Register, Register),
    Uneq(Register, Register, Register),
    Less(Register, Register, Register),
    LessEq(Register, Register, Register),
    Not(Register, Register),

    Jmp(i8),
    JmpCond(i8, Register),
    JmpFar(&'static [Op]),
    JmpDyn(Register),

    Alloc(Register, usize),
    GetRec(Register, Register, usize),
    GetRecDyn(Register, Register, Register),
    SetRec(Register, usize, Register),
    SetRecDyn(Register, Register, Register),

    Cons(Register, Register, Register),
    Car(Register, Register),
    Cdr(Register, Register),
}

pub fn run(mut code: &[Op], storage: &RecordStorage) -> PrimitiveValue {
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
                register[r as usize] = register[a as usize].add(register[b as usize]);
                pc += 1;
            }
            Op::Sub(r, a, b) => {
                register[r as usize] = register[a as usize].sub(register[b as usize]);
                pc += 1;
            }
            Op::Mul(r, a, b) => {
                register[r as usize] = register[a as usize].mul(register[b as usize]);
                pc += 1;
            }
            Op::Div(r, s, a, b) => {
                let (quot, rem) = register[a as usize].div(register[b as usize]);
                register[r as usize] = quot;
                register[s as usize] = rem;
                pc += 1;
            }
            Op::Equal(r, a, b) => {
                register[r as usize] = register[a as usize].ptr_equal(register[b as usize]);
                pc += 1;
            }
            Op::Uneq(r, a, b) => {
                register[r as usize] = (register[a as usize].ptr_equal(register[b as usize])).not();
                pc += 1;
            }
            Op::Less(r, a, b) => {
                register[r as usize] = register[a as usize].less(register[b as usize]);
                pc += 1;
            }
            Op::LessEq(r, a, b) => {
                register[r as usize] = register[a as usize].less_eq(register[b as usize]);
                pc += 1;
            }
            Op::Not(dst, src) => {
                register[dst as usize] = register[src as usize].not();
                pc += 1;
            }
            Op::Jmp(relative) => pc = (pc as isize + relative as isize) as usize,
            Op::JmpCond(relative, r) => {
                if register[r as usize].as_bool() {
                    pc = (pc as isize + relative as isize) as usize;
                } else {
                    pc += 1;
                }
            }
            Op::JmpFar(target) => {
                code = target;
                pc = 0;
            }
            Op::JmpDyn(r) => {
                code = register[r as usize].as_codeblock();
                pc = 0;
            }
            Op::Alloc(r, size) => {
                let rec = storage.allocate_record(size, &mut register);
                register[r as usize] = PrimitiveValue::Record(rec);
                pc += 1;
            }
            Op::GetRec(dst, r, idx) => {
                let rec = register[r as usize].as_record();
                register[dst as usize] = storage.get_record(rec)[idx];
                pc += 1;
            }
            Op::GetRecDyn(dst, r, i) => {
                let rec = register[r as usize].as_record();
                let idx = register[i as usize].as_int() as usize;
                register[dst as usize] = storage.get_record(rec)[idx];
                pc += 1;
            }
            Op::SetRec(r, idx, src) => {
                let rec = register[r as usize].as_record();
                let value = register[src as usize];
                storage.set_element(rec, idx, value);
                pc += 1;
            }
            Op::SetRecDyn(r, i, v) => {
                let rec = register[r as usize].as_record();
                let idx = register[i as usize].as_int() as usize;
                storage.set_element(rec, idx, register[v as usize]);
                pc += 1;
            }
            Op::Cons(dst, car, cdr) => {
                let rec = storage.allocate_record(2, &mut register);
                storage.set_element(rec, 0, register[car as usize]);
                storage.set_element(rec, 1, register[cdr as usize]);
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

    #[derive(Debug, PartialEq)]
    enum TestResult {
        PrimitiveValue(PrimitiveValue),
        Compound(Vec<TestResult>),
    }

    impl From<i64> for TestResult {
        fn from(i: i64) -> Self {
            TestResult::PrimitiveValue(PrimitiveValue::Integer(i))
        }
    }

    impl From<PrimitiveValue> for TestResult {
        fn from(p: PrimitiveValue) -> Self {
            TestResult::PrimitiveValue(p)
        }
    }

    impl<T: Into<TestResult>> From<Vec<T>> for TestResult {
        fn from(v: Vec<T>) -> Self {
            TestResult::Compound(v.into_iter().map(T::into).collect())
        }
    }

    impl<CAR: Into<TestResult>, CDR: Into<TestResult>> From<(CAR, CDR)> for TestResult {
        fn from((car, cdr): (CAR, CDR)) -> Self {
            TestResult::Compound(vec![car.into(), cdr.into()])
        }
    }

    impl TestResult {
        fn from_run(r: PrimitiveValue, storage: &RecordStorage) -> Self {
            match r {
                PrimitiveValue::Record(r) => storage
                    .get_record(r)
                    .iter()
                    .map(|&x| TestResult::from_run(x, storage))
                    .collect::<Vec<_>>()
                    .into(),
                PrimitiveValue::Pair(p) => storage
                    .get_pair(p)
                    .iter()
                    .map(|&&x| TestResult::from_run(x, storage))
                    .collect::<Vec<_>>()
                    .into(),
                p => p.into(),
            }
        }
    }

    fn run_test(code: &[Op], expect: impl Into<TestResult>) {
        let storage = RecordStorage::new(0);
        let result = run(code, &storage);
        assert_eq!(TestResult::from_run(result, &storage), expect.into());
    }

    #[test]
    fn trivial_program_terminates_without_error() {
        run_test(&[Op::Term], PrimitiveValue::Undefined)
    }

    #[test]
    fn simple_block() {
        run_test(&[Op::Nop, Op::Nop, Op::Term], PrimitiveValue::Undefined)
    }

    #[test]
    fn constant() {
        run_test(
            &[Op::Const(0, PrimitiveValue::Integer(42)), Op::Term],
            42
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
            Op::Cons(0, 0, 1),
            Op::Term,
        ];
        run_test(&code, vec![2, 1]);
        run_test(&code, vec![2, 1]);
    }

    #[test]
    fn copy() {
        run_test(
            &[
                Op::Const(0, PrimitiveValue::Integer(1)),
                Op::Const(1, PrimitiveValue::Integer(2)),
                Op::Copy(0, 1),
                Op::Cons(0, 0, 1),
                Op::Term,
            ],
            vec![2, 2]
        )
    }

    #[test]
    fn swap() {
        run_test(
            &[
                Op::Const(0, PrimitiveValue::Integer(1)),
                Op::Const(1, PrimitiveValue::Integer(2)),
                Op::Swap(0, 1),
                Op::Cons(0, 0, 1),
                Op::Term,
            ],
            vec![2, 1]
        );
    }

    #[test]
    fn increment() {
        run_test(
            &[
                Op::Const(0, PrimitiveValue::Integer(1)),
                Op::Inc(0),
                Op::Term,
            ],
            2
        )
    }

    #[test]
    fn decrement() {
        run_test(
            &[
                Op::Const(0, PrimitiveValue::Integer(0)),
                Op::Dec(0),
                Op::Term,
            ],
            -1
        )
    }

    #[test]
    fn integer_addition() {
        run_test(
            &[
                Op::Const(1, PrimitiveValue::Integer(10)),
                Op::Const(2, PrimitiveValue::Integer(20)),
                Op::Add(0, 1, 2),
                Op::Term,
            ],
            30,
        )
    }

    #[test]
    fn integer_subtraction() {
        run_test(
            &[
                Op::Const(1, PrimitiveValue::Integer(10)),
                Op::Const(2, PrimitiveValue::Integer(20)),
                Op::Sub(0, 1, 2),
                Op::Term,
            ],
            -10
        )
    }

    #[test]
    fn integer_multiplication() {
        run_test(
            &[
                Op::Const(1, PrimitiveValue::Integer(10)),
                Op::Const(2, PrimitiveValue::Integer(20)),
                Op::Mul(0, 1, 2),
                Op::Term,
            ],
            200
        )
    }

    #[test]
    fn integer_division() {
        run_test(
            &[
                Op::Const(1, PrimitiveValue::Integer(50)),
                Op::Const(2, PrimitiveValue::Integer(20)),
                Op::Div(0, 1, 1, 2),
                Op::Cons(0, 0, 1),
                Op::Term,
            ],
            vec![2, 10]
        )
    }

    #[test]
    fn jump() {
        run_test(
            &[
                Op::Const(0, PrimitiveValue::Integer(0)),
                Op::Inc(0),
                Op::Jmp(4),
                Op::Inc(0),
                Op::Jmp(3),
                Op::Inc(0),
                Op::Jmp(-3),
                Op::Term,
            ],
            2,
        );
    }

    #[test]
    fn jump_far() {
        let func = store_code_block(vec![
            Op::Inc(0),
            Op::Term,
        ]);
        let code = vec![
            Op::Const(0, PrimitiveValue::Integer(10)),
            Op::JmpFar(func),
            Op::Term,
        ];

        run_test(&code, 11);
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
            Op::JmpDyn(1),
            Op::Term,
        ];

        run_test(&code, 11);
    }

    #[test]
    fn record_allocation() {
        run_test(
            &[
                Op::Alloc(0, 1234),
                Op::Term,
            ],
            vec![PrimitiveValue::Undefined; 1234]
        )
    }

    #[test]
    fn record_set_element() {
        run_test(
            &[
                Op::Alloc(0, 3),
                Op::Const(1, PrimitiveValue::Integer(3)),
                Op::SetRec(0, 0, 1),
                Op::Const(1, PrimitiveValue::Integer(42)),
                Op::SetRec(0, 1, 1),
                Op::Const(1, PrimitiveValue::Integer(5)),
                Op::SetRec(0, 1, 1),
                Op::Const(1, PrimitiveValue::Integer(7)),
                Op::SetRec(0, 2, 1),
                Op::Term,
            ],
            vec![3, 5, 7]
        )
    }

    #[test]
    fn record_dynamic_set_element() {
        run_test(
            &[
                Op::Const(1, PrimitiveValue::Integer(0)),
                Op::Const(2, PrimitiveValue::Integer(100)),
                Op::Alloc(0, 3),
                Op::SetRecDyn(0, 1, 2),
                Op::Inc(1),
                Op::Dec(2),
                Op::SetRecDyn(0, 1, 2),
                Op::Inc(1),
                Op::Dec(2),
                Op::SetRecDyn(0, 1, 2),
                Op::Inc(1),
                Op::Dec(2),
                Op::Term,
            ],
            vec![100, 99, 98]
        )
    }

    #[test]
    fn record_get_element() {
        run_test(
            &[
                Op::Alloc(1, 3),
                Op::Const(2, PrimitiveValue::Integer(3)),
                Op::SetRec(1, 0, 2),
                Op::Const(2, PrimitiveValue::Integer(42)),
                Op::SetRec(1, 1, 2),
                Op::Const(2, PrimitiveValue::Integer(5)),
                Op::SetRec(1, 1, 2),
                Op::Const(2, PrimitiveValue::Integer(7)),
                Op::SetRec(1, 2, 2),
                Op::GetRec(0, 1, 2),
                Op::Term,
            ],
            7
        )
    }

    #[test]
    fn record_dynamic_get_element() {
        run_test(
            &[
                Op::Alloc(1, 3),
                Op::Const(2, PrimitiveValue::Integer(3)),
                Op::SetRec(1, 0, 2),
                Op::Const(2, PrimitiveValue::Integer(42)),
                Op::SetRec(1, 1, 2),
                Op::Const(2, PrimitiveValue::Integer(5)),
                Op::SetRec(1, 1, 2),
                Op::Const(2, PrimitiveValue::Integer(7)),
                Op::SetRec(1, 2, 2),
                Op::Const(2, PrimitiveValue::Integer(1)),
                Op::GetRecDyn(0, 1, 2),
                Op::Term,
            ],
            5
        )
    }

    #[test]
    fn cons() {
        run_test(
            &[
                Op::Const(2, PrimitiveValue::Nil),
                Op::Const(1, PrimitiveValue::Integer(1)),
                Op::Cons(0, 1, 2),
                Op::Const(1, PrimitiveValue::Integer(2)),
                Op::Cons(0, 1, 0),
                Op::Const(1, PrimitiveValue::Integer(3)),
                Op::Cons(0, 1, 0),
                Op::Term,
            ],
            (3, (2, (1, PrimitiveValue::Nil)))
        )
    }

    #[test]
    fn car() {
        run_test(
            &[
                Op::Const(2, PrimitiveValue::Integer(2)),
                Op::Const(1, PrimitiveValue::Integer(1)),
                Op::Cons(0, 1, 2),
                Op::Car(0, 0),
                Op::Term,
            ],
            1
        )
    }

    #[test]
    fn cdr() {
        run_test(
            &[
                Op::Const(2, PrimitiveValue::Integer(2)),
                Op::Const(1, PrimitiveValue::Integer(1)),
                Op::Cons(0, 1, 2),
                Op::Cdr(0, 0),
                Op::Term,
            ],
            2
        )
    }

    #[test]
    fn factorial() {
        // Implement the following Scheme program in bytecode:
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
            Op::GetRec(3, 2, 1),  // put n in r3
            Op::GetRec(2, 2, 2),  // put k in r2
            Op::GetRec(0, 2, 0),  // put k's code in r0
            Op::Mul(1, 1, 3),     // put n * f in r1
            Op::JmpDyn(0),        // call k with multiplication result in r1 and closure in r2
        ]);

        let fact = store_code_block( vec![
            Op::Const(7, PrimitiveValue::Integer(0)),   // constant 0

            // if n == 0
            Op::Uneq(0, 1, 7),
            Op::JmpCond(4, 0),
            // then k(1)
            Op::Const(1, PrimitiveValue::Integer(1)),
            Op::GetRec(0, 2, 0),  // put k's code in r0
            Op::JmpDyn(0),        // call k with 1 in r1 and closure in r2
            // else
            Op::Alloc(4, 3),                                      // allocate closure
            Op::Const(3, PrimitiveValue::CodeBlock(accumulate)),
            Op::SetRec(4, 0, 3),                                  // closure function
            Op::SetRec(4, 1, 1),                                  // n
            Op::SetRec(4, 2, 2),                                  // k
            Op::Copy(2, 4),                                       // put closure in r2 as the new continuation of the recursive call to fact
            Op::Dec(1),                                           // n -= 1
            Op::Jmp(-12),                                         // call fact(n-1, accumulate)
        ]);

        let final_continuation = store_code_block(vec![Op::Copy(0, 1), Op::Term]);

        run_test(
            &[
                Op::Const(1, PrimitiveValue::Integer(0)),                    // initial n
                Op::Alloc(2, 1),                                             // allocate closure with no variables
                Op::Const(3, PrimitiveValue::CodeBlock(final_continuation)),
                Op::SetRec(2, 0, 3),                                         // put code in closure
                Op::JmpFar(fact),                                            // call fact(n=0, final_continuation)
            ],
            1
        );

        run_test(
            &[
                // fact(1, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(1)),
                Op::Alloc(2, 1),
                Op::Const(3, PrimitiveValue::CodeBlock(final_continuation)),
                Op::SetRec(2, 0, 3),
                Op::JmpFar(fact),
            ],
            1
        );

        run_test(
            &[
                // fact(2, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(2)),
                Op::Alloc(2, 1),
                Op::Const(3, PrimitiveValue::CodeBlock(final_continuation)),
                Op::SetRec(2, 0, 3),
                Op::JmpFar(fact),
            ],
            2
        );

        run_test(
            &[
                // fact(3, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(3)),
                Op::Alloc(2, 1),
                Op::Const(3, PrimitiveValue::CodeBlock(final_continuation)),
                Op::SetRec(2, 0, 3),
                Op::JmpFar(fact),
            ],
            6
        );

        run_test(
            &[
                // fact(5, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(5)),
                Op::Alloc(2, 1),
                Op::Const(3, PrimitiveValue::CodeBlock(final_continuation)),
                Op::SetRec(2, 0, 3),
                Op::JmpFar(fact),
            ],
            120
        );

        run_test(
            &[
                // fact(10, final_continuation)
                Op::Const(1, PrimitiveValue::Integer(10)),
                Op::Alloc(2, 1),
                Op::Const(3, PrimitiveValue::CodeBlock(final_continuation)),
                Op::SetRec(2, 0, 3),
                Op::JmpFar(fact),
            ],
            3628800
        );
    }
}
