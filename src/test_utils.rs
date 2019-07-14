use crate::memory::RecordStorage;
use crate::primitive_value::PrimitiveValue;
use crate::virtual_machine::{run, Op};

#[derive(Debug, PartialEq)]
pub enum TestResult {
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

pub fn run_vm_test(code: &[Op], expect: impl Into<TestResult>) {
    let storage = RecordStorage::new(0);
    let result = run(code, &storage);
    assert_eq!(TestResult::from_run(result, &storage), expect.into());
}
