use crate::memory::{Cell, Pair, Record};
use crate::virtual_machine::Op;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum PrimitiveValue {
    Undefined,
    Nil,
    True,
    False,
    Integer(i64),
    Record(Record),
    Pair(Pair),
    Cell(Cell),

    CodeBlock(&'static [Op]),

    Relocated(usize),
}

impl PrimitiveValue {
    pub fn as_bool(&self) -> bool {
        match self {
            PrimitiveValue::True => true,
            PrimitiveValue::False => false,
            _ => panic!("Type Error: not a boolean -- {:?}", self),
        }
    }

    pub fn as_int(&self) -> i64 {
        if let PrimitiveValue::Integer(x) = self {
            *x
        } else {
            panic!("Type Error: not an integer -- {:?}", self)
        }
    }

    pub fn as_codeblock(&self) -> &'static [Op] {
        if let PrimitiveValue::CodeBlock(c) = self {
            c
        } else {
            panic!("Type Error: not code -- {:?}", self)
        }
    }

    pub fn as_record(&self) -> Record {
        match self.try_as_record() {
            Some(r) => r,
            None => panic!("Type Error: not a record -- {:?}", self),
        }
    }

    pub fn try_as_record(&self) -> Option<Record> {
        match *self {
            PrimitiveValue::Record(r) => Some(r),
            PrimitiveValue::Pair(p) => Some(p.into()),
            _ => None,
        }
    }
}

impl PrimitiveValue {
    pub fn add(self, x: impl Into<PrimitiveValue>) -> PrimitiveValue {
        use PrimitiveValue::*;
        match (self, x.into()) {
            (Integer(a), Integer(b)) => Integer(a + b),
            _ => panic!("Type Error"),
        }
    }

    pub fn sub(self, x: impl Into<PrimitiveValue>) -> PrimitiveValue {
        use PrimitiveValue::*;
        match (self, x.into()) {
            (Integer(a), Integer(b)) => Integer(a - b),
            _ => panic!("Type Error"),
        }
    }

    pub fn mul(self, x: impl Into<PrimitiveValue>) -> PrimitiveValue {
        use PrimitiveValue::*;
        match (self, x.into()) {
            (Integer(a), Integer(b)) => Integer(a * b),
            _ => panic!("Type Error"),
        }
    }

    pub fn div(self, x: impl Into<PrimitiveValue>) -> PrimitiveValue {
        use PrimitiveValue::*;
        match (self, x.into()) {
            (Integer(a), Integer(b)) => Integer(a / b),
            _ => panic!("Type Error"),
        }
    }
}

impl PrimitiveValue {
    pub fn ptr_equal(self, x: impl Into<PrimitiveValue>) -> PrimitiveValue {
        use PrimitiveValue::*;
        match (self, x.into()) {
            (Undefined, Undefined) => false,
            (Nil, Nil) => false,
            (True, True) => false,
            (False, False) => false,
            (Integer(a), Integer(b)) => a == b,
            (Record(r1), Record(r2)) => r1.start_idx == r2.start_idx && r1.len == r2.len,
            (Pair(a), Pair(b)) => a.start_idx == b.start_idx,
            (CodeBlock(a), CodeBlock(b)) => a.as_ptr() == b.as_ptr(),
            (Relocated(_), _) | (_, Relocated(_)) => unreachable!(),
            _ => false,
        }
        .into()
    }

    pub fn not(self) -> PrimitiveValue {
        use PrimitiveValue::*;
        match self {
            True => False,
            False => True,
            _ => panic!("Type Error"),
        }
        .into()
    }

    pub fn less(self, x: impl Into<PrimitiveValue>) -> PrimitiveValue {
        use PrimitiveValue::*;
        match (self, x.into()) {
            (Integer(a), Integer(b)) => a < b,
            _ => panic!("Type Error"),
        }
        .into()
    }

    pub fn less_eq(self, x: impl Into<PrimitiveValue>) -> PrimitiveValue {
        use PrimitiveValue::*;
        match (self, x.into()) {
            (Integer(a), Integer(b)) => a <= b,
            _ => panic!("Type Error"),
        }
        .into()
    }
}

impl From<i64> for PrimitiveValue {
    fn from(x: i64) -> PrimitiveValue {
        PrimitiveValue::Integer(x)
    }
}

impl From<bool> for PrimitiveValue {
    fn from(x: bool) -> PrimitiveValue {
        match x {
            true => PrimitiveValue::True,
            false => PrimitiveValue::False,
        }
    }
}

impl From<PrimitiveValue> for usize {
    fn from(x: PrimitiveValue) -> Self {
        match x {
            PrimitiveValue::Integer(i) if i >= 0 => i as usize,
            PrimitiveValue::Integer(i) if i < 0 => panic!("Value error: expected positive integer"),
            _ => panic!("Type Error"),
        }
    }
}
