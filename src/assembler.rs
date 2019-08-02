use crate::primitive_value::PrimitiveValue;
use crate::virtual_machine::{Op, Operand};
use std::collections::HashMap;

pub type AssemblerResult<T> = std::result::Result<T, AssemblerError>;

#[derive(Debug)]
pub enum AssemblerError {
    UndefinedLabel(String),
    LabelNotUnique(String),
}

#[derive(Debug)]
pub struct Assembler {
    source: Vec<AsmOp>,
}

impl Assembler {
    pub fn new() -> Self {
        Assembler { source: vec![] }
    }

    pub fn append(&mut self, other: &Self) {
        self.source.extend(other.source.iter().cloned())
    }

    pub fn assemble(self) -> AssemblerResult<(Vec<Op>, HashMap<String, usize>)> {
        let mut label_positions = map![];
        let mut pos = 0;
        for instruction in &self.source {
            match instruction {
                AsmOp::Label(l) => {
                    if label_positions.contains_key(l) {
                        Err(AssemblerError::LabelNotUnique(l.clone()))?;
                    }
                    label_positions.insert(l.clone(), pos);
                }
                AsmOp::Opcode(op) => pos += 1,
            }
        }

        self.source
            .into_iter()
            .filter_map(AsmOp::into_op)
            .enumerate()
            .map(|(pos, op)| convert_operation(op, pos, &label_positions))
            .collect::<AssemblerResult<_>>()
            .map(|ops| (ops, label_positions))
    }

    pub fn label(&mut self, l: String) {
        self.source.push(AsmOp::Label(l))
    }

    pub fn op(&mut self, op: Op<String>) {
        self.source.push(AsmOp::Opcode(op))
    }
}

fn convert_operation(
    op: Op<String>,
    pos: usize,
    label_positions: &HashMap<String, usize>,
) -> AssemblerResult<Op<isize>> {
    Ok(match op {
        Op::Term => Op::Term,
        Op::Nop => Op::Nop,
        Op::Debug(s, r) => Op::Debug(s, r),
        Op::Const(r, c) => Op::Const(r, c),
        Op::Copy(a, b) => Op::Copy(a, b),
        Op::Inc(r) => Op::Inc(r),
        Op::Dec(r) => Op::Dec(r),
        Op::Add(c, a, b) => Op::Add(c, a, b),
        Op::Sub(c, a, b) => Op::Sub(c, a, b),
        Op::Mul(c, a, b) => Op::Mul(c, a, b),
        Op::Div(c, a, b) => Op::Div(c, a, b),
        Op::Equal(c, a, b) => Op::Equal(c, a, b),
        Op::Uneq(c, a, b) => Op::Uneq(c, a, b),
        Op::Less(c, a, b) => Op::Less(c, a, b),
        Op::LessEq(c, a, b) => Op::LessEq(c, a, b),
        Op::Not(a, b) => Op::Not(a, b),
        Op::JmpFar(x) => Op::JmpFar(x),
        Op::Jmp(a) => Op::Jmp(convert_operand(a, pos, &label_positions)?),
        Op::JmpCond(x, c) => Op::JmpCond(convert_operand(x, pos, &label_positions)?, c),
        Op::LoadLabel(r, x) => Op::LoadLabel(r, convert_label(x, pos, &label_positions)?),
        Op::Alloc(r, n) => Op::Alloc(r, n),
        Op::GetRec(a, b, c) => Op::GetRec(a, b, c),
        Op::SetRec(a, b, c) => Op::SetRec(a, b, c),
        Op::Cons(a, b, c) => Op::Cons(a, b, c),
        Op::Car(a, b) => Op::Car(a, b),
        Op::Cdr(a, b) => Op::Cdr(a, b),
        Op::Cell(a, b) => Op::Cell(a, b),
        Op::GetCell(a, b) => Op::GetCell(a, b),
        Op::SetCell(a, b) => Op::SetCell(a, b),
    })
}

fn convert_operand(
    o: Operand<String>,
    pos: usize,
    label_positions: &HashMap<String, usize>,
) -> AssemblerResult<Operand<isize>> {
    match o {
        Operand::R(register) => Ok(Operand::R(register)),
        Operand::I(label) => Ok(Operand::I(convert_label(label, pos, label_positions)?)),
    }
}

fn convert_label(
    label: String,
    pos: usize,
    label_positions: &HashMap<String, usize>,
) -> AssemblerResult<isize> {
    label_positions
        .get(&label)
        .ok_or(AssemblerError::UndefinedLabel(label))
        .map(|&label_pos| label_pos as isize - pos as isize)
}

#[derive(Debug, Clone)]
enum AsmOp {
    Label(String),
    Opcode(Op<String>),
}

impl AsmOp {
    fn into_op(self) -> Option<Op<String>> {
        match self {
            AsmOp::Opcode(op) => Some(op),
            _ => None,
        }
    }
}

impl std::fmt::Display for Assembler {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for op in &self.source {
            write!(f, "{}", op)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for AsmOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AsmOp::Label(l) => writeln!(f, "{}:", l),
            AsmOp::Opcode(op) => writeln!(f, "    {}", op),
        }
    }
}

impl std::fmt::Display for Op<String> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Op::Term => write!(f, "TERM"),
            Op::Nop => write!(f, "NOP"),
            Op::Debug(s, r) => write!(f, "DEBUG {:?} r{}", s, r),
            Op::Const(r, x) => write!(f, "r{} := {:?}", r, x),
            Op::Copy(r, x) => write!(f, "r{} := r{}", r, x),
            Op::Inc(r) => write!(f, "r{} += 1", r),
            Op::Dec(r) => write!(f, "r{} -= 1", r),
            Op::Add(r, a, b) => write!(f, "r{} := r{} + {}", r, a, b),
            Op::Sub(r, a, b) => write!(f, "r{} := r{} - {}", r, a, b),
            Op::Mul(r, a, b) => write!(f, "r{} := r{} * {}", r, a, b),
            Op::Div(r, a, b) => write!(f, "r{} := {} / {}", r, a, b),
            Op::Equal(r, a, b) => write!(f, "r{} := r{} == {}", r, a, b),
            Op::Uneq(r, a, b) => write!(f, "r{} := r{} != {}", r, a, b),
            Op::Less(r, a, b) => write!(f, "r{} := {} < {}", r, a, b),
            Op::LessEq(r, a, b) => write!(f, "r{} := {} <= {}", r, a, b),
            Op::Not(r, a) => write!(f, "r{} := !{}", r, a),
            Op::Jmp(z) => write!(f, "JMP {}", z),
            Op::JmpFar(code_slice) => write!(f, "JMP-FAR {:p}", *code_slice),
            Op::JmpCond(z, r) => write!(f, "IF r{} JMP {}", r, z),
            Op::LoadLabel(r, z) => write!(f, "r{} := {}", r, z),
            Op::Alloc(r, n) => write!(f, "r{} := [_; {}]", r, n),
            Op::GetRec(r, rec, i) => write!(f, "r{} := r{}[{}]", r, rec, i),
            Op::SetRec(rec, i, x) => write!(f, "r{}[{}] := {}", rec, i, x),
            Op::Cons(r, a, b) => write!(f, "r{} := [{}, {}]", r, a, b),
            Op::Car(r, a) => write!(f, "r{} := r{}[0]", r, a),
            Op::Cdr(r, a) => write!(f, "r{} := r{}[1]", r, a),
            Op::Cell(r, x) => write!(f, "r{} := [{}]", r, x),
            Op::GetCell(r, c) => write!(f, "r{} := @{}", r, c),
            Op::SetCell(c, x) => write!(f, "@{} := {}", c, x),
            _ => write!(f, "<op>"),
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Display for Operand<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Operand::R(r) => write!(f, "r{}", r),
            Operand::I(i) => write!(f, "{:?}", i),
        }
    }
}
