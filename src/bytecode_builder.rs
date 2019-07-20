//! This module provides a `Block` based `Builder` for conveniently assembling bytecode.
//!
//! Directly creating bytecode instructions is cumbersome because jump targets or offsets are
//! not known in advance. This instruction `Block`s in this module define a call flow graph.
//! No jumps may happen into or out of the middle of a `Block`. Every block must be terminated
//! with a branch (jump) or other terminal instruction.
//! Starting from an entry block, the `Builder` traverses the `Block` graph and automatically
//! generates bytecode.

use crate::virtual_machine::{Op, Operand, Register};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, PartialEq)]
pub enum Error {
    UnterminatedBlock(Block),
}

#[derive(Debug, Clone, PartialEq)]
enum TermOp {
    Term,
    Branch(Block),
    CondBranch(Register, Block, Block),
}

pub struct Builder {
    code: Vec<Op>,
    labels: HashMap<usize, isize>,
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            code: vec![],
            labels: HashMap::new(),
        }
    }

    pub fn build(block: &Block) -> Result<Vec<Op>> {
        let mut finalizer = Self::new();
        finalizer.append_block(block)?;
        Ok(finalizer.code)
    }

    fn append_block(&mut self, block: &Block) -> Result<()> {
        if self.labels.contains_key(&Block::id(&block)) {
            return Ok(());
        }
        Block::verify(&block)?;

        self.labels.insert(Block::id(&block), self.current_addr());

        if let Some((r, fnblock)) = block.function() {
            let func = Builder::build(&fnblock)?;
            self.code.push(Op::LoadLabel(r, 2));
            self.code.push(Op::Jmp(Operand::I(1 + func.len() as isize)));
            self.code.extend(func);
        } else {
            self.code.extend(block.get_ops());
        }

        match block.get_terminal_op().unwrap() {
            TermOp::Term => self.append_termination(),
            TermOp::Branch(dst) => self.append_branch(&dst),
            TermOp::CondBranch(reg, a, b) => self.append_conditional_branch(reg, &a, &b),
        }
    }

    fn append_termination(&mut self) -> Result<()> {
        self.code.push(Op::Term);
        Ok(())
    }

    fn append_branch(&mut self, dst: &Block) -> Result<()> {
        if let Some(dst_addr) = self.labels.get(&dst.id()) {
            let offset = dst_addr - self.current_addr();
            self.code.push(Op::Jmp(Operand::I(offset)));
            Ok(())
        } else {
            self.append_block(dst)
        }
    }

    fn append_conditional_branch(
        &mut self,
        reg: Register,
        then_block: &Block,
        else_block: &Block,
    ) -> Result<()> {
        let found_then_block = self.labels.get(&then_block.id());
        let found_else_block = self.labels.get(&else_block.id());

        match (found_then_block, found_else_block) {
            (Some(then_addr), Some(_)) => {
                self.code.push(Op::JmpCond(
                    Operand::I(then_addr - self.current_addr()),
                    reg,
                ));
                self.append_branch(else_block)
            }
            (Some(then_addr), None) => {
                self.code.push(Op::JmpCond(
                    Operand::I(then_addr - self.current_addr()),
                    reg,
                ));
                self.append_block(&else_block)
            }
            (None, Some(_)) => {
                self.code.push(Op::JmpCond(Operand::I(2), reg));
                self.append_branch(else_block)?;
                self.append_block(&then_block)
            }
            (None, None) => {
                let jmp_pos = self.current_addr();
                self.code.push(Op::Nop); // placeholder
                self.append_block(&else_block)?;
                self.append_block(&then_block)?;
                let then_addr = self.labels.get(&Block::id(&then_block)).unwrap();
                self.code[jmp_pos as usize] = Op::JmpCond(Operand::I(then_addr - jmp_pos), reg);
                Ok(())
            }
        }
    }

    fn current_addr(&self) -> isize {
        self.code.len() as isize
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    data: Rc<RefCell<BlockData>>,
}

impl Block {
    pub fn new() -> Block {
        Block {
            data: Rc::new(RefCell::new(BlockData::new())),
        }
    }

    pub fn with(&self, body: impl FnOnce(&mut BlockData)) {
        let mut data = self.data.borrow_mut();
        body(&mut data)
    }

    fn has_terminal_instruction(&self) -> bool {
        (*self.data).borrow().has_terminal_instruction()
    }

    pub fn add_op(&self, op: Op) {
        let mut data = self.data.borrow_mut();
        assert!(!data.has_terminal_instruction());
        data.add_op(op)
    }

    pub fn set_function(&self, r: Register, fnblock: Block) {
        let mut data = self.data.borrow_mut();
        data.set_function(r, fnblock)
    }

    pub fn branch(&self, target: &Block) {
        let mut data = self.data.borrow_mut();
        assert!(!data.has_terminal_instruction());
        data.branch(target)
    }

    pub fn branch_conditional(&self, r: Register, then_block: &Block, else_block: &Block) {
        let mut data = self.data.borrow_mut();
        assert!(!data.has_terminal_instruction());
        data.branch_conditional(r, then_block, else_block)
    }

    pub fn terminate(&self) {
        let mut data = self.data.borrow_mut();
        assert!(!data.has_terminal_instruction());
        data.terminate()
    }

    pub fn function(&self) -> Option<(Register, Block)> {
        self.data.borrow().function().cloned()
    }

    fn get_ops(&self) -> Vec<Op> {
        (*self.data).borrow().get_ops().to_vec()
    }

    fn get_terminal_op(&self) -> Option<TermOp> {
        (*self.data).borrow().get_terminal_op().cloned()
    }

    fn verify(&self) -> Result<()> {
        if !self.has_terminal_instruction() {
            return Err(Error::UnterminatedBlock(self.clone()));
        }
        Ok(())
    }

    fn id(&self) -> usize {
        self.data.as_ref() as *const _ as usize
    }
}

#[derive(Debug, PartialEq)]
pub struct BlockData {
    ops: Vec<Op>,
    last: Option<TermOp>,
    func: Option<(Register, Block)>,
}

impl BlockData {
    fn new() -> Self {
        BlockData {
            ops: vec![],
            last: None,
            func: None,
        }
    }

    fn has_terminal_instruction(&self) -> bool {
        self.last.is_some()
    }

    fn add_op(&mut self, op: Op) {
        self.ops.push(op);
    }

    pub fn set_function(&mut self, r: Register, fnblock: Block) {
        self.func = Some((r, fnblock));
    }

    fn branch(&mut self, target: &Block) {
        self.last = Some(TermOp::Branch(target.clone()));
    }

    fn branch_conditional(&mut self, r: Register, then_block: &Block, else_block: &Block) {
        self.last = Some(TermOp::CondBranch(
            r,
            then_block.clone(),
            else_block.clone(),
        ));
    }

    pub fn function(&self) -> Option<&(Register, Block)> {
        self.func.as_ref()
    }

    fn get_ops(&self) -> &[Op] {
        self.ops.as_slice()
    }

    fn get_terminal_op(&self) -> Option<&TermOp> {
        self.last.as_ref()
    }

    fn terminate(&mut self) {
        self.last = Some(TermOp::Term);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_block() {
        let block = Block::new();
        block.terminate();
        assert_eq!(Builder::build(&block), Ok(vec![Op::Term]));
    }

    #[test]
    fn invalid_block() {
        let block = Block::new();
        block.add_op(Op::Nop);
        assert_eq!(Builder::build(&block), Err(Error::UnterminatedBlock(block)));
    }

    #[test]
    fn spurious_branch() {
        let entry = Block::new();
        let exit = Block::new();
        entry.branch(&exit);
        exit.terminate();
        assert_eq!(Builder::build(&entry), Ok(vec![Op::Term]));
    }

    #[test]
    fn infinite_loop() {
        let entry = Block::new();
        let again = Block::new();
        entry.add_op(Op::Nop);
        entry.branch(&again);
        again.add_op(Op::Nop);
        again.branch(&entry);
        assert_eq!(
            Builder::build(&entry),
            Ok(vec![Op::Nop, Op::Nop, Op::Jmp(Operand::I(-2))])
        );
    }

    #[test]
    fn two_way_branch() {
        let entry = Block::new();
        let yes = Block::new();
        let no = Block::new();
        entry.branch_conditional(0, &yes, &no);
        yes.add_op(Op::Const(0, 1.into()));
        yes.terminate();
        no.add_op(Op::Const(0, 2.into()));
        no.terminate();
        assert_eq!(
            Builder::build(&entry),
            Ok(vec![
                Op::JmpCond(Operand::I(3), 0),
                Op::Const(0, 2.into()),
                Op::Term,
                Op::Const(0, 1.into()),
                Op::Term
            ])
        );
    }

    #[test]
    fn reuse_yes_branch() {
        let entry = Block::new();
        let yes = Block::new();
        let no1 = Block::new();
        let no2 = Block::new();
        entry.branch_conditional(0, &yes, &no1);
        yes.add_op(Op::Const(0, 1.into()));
        yes.terminate();
        no1.branch_conditional(1, &yes, &no2);
        no2.add_op(Op::Const(0, 2.into()));
        no2.terminate();
        assert_eq!(
            Builder::build(&entry),
            Ok(vec![
                Op::JmpCond(Operand::I(4), 0),
                Op::JmpCond(Operand::I(3), 1),
                Op::Const(0, 2.into()),
                Op::Term,
                Op::Const(0, 1.into()),
                Op::Term
            ])
        );
    }

    #[test]
    fn reuse_no_branch() {
        let entry = Block::new();
        let yes1 = Block::new();
        let yes2 = Block::new();
        let no = Block::new();
        entry.branch_conditional(0, &yes1, &no);
        yes1.branch_conditional(1, &yes2, &no);
        yes2.add_op(Op::Const(0, 1.into()));
        yes2.terminate();
        no.add_op(Op::Const(0, 2.into()));
        no.terminate();
        assert_eq!(
            Builder::build(&entry),
            Ok(vec![
                Op::JmpCond(Operand::I(3), 0),
                Op::Const(0, 2.into()),
                Op::Term,
                Op::JmpCond(Operand::I(2), 1),
                Op::Jmp(Operand::I(-3)),
                Op::Const(0, 1.into()),
                Op::Term
            ])
        );
    }
}
