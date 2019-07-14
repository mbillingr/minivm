use crate::virtual_machine::{Op, Operand, Register};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

type BlockRef = Rc<Block>;
type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, PartialEq)]
enum Error {
    UnterminatedBlock(BlockRef),
}

#[derive(Debug, PartialEq)]
enum TermOp {
    Term,
    Branch(BlockRef),
    CondBranch(Register, BlockRef, BlockRef),
}

struct Builder {
    code: Vec<Op>,
    labels: HashMap<usize, usize>,
}

impl Builder {
    fn new() -> Self {
        Builder {
            code: vec![],
            labels: HashMap::new(),
        }
    }

    fn build(block: &BlockRef) -> Result<Vec<Op>> {
        let mut finalizer = Self::new();
        finalizer.append_block(block)?;
        Ok(finalizer.code)
    }

    fn append_block(&mut self, block: &BlockRef) -> Result<()> {
        if self.labels.contains_key(&Block::addr(&block)) {
            return Ok(());
        }
        Block::verify(&block)?;

        self.labels.insert(Block::addr(&block), self.code.len());
        self.code.extend_from_slice(block.ops.borrow().as_slice());

        match &*block.last.borrow() {
            Some(TermOp::Term) => self.code.push(Op::Term),
            Some(TermOp::Branch(dst)) => {
                if let Some(l) = self.labels.get(&Block::addr(&dst)) {
                    self.code
                        .push(Op::Jmp(Operand::I(*l as isize - self.code.len() as isize)))
                } else {
                    self.append_block(dst)?;
                }
            }
            Some(TermOp::CondBranch(reg, a, b)) => {
                let block_a = self.labels.get(&Block::addr(&a));
                let block_b = self.labels.get(&Block::addr(&b));

                match (block_a, block_b) {
                    (Some(la), Some(lb)) => {
                        self.code.push(Op::JmpCond(
                            Operand::I(*la as isize - self.code.len() as isize),
                            *reg,
                        ));
                        self.code
                            .push(Op::Jmp(Operand::I(*lb as isize - self.code.len() as isize)));
                    }
                    (Some(la), None) => {
                        self.code.push(Op::JmpCond(
                            Operand::I(*la as isize - self.code.len() as isize),
                            *reg,
                        ));
                        self.append_block(b)?;
                    }
                    (None, Some(lb)) => {
                        self.code.push(Op::JmpCond(Operand::I(2), *reg));
                        self.code
                            .push(Op::Jmp(Operand::I(*lb as isize - self.code.len() as isize)));
                        self.append_block(a)?;
                    }
                    (None, None) => {
                        let jmp_pos = self.code.len();
                        self.code.push(Op::Nop); // placeholder
                        self.append_block(b)?;
                        self.append_block(a)?;
                        let la = self.labels.get(&Block::addr(&a)).unwrap();
                        self.code[jmp_pos] =
                            Op::JmpCond(Operand::I(*la as isize - jmp_pos as isize), *reg);
                    }
                }
            }
            None => unreachable!(),
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq)]
struct Block {
    ops: RefCell<Vec<Op>>,
    last: RefCell<Option<TermOp>>,
}

impl Block {
    fn new() -> BlockRef {
        Rc::new(Block {
            ops: RefCell::new(vec![]),
            last: RefCell::new(None),
        })
    }

    fn add_op(&self, op: Op) {
        assert!(self.last.borrow().is_none());
        self.ops.borrow_mut().push(op);
    }

    fn branch(&self, target: &BlockRef) {
        assert!(self.last.borrow().is_none());
        *self.last.borrow_mut() = Some(TermOp::Branch(target.clone()));
    }

    fn branch_conditional(&self, r: Register, then_block: &BlockRef, else_block: &BlockRef) {
        assert!(self.last.borrow().is_none());
        *self.last.borrow_mut() = Some(TermOp::CondBranch(
            r,
            then_block.clone(),
            else_block.clone(),
        ));
    }

    fn terminate(&self) {
        assert!(self.last.borrow().is_none());
        *self.last.borrow_mut() = Some(TermOp::Term);
    }

    fn verify(blk: &BlockRef) -> Result<()> {
        if blk.last.borrow().is_none() {
            return Err(Error::UnterminatedBlock(blk.clone()));
        }
        Ok(())
    }

    fn addr(blk: &BlockRef) -> usize {
        blk.as_ref() as *const _ as usize
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
