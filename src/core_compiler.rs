use crate::bytecode_builder::{Block, Builder};
use crate::primitive_value::PrimitiveValue;
use crate::virtual_machine::{Op, Register};

#[derive(Debug, Clone)]
pub enum AtomicExpression {
    Undefined,
    Nil,
    Integer(i64),
    Lambda(Vec<String>, Box<Expression>),
}

#[derive(Debug, Clone)]
pub enum ComplexExpression {
    Apply(AtomicExpression, Vec<AtomicExpression>),
    If(AtomicExpression, Box<Expression>, Box<Expression>),
    Atomic(AtomicExpression),
}

#[derive(Debug, Clone)]
pub enum Expression {
    Let(String, ComplexExpression, Box<Expression>),
    Atomic(AtomicExpression),
    Complex(ComplexExpression),
}

pub struct Compiler {
    variables_in_registers: Vec<String>,
    total_registers_used: usize,
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            variables_in_registers: vec![],
            total_registers_used: 0,
        }
    }

    pub fn compile(&mut self, exp: Expression) -> Vec<Op> {
        let final_block = Block::new();
        final_block.terminate();
        let block = self.compile_expr(exp, Linkage::branch(0, &final_block));
        Builder::build(&block).unwrap()
    }

    fn compile_expr(&mut self, exp: Expression, linkage: Linkage) -> Block {
        match exp {
            Expression::Let(varname, vardef, body) => {
                self.compile_let(varname, vardef, *body, linkage)
            }
            Expression::Atomic(aexp) => self.compile_atomic(aexp, linkage),
            Expression::Complex(cexp) => self.compile_complex(cexp, linkage),
        }
    }

    fn compile_let(
        &mut self,
        varname: String,
        vardef: ComplexExpression,
        body: Expression,
        linkage: Linkage,
    ) -> Block {
        let register = self.assign_register();

        let connector = Block::new();
        let def_block = self.compile_complex(vardef, Linkage::branch(register, &connector));

        self.begin_scope(&varname, register);
        let body_block = self.compile_expr(body, linkage);
        self.end_scope(&varname);

        connector.branch(&body_block);
        def_block
    }

    fn compile_complex(&mut self, cexp: ComplexExpression, linkage: Linkage) -> Block {
        use ComplexExpression::*;
        match cexp {
            If(cond, then_exp, else_exp) => {
                let cond_register = self.assign_register();
                let cond_block = self.compile_atomic(cond, Linkage::none(cond_register));

                let then_block = self.compile_expr(*then_exp, linkage);
                let else_block = self.compile_expr(*else_exp, linkage);

                cond_block.branch_conditional(cond_register, &then_block, &else_block);
                cond_block
            }
            Atomic(aexp) => self.compile_atomic(aexp, linkage),
            _ => unimplemented!(),
        }
    }

    fn compile_atomic(&mut self, aexp: AtomicExpression, linkage: Linkage) -> Block {
        use AtomicExpression::*;
        let block = Block::new();
        match aexp {
            Undefined => block.add_op(Op::Const(
                linkage.output_register(),
                PrimitiveValue::Undefined,
            )),
            Nil => block.add_op(Op::Const(linkage.output_register(), PrimitiveValue::Nil)),
            Integer(i) => block.add_op(Op::Const(linkage.output_register(), i.into())),
            _ => unimplemented!(),
        }
        linkage.compile(block)
    }

    fn assign_register(&mut self) -> Register {
        self.variables_in_registers.len() as Register
    }

    fn begin_scope(&mut self, varname: &str, register: Register) {
        assert_eq!(register as usize, self.variables_in_registers.len());
        self.variables_in_registers.push(varname.to_string());
        self.total_registers_used = self
            .total_registers_used
            .max(self.variables_in_registers.len());
    }

    fn end_scope(&mut self, varname: &str) {
        assert_eq!(self.variables_in_registers.pop().unwrap(), varname);
    }
}

#[derive(Copy, Clone)]
struct Linkage<'a> {
    output_register: Register,
    branch_to: Option<&'a Block>,
}

impl<'a> Linkage<'a> {
    fn branch(output_register: Register, to: &'a Block) -> Self {
        Linkage {
            output_register,
            branch_to: Some(to),
        }
    }

    fn none(output_register: Register) -> Self {
        Linkage {
            output_register,
            branch_to: None,
        }
    }

    fn compile(&self, block: Block) -> Block {
        if let Some(to) = self.branch_to {
            block.branch(to)
        }
        block
    }

    fn output_register(&self) -> Register {
        self.output_register
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_machine::Operand;

    #[test]
    fn test_linkage_none() {
        let linkage = Linkage::none(4);
        assert_eq!(linkage.compile(Block::new()), Block::new());
    }

    #[test]
    fn test_linkage_branch() {
        let target = Block::new();
        target.terminate();
        let linkage = Linkage::branch(1, &target);

        let expect = Block::new();
        expect.branch(&target);

        assert_eq!(linkage.compile(Block::new()), expect);
    }

    fn compile_atomic_constant(
        register: Register,
        aexp: AtomicExpression,
        value: PrimitiveValue,
    ) -> (Block, Block) {
        let actual = Compiler::new().compile_atomic(aexp, Linkage::none(register));
        let expect = Block::new();
        expect.add_op(Op::Const(register, value));
        (actual, expect)
    }

    #[test]
    fn compile_atomic_undefined() {
        let (actual, expect) =
            compile_atomic_constant(31, AtomicExpression::Undefined, PrimitiveValue::Undefined);
        assert_eq!(actual, expect);
    }

    #[test]
    fn compile_atomic_nil() {
        let (actual, expect) =
            compile_atomic_constant(100, AtomicExpression::Nil, PrimitiveValue::Nil);
        assert_eq!(actual, expect);
    }

    #[test]
    fn compile_atomic_int() {
        let (actual, expect) = compile_atomic_constant(
            100,
            AtomicExpression::Integer(42),
            PrimitiveValue::Integer(42),
        );
        assert_eq!(actual, expect);
    }

    #[test]
    fn compile_complex_if() {
        let register = 1;
        let actual = Compiler::new().compile_complex(
            ComplexExpression::If(
                AtomicExpression::Integer(1),
                Box::new(Expression::Atomic(AtomicExpression::Integer(2))),
                Box::new(Expression::Atomic(AtomicExpression::Integer(3))),
            ),
            Linkage::none(register),
        );
        let expect = Block::new();
        let then_block = Block::new();
        let else_block = Block::new();
        expect.add_op(Op::Const(0, PrimitiveValue::Integer(1)));
        expect.branch_conditional(0, &then_block, &else_block);
        then_block.add_op(Op::Const(register, PrimitiveValue::Integer(2)));
        else_block.add_op(Op::Const(register, PrimitiveValue::Integer(3)));
        assert_eq!(actual, expect);
    }

    #[test]
    fn compile_complex_atomic() {
        let register = 1;
        let aexp = AtomicExpression::Integer(-7);
        let linkage = Linkage::none(register);
        let actual =
            Compiler::new().compile_complex(ComplexExpression::Atomic(aexp.clone()), linkage);
        let expect = Compiler::new().compile_atomic(aexp.clone(), linkage);
        assert_eq!(actual, expect);
    }

    #[test]
    fn constant() {
        let mut compiler = Compiler::new();
        let code = compiler.compile(Expression::Atomic(AtomicExpression::Integer(42)));
        assert_eq!(code, vec![Op::Const(0, 42.into()), Op::Term])
    }

    #[test]
    fn if_expr() {
        let mut compiler = Compiler::new();
        let code = compiler.compile(Expression::Complex(ComplexExpression::If(
            AtomicExpression::Nil,
            Box::new(Expression::Atomic(AtomicExpression::Integer(1))),
            Box::new(Expression::Atomic(AtomicExpression::Integer(2))),
        )));

        assert_eq!(
            code,
            vec![
                Op::Const(0, PrimitiveValue::Nil),
                Op::JmpCond(Operand::I(3), 0),
                Op::Const(0, 2.into()),
                Op::Term,
                Op::Const(0, 1.into()),
                Op::Jmp(Operand::I(-2)),
            ]
        );
    }
}
