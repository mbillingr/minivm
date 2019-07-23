use crate::bytecode_builder::{Block, Builder};
use crate::primitive_value::PrimitiveValue;
use crate::virtual_machine::{Op, Operand, Register};

#[derive(Debug, Clone)]
pub enum Expression {
    Let(String, ComplexExpression, Box<Expression>),
    Atomic(AtomicExpression),
    Complex(ComplexExpression),
}

#[derive(Debug, Clone)]
pub enum ComplexExpression {
    Apply(AtomicExpression, Vec<AtomicExpression>),
    ApplyPrimitive(PrimitiveFunction, Vec<AtomicExpression>),
    If(AtomicExpression, Box<Expression>, Box<Expression>),
    Atomic(AtomicExpression),
}

#[derive(Debug, Clone)]
pub enum AtomicExpression {
    Undefined,
    Nil,
    Integer(i64),
    Variable(String),
    Lambda(Vec<String>, Box<Expression>),
}

#[derive(Debug, Clone)]
pub enum PrimitiveFunction {
    Add,
    Mul,
}

impl From<AtomicExpression> for ComplexExpression {
    fn from(ax: AtomicExpression) -> Self {
        ComplexExpression::Atomic(ax)
    }
}

impl From<AtomicExpression> for Expression {
    fn from(ax: AtomicExpression) -> Self {
        Expression::Atomic(ax)
    }
}

impl From<ComplexExpression> for Expression {
    fn from(cx: ComplexExpression) -> Self {
        Expression::Complex(cx)
    }
}

impl From<AtomicExpression> for Box<Expression> {
    fn from(ax: AtomicExpression) -> Self {
        Box::new(Expression::Atomic(ax))
    }
}

impl From<ComplexExpression> for Box<Expression> {
    fn from(cx: ComplexExpression) -> Self {
        Box::new(Expression::Complex(cx))
    }
}

const INDENT: usize = 2;

trait PrettyPrint {
    fn to_pretty_string(&self, indent: usize) -> String;

    fn println(&self) {
        println!("{}", self.to_pretty_string(0));
    }
}

impl PrettyPrint for AtomicExpression {
    fn to_pretty_string(&self, indent: usize) -> String {
        match self {
            AtomicExpression::Undefined => format!("undefined"),
            AtomicExpression::Nil => format!("'()"),
            AtomicExpression::Variable(v) => v.clone(),
            AtomicExpression::Integer(i) => format!("{}", i),
            AtomicExpression::Lambda(params, body) => format!(
                "(lambda ({})\n{}{})",
                params.join(" "),
                " ".repeat(indent + INDENT),
                body.to_pretty_string(indent + INDENT)
            ),
        }
    }
}

impl PrettyPrint for Expression {
    fn to_pretty_string(&self, indent: usize) -> String {
        match self {
            Expression::Atomic(ax) => ax.to_pretty_string(indent),
            Expression::Complex(cx) => cx.to_pretty_string(indent),
            Expression::Let(var, def, body) => format!(
                "(let {} {}\n{}{})",
                var,
                def.to_pretty_string(indent + 6 + var.len()),
                " ".repeat(indent + INDENT),
                body.to_pretty_string(indent + INDENT)
            ),
        }
    }
}

impl PrettyPrint for ComplexExpression {
    fn to_pretty_string(&self, indent: usize) -> String {
        fn print_apply(argstrs: Vec<String>, indent: usize) -> String {
            if argstrs.iter().any(|s| s.contains('\n')) {
                let argstrs: Vec<_> = argstrs
                    .into_iter()
                    .map(|s| s.replace("\n", &("\n".to_string() + &" ".repeat(indent + 1))))
                    .collect();
                format!(
                    "({})",
                    argstrs.join(&("\n".to_string() + &" ".repeat(indent + 1)))
                )
            } else {
                format!("({})", argstrs.join(" "))
            }
        }

        match self {
            ComplexExpression::Apply(func, args) => print_apply(
                std::iter::once(func)
                    .chain(args)
                    .map(|a| a.to_pretty_string(0))
                    .collect(),
                indent,
            ),
            ComplexExpression::ApplyPrimitive(op, args) => print_apply(
                std::iter::once(op.to_pretty_string(0))
                    .chain(args.iter().map(|a| a.to_pretty_string(0)))
                    .collect(),
                indent,
            ),
            ComplexExpression::Atomic(ax) => ax.to_pretty_string(indent),
            _ => format!("<Complex>"),
        }
    }
}

impl PrettyPrint for PrimitiveFunction {
    fn to_pretty_string(&self, _indent: usize) -> String {
        match self {
            PrimitiveFunction::Add => format!("+"),
            PrimitiveFunction::Mul => format!("*"),
        }
    }
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

        self.begin_scope(&varname, register);
        let body_block = self.compile_expr(body, linkage);
        self.end_scope(&varname);

        self.compile_complex(vardef, Linkage::branch(register, &body_block))
    }

    fn compile_complex(&mut self, cexp: ComplexExpression, linkage: Linkage) -> Block {
        use ComplexExpression::*;
        match cexp {
            Atomic(aexp) => self.compile_atomic(aexp, linkage),
            If(cond, then_exp, else_exp) => {
                let cond_register = self.assign_register();
                let cond_block = self.compile_atomic(cond, Linkage::none(cond_register));

                let then_block = self.compile_expr(*then_exp, linkage);
                let else_block = self.compile_expr(*else_exp, linkage);

                cond_block.branch_conditional(cond_register, &then_block, &else_block);
                cond_block
            }
            ApplyPrimitive(prim, args) => {
                let mut regs = vec![];
                let last_block = Block::new();
                let mut block = last_block.clone();
                for arg in args {
                    let reg = self.assign_register();
                    regs.push(reg);
                    self.begin_scope("argtmp", reg);
                    block = self.compile_atomic(arg, Linkage::branch(reg, &block));
                }

                let r_out = linkage.output_register();

                match prim {
                    PrimitiveFunction::Add => {
                        last_block.add_op(Op::Add(r_out, regs[0], Operand::R(regs[1])));
                        for r in regs.iter().skip(2) {
                            last_block.add_op(Op::Add(r_out, r_out, Operand::R(*r)));
                        }
                    }

                    PrimitiveFunction::Mul => {
                        last_block.add_op(Op::Mul(r_out, regs[0], Operand::R(regs[1])));
                        for r in regs.iter().skip(2) {
                            last_block.add_op(Op::Mul(r_out, r_out, Operand::R(*r)));
                        }
                    }
                }

                for _ in regs {
                    self.end_scope("argtmp");
                }

                linkage.compile(last_block);
                block
            }
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
            Variable(name) => block.add_op(Op::Copy(
                linkage.output_register(),
                self.lookup(&name).expect("undefined variable"),
            )),
            Lambda(params, body) => {
                let mut func_compiler = Compiler::new();

                for p in &params {
                    let r = func_compiler.assign_register();
                    func_compiler.begin_scope(p, r);
                }

                let kill = Block::new();
                kill.terminate();
                let body_block = func_compiler.compile_expr(*body, Linkage::branch(0, &kill));

                for p in &params {
                    func_compiler.end_scope(p);
                }

                block.set_function(linkage.output_register(), body_block);
            }
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

    fn lookup(&mut self, varname: &str) -> Option<Register> {
        self.variables_in_registers
            .iter()
            .position(|var| var == varname)
            .map(|idx| idx as Register)
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

    fn function() -> Self {
        unimplemented!()
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
    use crate::bytecode_builder::Builder;
    use crate::memory::store_code_block;
    use crate::virtual_machine::eval;
    use crate::virtual_machine::Operand;

    #[test]
    fn pretty_print() {
        let prog = Expression::Let(
            "func".into(),
            AtomicExpression::Lambda(
                vec!["x".to_string()],
                Box::new(Expression::Complex(ComplexExpression::ApplyPrimitive(
                    PrimitiveFunction::Mul,
                    vec![
                        AtomicExpression::Variable("x".to_string()),
                        AtomicExpression::Lambda(
                            vec!["x".to_string()],
                            Box::new(Expression::Complex(ComplexExpression::ApplyPrimitive(
                                PrimitiveFunction::Mul,
                                vec![
                                    AtomicExpression::Variable("x".to_string()),
                                    AtomicExpression::Variable("x".to_string()),
                                ],
                            ))),
                        ),
                    ],
                ))),
            )
            .into(),
            ComplexExpression::Apply(
                AtomicExpression::Variable("func".into()),
                vec![AtomicExpression::Integer(42)],
            )
            .into(),
        );

        let expected = "(let func (lambda (x)
            (*
             x
             (lambda (x)
               (* x x))))
  (func 42))";

        assert_eq!(prog.to_pretty_string(0), expected);
    }

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
    fn compile_atomic_lambda() {
        let done = Block::new();
        done.terminate();
        let actual = Compiler::new().compile_atomic(
            AtomicExpression::Lambda(
                vec!["x".to_string()],
                Box::new(Expression::Complex(ComplexExpression::ApplyPrimitive(
                    PrimitiveFunction::Mul,
                    vec![
                        AtomicExpression::Variable("x".to_string()),
                        AtomicExpression::Variable("x".to_string()),
                    ],
                ))),
            ),
            Linkage::branch(0, &done),
        );
        let expect = Block::new();
        expect.add_op(Op::Const(0, PrimitiveValue::Integer(42)));

        let code = store_code_block(Builder::build(&actual).unwrap());
        let result = eval(code);

        assert!(if let PrimitiveValue::CodeBlock(_) = result {
            true
        } else {
            false
        });
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
    fn compile_complex_primitive_add() {
        use AtomicExpression::*;
        use ComplexExpression::*;
        use PrimitiveFunction::*;

        let done = Block::new();
        done.terminate();

        let actual = Compiler::new().compile_expr(
            Expression::Complex(ApplyPrimitive(
                Add,
                vec![Integer(1), Integer(2), Integer(3), Integer(4)],
            )),
            Linkage::branch(0, &done),
        );

        let code = store_code_block(Builder::build(&actual).unwrap());
        assert_eq!(eval(code), 10.into());
    }

    #[test]
    fn compile_expression_let() {
        let actual = Compiler::new().compile_expr(
            Expression::Let(
                "x".to_string(),
                ComplexExpression::Atomic(AtomicExpression::Integer(12345)),
                Box::new(Expression::Atomic(AtomicExpression::Variable(
                    "x".to_string(),
                ))),
            ),
            Linkage::none(3),
        );
        let expect = Block::new();
        let body = Block::new();
        expect.add_op(Op::Const(0, PrimitiveValue::Integer(12345)));
        expect.branch(&body);
        body.add_op(Op::Copy(3, 0));
        assert_eq!(actual, expect);
    }

    #[test]
    fn compile_expression_nested_let() {
        use AtomicExpression::*;
        use ComplexExpression::*;

        let done = Block::new();
        done.terminate();

        let actual = Compiler::new().compile_expr(
            Expression::Let(
                "x".to_string(),
                Atomic(Integer(10)),
                Box::new(Expression::Let(
                    "y".to_string(),
                    Atomic(Variable("x".to_string())),
                    Box::new(Expression::Let(
                        "z".to_string(),
                        Atomic(Variable("y".to_string())),
                        Box::new(Expression::Atomic(Variable("z".to_string()))),
                    )),
                )),
            ),
            Linkage::branch(0, &done),
        );

        let code = store_code_block(Builder::build(&actual).unwrap());
        assert_eq!(eval(code), 10.into());
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
