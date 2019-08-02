use crate::memory::store_code_block;
use crate::primitive_value::PrimitiveValue;
use crate::ssa_builder;
use crate::ssa_builder::TranslationUnit;
use crate::virtual_machine as vm;
use std::collections::HashMap;

type Block = ssa_builder::Block<PrimitiveValue>;
type Var = ssa_builder::Var<PrimitiveValue>;

struct Prog {
    function_definitions: Vec<FunctionDefinition>,
    body: Expr,
}

struct FunctionDefinition {
    name: String,
    params: Vec<String>,
    body: Expr,
}

enum Expr {
    Let(String, Cexp, Box<Expr>),
    LetMut(String, Cexp, Box<Expr>),
    Atomic(Aexp),
    Complex(Cexp),
}

enum Cexp {
    Atomic(Aexp),
    ApplyPrimitive(PrimOp, Vec<Aexp>),
    ApplyStatic(String, Vec<Aexp>),
    Apply(Aexp, Vec<Aexp>),
    If(Aexp, Box<Expr>, Box<Expr>),
}

enum Aexp {
    Undefined,
    Integer(i64),
    Var(String),
    Function(String),
}

impl From<Aexp> for Cexp {
    fn from(aexp: Aexp) -> Self {
        Cexp::Atomic(aexp)
    }
}

impl From<Aexp> for Expr {
    fn from(aexp: Aexp) -> Self {
        Expr::Atomic(aexp)
    }
}

impl From<Cexp> for Expr {
    fn from(cexp: Cexp) -> Self {
        Expr::Complex(cexp)
    }
}

enum PrimOp {
    Add,
    Sub,
    Mul,
    Div,

    Equal,
}

impl PrimOp {
    fn compile(&self, args: &[Var], block: &Block) -> Var {
        match args.len() {
            0 => self.invariant(block),
            1 => self.singular(&args[0], block),
            2 => self.binop(&args[0], &args[1], block),
            _ => {
                let mut aggregate = self.binop(&args[0], &args[1], block);
                for a in &args[2..] {
                    aggregate = self.binop(&aggregate, a, block);
                }
                aggregate
            }
        }
    }

    fn invariant(&self, block: &Block) -> Var {
        match self {
            PrimOp::Add | PrimOp::Sub => block.constant(0),
            PrimOp::Mul | PrimOp::Div => block.constant(1),
            _ => panic!("invalid number of arguments"),
        }
    }

    fn singular(&self, x: &Var, block: &Block) -> Var {
        match self {
            PrimOp::Add => x.clone(),
            PrimOp::Sub => self.binop(&self.invariant(block), x, block),
            PrimOp::Mul => x.clone(),
            PrimOp::Div => self.binop(&self.invariant(block), x, block),
            _ => panic!("invalid number of arguments"),
        }
    }

    fn binop(&self, x: &Var, y: &Var, block: &Block) -> Var {
        match self {
            PrimOp::Add => block.add(x, y),
            PrimOp::Sub => block.sub(x, y),
            PrimOp::Mul => block.mul(x, y),
            PrimOp::Div => block.div(x, y),
            PrimOp::Equal => block.equals(x, y),
        }
    }
}

enum VarSlot {
    Immutable(Var),
    Mutable(Var),
}

struct FixCompiler {
    scope: Vec<(String, VarSlot)>,
    funcs: HashMap<String, Block>,
}

impl FixCompiler {
    pub fn new() -> Self {
        FixCompiler {
            scope: vec![],
            funcs: map![],
        }
    }

    fn compile_prog(&mut self, prog: &Prog) -> &'static [vm::Op] {
        let mut entry_blocks = vec![];
        let mut body_blocks = vec![];
        let mut trans_units = vec![];

        for fdef in &prog.function_definitions {
            let tu = TranslationUnit::new();
            let code = tu.new_block();
            for p in &fdef.params {
                self.scope
                    .push((p.clone(), VarSlot::Immutable(code.append_parameter())));
            }
            let entry = tu.new_function(&code);
            self.funcs.insert(fdef.name.clone(), entry.clone());
            entry_blocks.push(entry);
            body_blocks.push(code);
            trans_units.push(tu);
        }

        let mut compilers = vec![];

        let mut body_tu = TranslationUnit::new();
        let body_block = body_tu.new_block();
        let (result, final_block) = self.compile_expr(&prog.body, &body_block);
        if let Some(result) = result {
            final_block.return_(&result);
        }
        let c = body_tu.compile_function(&body_block);
        compilers.push(c);

        for (((fdef, entry), code), mut tu) in prog
            .function_definitions
            .iter()
            .zip(entry_blocks)
            .zip(body_blocks)
            .zip(trans_units)
        {
            let (ret, exit) = self.compile_expr(&fdef.body, &code);
            if let Some(ret) = ret {
                exit.return_(&ret);
            }

            let c = tu.compile_function(&entry);
            compilers.push(c);
        }

        let (code, offsets) = ssa_builder::link(&compilers);

        store_code_block(code)
    }

    fn compile_expr(&mut self, expr: &Expr, block: &Block) -> (Option<Var>, Block) {
        match expr {
            Expr::Atomic(aexp) => self.compile_aexp(aexp, block),
            Expr::Complex(cexp) => self.compile_cexp(cexp, block, true),
            Expr::Let(varname, def, body) => {
                let (v, block) = self.compile_cexp(def, block, false);
                self.scope
                    .push((varname.clone(), VarSlot::Immutable(v.unwrap())));
                let (r, block) = self.compile_expr(body, &block);
                self.scope.pop();
                (r, block)
            }
            Expr::LetMut(varname, def, body) => {
                let (v, block) = self.compile_cexp(def, block, false);
                self.scope
                    .push((varname.clone(), VarSlot::Mutable(v.unwrap())));
                let (r, block) = self.compile_expr(body, &block);
                self.scope.pop();
                (r, block)
            }
        }
    }

    fn compile_cexp(&mut self, cexp: &Cexp, block: &Block, tail_pos: bool) -> (Option<Var>, Block) {
        match cexp {
            Cexp::Atomic(aexp) => self.compile_aexp(aexp, block),
            Cexp::ApplyPrimitive(op, args) => {
                let args: Vec<_> = args
                    .iter()
                    .map(|a| self.compile_aexp(a, block).0.unwrap())
                    .collect();
                (Some(op.compile(&args, block)), block.clone())
            }
            Cexp::ApplyStatic(func, args) => {
                let args: Vec<_> = args
                    .iter()
                    .map(|a| self.compile_aexp(a, block).0.unwrap())
                    .collect();
                let ref_args: Vec<_> = args.iter().collect();
                let func = &self.funcs[func];
                if tail_pos {
                    block.tail_call_static(func, &ref_args);
                    (None, block.clone())
                } else {
                    (Some(block.call_static(func, &ref_args)), block.clone())
                }
            }
            Cexp::Apply(func, args) => {
                let args: Vec<_> = args
                    .iter()
                    .map(|a| self.compile_aexp(a, block).0.unwrap())
                    .collect();
                let ref_args: Vec<_> = args.iter().collect();
                let func = self.compile_aexp(func, block).0.unwrap();
                if tail_pos {
                    block.tail_call(&func, &ref_args);
                    (None, block.clone())
                } else {
                    (Some(block.call(&func, &ref_args)), block.clone())
                }
            }
            Cexp::If(cond, yes, no) => {
                let yes_block = block.create_sibling();
                let no_block = block.create_sibling();
                let after_if = block.create_sibling();

                let cond = self.compile_aexp(cond, block).0.unwrap();
                block.branch_conditionally(&cond, &yes_block, &no_block);
                let (yes_ret, yes_block) = self.compile_expr(yes, &yes_block);
                let (no_ret, no_block) = self.compile_expr(no, &no_block);

                if let Some(r) = yes_ret {
                    yes_block.branch(&after_if, &[&r]);
                }

                if let Some(r) = no_ret {
                    no_block.branch(&after_if, &[&r]);
                }

                (Some(after_if.append_parameter()), after_if)
            }
        }
    }

    fn compile_aexp(&mut self, aexp: &Aexp, block: &Block) -> (Option<Var>, Block) {
        (
            Some(match aexp {
                Aexp::Undefined => block.constant(PrimitiveValue::Undefined),
                Aexp::Integer(i) => block.constant(*i),
                Aexp::Var(var_name) => self.lookup(var_name, block).unwrap(),
                Aexp::Function(func_name) => block.label(self.funcs.get(func_name).unwrap()),
            }),
            block.clone(),
        )
    }

    fn lookup(&self, var_name: &str, block: &Block) -> Option<Var> {
        self.scope
            .iter()
            .rev()
            .find(|(name, _)| name == var_name)
            .map(|(_, slot)| match slot {
                VarSlot::Immutable(var) => var.clone(),
                VarSlot::Mutable(cell) => block.get_cell(cell),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa_builder::{
        TranslationUnit, FIRST_ARG_REGISTER, RETURN_TARGET_REGISTER, RETURN_VALUE_REGISTER,
    };
    use crate::virtual_machine::Operand::*;

    fn letvar(v: &str, cexp: Cexp, body: Expr) -> Expr {
        Expr::Let(v.to_string(), cexp, Box::new(body))
    }

    #[test]
    fn it_works() {
        let prog = Prog {
            function_definitions: vec![FunctionDefinition {
                name: "sqr".to_string(),
                params: vec!["x".to_string()],
                body: Cexp::ApplyPrimitive(
                    PrimOp::Mul,
                    vec![Aexp::Var("x".to_string()), Aexp::Var("x".to_string())],
                )
                .into(),
            }],
            //body: Cexp::Apply("sqr".to_string(), vec![Aexp::Integer(42)]).into(),
            body: letvar(
                "s",
                Cexp::ApplyStatic("sqr".to_string(), vec![Aexp::Integer(3)]),
                Cexp::ApplyStatic("sqr".to_string(), vec![Aexp::Var("s".to_string())]).into(),
            ),
        };

        let mut c = FixCompiler::new();
        let code = c.compile_prog(&prog);

        let main = store_code_block(vec![
            vm::Op::Alloc(ssa_builder::STACK_REGISTER, 100),
            vm::Op::Const(ssa_builder::STACK_POINTER_REGISTER, 0.into()),
            vm::Op::LoadLabel(ssa_builder::RETURN_TARGET_REGISTER, 2),
            vm::Op::JmpFar(code),
            vm::Op::Copy(0, ssa_builder::RETURN_VALUE_REGISTER),
            vm::Op::Term,
        ]);

        assert_eq!(vm::eval(main), 81.into());
    }

    #[test]
    fn escaping_function() {
        let prog = Prog {
            function_definitions: vec![FunctionDefinition {
                name: "sqr".to_string(),
                params: vec!["x".to_string()],
                body: Cexp::ApplyPrimitive(
                    PrimOp::Mul,
                    vec![Aexp::Var("x".to_string()), Aexp::Var("x".to_string())],
                )
                .into(),
            }],
            body: letvar(
                "s",
                Aexp::Function("sqr".to_string()).into(),
                Aexp::Var("s".to_string()).into(),
            ),
        };

        let mut c = FixCompiler::new();
        let code = c.compile_prog(&prog);

        let main = store_code_block(vec![
            vm::Op::Alloc(ssa_builder::STACK_REGISTER, 100),
            vm::Op::Const(ssa_builder::STACK_POINTER_REGISTER, 0.into()),
            vm::Op::LoadLabel(ssa_builder::RETURN_TARGET_REGISTER, 2),
            vm::Op::JmpFar(code),
            vm::Op::Copy(0, ssa_builder::RETURN_VALUE_REGISTER),
            vm::Op::Term,
        ]);

        println!("{:?}", code);

        let expected = store_code_block(vec![
            vm::Op::Mul(
                RETURN_VALUE_REGISTER,
                FIRST_ARG_REGISTER,
                R(FIRST_ARG_REGISTER),
            ),
            vm::Op::Jmp(R(RETURN_TARGET_REGISTER)),
        ]);
        assert_eq!(vm::eval(main), PrimitiveValue::CodeBlock(expected));
    }

    #[test]
    fn nested_let() {
        let prog = Prog {
            function_definitions: vec![],
            body: letvar(
                "a",
                Aexp::Integer(1).into(),
                letvar(
                    "b",
                    Aexp::Integer(2).into(),
                    letvar(
                        "c",
                        Aexp::Integer(3).into(),
                        Cexp::ApplyPrimitive(
                            PrimOp::Add,
                            vec![
                                Aexp::Var("a".to_string()),
                                Aexp::Var("b".to_string()),
                                Aexp::Var("c".to_string()),
                            ],
                        )
                        .into(),
                    ),
                ),
            ),
        };

        let mut c = FixCompiler::new();
        let code = c.compile_prog(&prog);

        let main = store_code_block(vec![
            vm::Op::Alloc(ssa_builder::STACK_REGISTER, 100),
            vm::Op::Const(ssa_builder::STACK_POINTER_REGISTER, 0.into()),
            vm::Op::LoadLabel(ssa_builder::RETURN_TARGET_REGISTER, 2),
            vm::Op::JmpFar(code),
            vm::Op::Copy(0, ssa_builder::RETURN_VALUE_REGISTER),
            vm::Op::Term,
        ]);

        println!("{:?}", code);
        assert_eq!(vm::eval(main), PrimitiveValue::Integer(6));
    }

    #[test]
    fn branching() {
        let prog = Prog {
            function_definitions: vec![FunctionDefinition {
                name: "is_zero".to_string(),
                params: vec!["x".to_string()],
                body: letvar(
                    "cond",
                    Cexp::ApplyPrimitive(
                        PrimOp::Equal,
                        vec![Aexp::Var("x".to_string()), Aexp::Integer(0)],
                    ),
                    Cexp::If(
                        Aexp::Var("cond".to_string()),
                        Box::new(Aexp::Integer(1).into()),
                        Box::new(Aexp::Integer(2).into()),
                    )
                    .into(),
                ),
            }],
            body: letvar(
                "a",
                Cexp::ApplyStatic("is_zero".to_string(), vec![Aexp::Integer(0)]),
                letvar(
                    "b",
                    Cexp::ApplyStatic("is_zero".to_string(), vec![Aexp::Integer(1)]),
                    Cexp::ApplyPrimitive(
                        PrimOp::Add,
                        vec![Aexp::Var("a".to_string()), Aexp::Var("b".to_string())],
                    )
                    .into(),
                ),
            ),
        };

        let mut c = FixCompiler::new();
        let code = c.compile_prog(&prog);

        let main = store_code_block(vec![
            vm::Op::Alloc(ssa_builder::STACK_REGISTER, 100),
            vm::Op::Const(ssa_builder::STACK_POINTER_REGISTER, 0.into()),
            vm::Op::LoadLabel(ssa_builder::RETURN_TARGET_REGISTER, 2),
            vm::Op::JmpFar(code),
            vm::Op::Copy(0, ssa_builder::RETURN_VALUE_REGISTER),
            vm::Op::Term,
        ]);

        println!("{:?}", code);
        assert_eq!(vm::eval(main), PrimitiveValue::Integer(3));
    }
}
