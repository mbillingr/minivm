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
    Atomic(Aexp),
    Complex(Cexp),
}

enum Cexp {
    Atomic(Aexp),
    PrimitiveAdd(Aexp, Aexp),
    PrimitiveMul(Aexp, Aexp),
    Apply(String, Vec<Aexp>),
}

enum Aexp {
    Undefined,
    Integer(i64),
    Var(String),
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

struct FixCompiler {
    scope: Vec<(String, Var)>,
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
                self.scope.push((p.clone(), code.append_parameter()));
            }
            let entry = tu.new_function(&code);
            self.funcs.insert(fdef.name.clone(), entry.clone());
            entry_blocks.push(entry);
            body_blocks.push(code);
            trans_units.push(tu);
        }

        let mut compilers = vec![];
        let mut labels = vec![];

        let mut body_tu = TranslationUnit::new();
        let body_block = body_tu.new_block();
        if let Some(result) = self.compile_expr(&prog.body, &body_block) {
            body_block.return_(&result);
        }
        let (c, l) = body_tu.compile_function(&body_block);
        compilers.push(c);
        labels.push(l);

        for (((fdef, entry), code), mut tu) in prog
            .function_definitions
            .iter()
            .zip(entry_blocks)
            .zip(body_blocks)
            .zip(trans_units)
        {
            if let Some(ret) = self.compile_expr(&fdef.body, &code) {
                code.return_(&ret);
            }

            let (c, l) = tu.compile_function(&entry);
            compilers.push(c);
            labels.push(l);
        }

        let (code, offsets) = ssa_builder::link(&compilers, &labels);

        let code = store_code_block(code);
        assert_eq!(offsets[0], 0);
        code
    }

    fn compile_expr(&mut self, expr: &Expr, block: &Block) -> Option<Var> {
        match expr {
            Expr::Atomic(aexp) => Some(self.compile_aexp(aexp, block)),
            Expr::Complex(cexp) => self.compile_cexp(cexp, block, true),
            Expr::Let(varname, def, body) => {
                let v = self.compile_cexp(def, block, false).unwrap();
                self.scope.push((varname.clone(), v));
                let r = self.compile_expr(body, block);
                self.scope.pop();
                r
            }
        }
    }

    fn compile_cexp(&mut self, cexp: &Cexp, block: &Block, tail_pos: bool) -> Option<Var> {
        Some(match cexp {
            Cexp::Atomic(aexp) => self.compile_aexp(aexp, block),
            Cexp::PrimitiveAdd(a, b) => {
                block.add(&self.compile_aexp(a, block), &self.compile_aexp(b, block))
            }
            Cexp::PrimitiveMul(a, b) => {
                block.mul(&self.compile_aexp(a, block), &self.compile_aexp(b, block))
            }
            Cexp::Apply(func, args) => {
                let args: Vec<_> = args.iter().map(|a| self.compile_aexp(a, block)).collect();
                let ref_args: Vec<_> = args.iter().collect();
                let func = &self.funcs[func];
                if tail_pos {
                    block.tail_call_static(func, &ref_args);
                    return None;
                } else {
                    block.call_static(func, &ref_args)
                }
            }
        })
    }

    fn compile_aexp(&mut self, aexp: &Aexp, block: &Block) -> Var {
        match aexp {
            Aexp::Undefined => block.constant(PrimitiveValue::Undefined),
            Aexp::Integer(i) => block.constant(*i),
            Aexp::Var(var_name) => self.lookup(var_name).unwrap(),
        }
    }

    fn lookup(&self, var_name: &str) -> Option<Var> {
        self.scope
            .iter()
            .rev()
            .find(|(name, _)| name == var_name)
            .map(|(_, var)| var.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa_builder::TranslationUnit;

    fn letvar(v: &str, cexp: Cexp, body: Expr) -> Expr {
        Expr::Let(v.to_string(), cexp, Box::new(body))
    }

    #[test]
    fn it_works() {
        let prog = Prog {
            function_definitions: vec![FunctionDefinition {
                name: "sqr".to_string(),
                params: vec!["x".to_string()],
                body: Cexp::PrimitiveMul(Aexp::Var("x".to_string()), Aexp::Var("x".to_string()))
                    .into(),
            }],
            //body: Cexp::Apply("sqr".to_string(), vec![Aexp::Integer(42)]).into(),
            body: letvar(
                "s",
                Cexp::Apply("sqr".to_string(), vec![Aexp::Integer(3)]),
                Cexp::Apply("sqr".to_string(), vec![Aexp::Var("s".to_string())]).into(),
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
        println!("{:?}", vm::eval(main));

        unimplemented!()
    }
}
