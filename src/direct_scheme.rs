use crate::fixlang::Aexp::PrimOp;
use crate::memory::store_code_block;
use crate::primitive_value::{CodePos, PrimitiveValue};
use crate::scheme_parser::{parse_datum, Object, ObjectKind};
use crate::ssa_builder;
use crate::virtual_machine as vm;
use std::collections::HashSet;

type Block = ssa_builder::Block<PrimitiveValue>;
type Var = ssa_builder::Var<PrimitiveValue>;

fn read(source: &str) -> Expr {
    parse_datum(source).unwrap().into()
}

fn eval(expr: &Expr) -> PrimitiveValue {
    let mut tu = ssa_builder::TranslationUnit::new();
    let block = tu.new_block();

    let mut sc = SchemeCompiler::new();

    let x = sc.compile_expr(&expr, &block);
    block.return_(&x);

    let xc = tu.compile_function(&block);
    println!("{}", xc.get_assembly());

    let mut funcs = vec![xc];

    for bl in &sc.lambda_blocks {
        let c = tu.compile_function(bl);
        println!("{}", c.get_assembly());
        funcs.push(c);
    }

    let (ops, labels) = ssa_builder::link(&funcs);

    println!("{:?}", ops);
    println!("{:?}", labels);

    let code = CodePos::new(store_code_block(ops), 0);
    ssa_builder::eval(code)
}

enum Expr {
    Lambda(Vec<String>, Box<Expr>),
    Apply(Box<Expr>, Vec<Expr>),

    Int(i64),
    Var(String),
    Mul,
}

impl Expr {
    fn free_vars(&self) -> HashSet<&str> {
        match self {
            Expr::Mul => set![],
            Expr::Int(_) => set![],
            Expr::Var(v) => set![v.as_str()],
            Expr::Apply(func, args) => {
                let mut fvs = func.free_vars();
                for a in args {
                    fvs.extend(a.free_vars())
                }
                fvs
            }
            Expr::Lambda(params, body) => {
                let mut fvs = body.free_vars();
                for p in params {
                    fvs.remove(p.as_str());
                }
                fvs
            }
        }
    }
}

impl From<Object<'_>> for Expr {
    fn from(obj: Object) -> Self {
        match obj.kind {
            ObjectKind::Exact(i) => Expr::Int(i),
            _ => unimplemented!(),
        }
    }
}

struct SchemeCompiler {
    lambda_blocks: Vec<Block>,
    env: Vec<(String, Var)>,
}

impl SchemeCompiler {
    pub fn new() -> Self {
        SchemeCompiler {
            lambda_blocks: vec![],
            env: vec![],
        }
    }

    fn compile_expr(&mut self, expr: &Expr, block: &Block) -> Var {
        match expr {
            Expr::Int(value) => block.constant(*value),
            Expr::Var(name) => self.lookup(name),
            Expr::Lambda(params, body) => {
                self.build_lambda_body(body, &params, expr.free_vars(), &block)
            }
            Expr::Apply(func, args) => {
                let closure = self.compile_expr(func, block);
                let callee = block.get_rec(&closure, 0);
                let mut compiled_args = vec![closure];
                compiled_args.extend(args.iter().map(|a| self.compile_expr(a, block)));
                block.call(&callee, &compiled_args.iter().collect::<Vec<_>>())
            }
            Expr::Mul => self.build_primitive_body(block, |blk| {
                let a = blk.append_parameter();
                let b = blk.append_parameter();
                blk.mul(&a, &b)
            }),
        }
    }

    fn lookup(&self, name: &str) -> Var {
        for (n, v) in self.env.iter().rev() {
            if n == name {
                return v.clone();
            }
        }
        panic!("Unbound variable: {}", name);
    }

    fn build_lambda_body<'a>(
        &mut self,
        body: &Expr,
        params: &[String],
        free_vars: HashSet<&str>,
        block: &Block,
    ) -> Var {
        let env_len_before_call = self.env.len();
        let body_block = block.create_sibling();
        let closure_param = body_block.append_parameter();

        for p in params {
            self.env.push((p.clone(), body_block.append_parameter()));
        }

        for (i, var) in (1..).zip(&free_vars) {
            self.env
                .push((var.to_string(), body_block.get_rec(&closure_param, i)));
        }

        let func_result = self.compile_expr(body, &body_block);
        body_block.return_(&func_result);

        self.env.truncate(env_len_before_call);

        let body_block = body_block.into_function();

        let closure_record = self.build_closure_record(&body_block, free_vars, &block);
        self.lambda_blocks.push(body_block);
        closure_record
    }

    fn build_primitive_body(
        &mut self,
        block: &Block,
        definition: impl FnOnce(&mut Block) -> Var,
    ) -> Var {
        let mut body_block = block.create_sibling();
        let _closure_param = body_block.append_parameter();
        let result = definition(&mut body_block);
        body_block.return_(&result);
        let body_block = body_block.into_function();

        let closure_record = self.build_closure_record(&body_block, set![], &block);
        self.lambda_blocks.push(body_block);
        closure_record
    }

    fn build_closure_record(
        &mut self,
        body_block: &Block,
        free_vars: HashSet<&str>,
        block: &Block,
    ) -> Var {
        let closure_record = block.make_rec(1 + free_vars.len());
        let function = block.label(body_block);
        block.set_rec(&closure_record, 0, &function);
        for (i, var) in (1..).zip(free_vars) {
            block.set_rec(&closure_record, i, &self.lookup(var));
        }
        closure_record
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::store_code_block;
    use crate::primitive_value::CodePos;

    fn lambda(params: &[&str], body: Expr) -> Expr {
        Expr::Lambda(
            params.iter().copied().map(str::to_string).collect(),
            Box::new(body),
        )
    }

    fn apply(func: Expr, args: Vec<Expr>) -> Expr {
        Expr::Apply(Box::new(func), args)
    }

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    fn verbose_build(expr: &Expr) -> Vec<vm::Op<isize>> {
        let mut tu = ssa_builder::TranslationUnit::new();
        let block = tu.new_block();

        let mut sc = SchemeCompiler::new();

        let x = sc.compile_expr(&expr, &block);
        block.return_(&x);

        let xc = tu.compile_function(&block);
        println!("{}", xc.get_assembly());

        let mut funcs = vec![xc];

        for bl in &sc.lambda_blocks {
            let c = tu.compile_function(bl);
            println!("{}", c.get_assembly());
            funcs.push(c);
        }

        let (ops, labels) = ssa_builder::link(&funcs);

        println!("{:?}", ops);
        println!("{:?}", labels);
        ops
    }

    #[test]
    fn it_works() {
        use Expr::*;

        let do_mul = lambda(&["y"], apply(Mul, vec![var("x"), var("y")]));
        let mul_gen = lambda(&["x"], do_mul);
        let mul_five = apply(mul_gen, vec![Int(5)]);
        let expr = apply(mul_five, vec![Int(4)]);

        //let constfn = lambda(&[], Int(42));
        //let expr = apply(constfn, vec![]);

        //let identfn = lambda(&["x"], var("x"));
        //let expr = apply(identfn, vec![Int(42)]);

        let code = verbose_build(&expr);
        let code = store_code_block(code);

        let main = store_code_block(vec![
            vm::Op::Alloc(ssa_builder::STACK_REGISTER, 100),
            vm::Op::Const(ssa_builder::STACK_POINTER_REGISTER, 0.into()),
            vm::Op::LoadLabel(ssa_builder::RETURN_TARGET_REGISTER, 2),
            vm::Op::JmpFar(CodePos::new(code, 0)),
            vm::Op::Copy(0, ssa_builder::RETURN_VALUE_REGISTER),
            vm::Op::Term,
        ]);

        assert_eq!(vm::eval(main), PrimitiveValue::Integer(20));
    }

    #[test]
    fn eval_int() {
        let expr = read("42");
        let value = eval(&expr);
        assert_eq!(value, 42.into());
    }
}
