use crate::fixlang::Aexp::PrimOp;
use crate::primitive_value::PrimitiveValue;
use crate::ssa_builder;
use crate::virtual_machine as vm;
use std::collections::HashSet;

type Block = ssa_builder::Block<PrimitiveValue>;
type Var = ssa_builder::Var<PrimitiveValue>;

enum Expr {
    Lambda(Vec<String>, Box<Expr>),
    Apply(Box<Expr>, Vec<Expr>),

    Int(i64),
    Var(String),
    Mul,
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
            Expr::Var(name) => lookup(name, &self.env),
            Expr::Lambda(params, body) => {
                let fvs = free_vars(expr);
                let body_block = block.create_sibling();
                for p in params {
                    self.env.push((p.clone(), body_block.append_parameter()));
                }

                // todo: handle free variables

                let func_result = self.compile_expr(body, &body_block);
                body_block.return_(&func_result);
                let body_block = body_block.into_function();

                self.env.truncate(self.env.len() - params.len());

                let closure_record = block.make_rec(1);
                let function = block.label(&body_block);
                block.set_rec(&closure_record, 0, &function);

                self.lambda_blocks.push(body_block);

                closure_record
            }
            Expr::Apply(func, args) => {
                // todo: handle free variables
                let fv = self.compile_expr(func, block);
                let args: Vec<_> = args.iter().map(|a| self.compile_expr(a, block)).collect();
                let callee = block.get_rec(&fv, 0);
                block.call(&callee, &args.iter().collect::<Vec<_>>())
            }
            Expr::Mul => block.constant(PrimitiveValue::Undefined),
            _ => unimplemented!(),
        }
    }
}

fn lookup(name: &str, env: &[(String, Var)]) -> Var {
    for (n, v) in env.iter().rev() {
        if n == name {
            return v.clone();
        }
    }
    panic!("Unbound variable: {}", name);
}

fn free_vars(expr: &Expr) -> HashSet<&str> {
    match expr {
        Expr::Mul => set![],
        Expr::Int(_) => set![],
        Expr::Var(v) => set![v.as_str()],
        Expr::Apply(func, args) => {
            let mut fvs = free_vars(func);
            for a in args {
                fvs.extend(free_vars(a))
            }
            fvs
        }
        Expr::Lambda(params, body) => {
            let mut fvs = free_vars(body);
            for p in params {
                fvs.remove(p.as_str());
            }
            fvs
        }
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

        println!("{:?}", vm::eval(main));

        panic!()
    }
}
