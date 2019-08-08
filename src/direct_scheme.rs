use crate::memory::store_code_block;
use crate::primitive_value::{CodePos, PrimitiveValue};
use crate::scheme_parser::{parse_datum, Object, ObjectKind};
use crate::ssa_builder;
use crate::virtual_machine as vm;
use std::collections::{HashMap, HashSet};

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

#[derive(Debug, PartialEq)]
enum Expr {
    Lambda(Vec<String>, Vec<Expr>),
    Apply(Box<Expr>, Vec<Expr>),

    Int(i64),
    Symbol(String),
    Mul,
}

impl Expr {}

impl From<Object<'_>> for Expr {
    fn from(obj: Object) -> Self {
        match obj.kind {
            ObjectKind::Exact(i) => Expr::Int(i),
            ObjectKind::Symbol(s) => Expr::Symbol(s.to_string()),
            ObjectKind::Pair(pair) => {
                let (car, cdr) = pair.into_inner();
                match car.kind {
                    ObjectKind::Symbol("lambda") => {
                        let (formals, body) = cdr.decons();
                        Expr::Lambda(formals.into(), body.into())
                    }
                    _ => Expr::Apply(car.into(), cdr.into()),
                }
            }
            _ => unimplemented!("{:?}", obj),
        }
    }
}

impl From<Object<'_>> for Box<Expr> {
    fn from(obj: Object) -> Self {
        Box::new(obj.into())
    }
}

impl From<Object<'_>> for Vec<Expr> {
    fn from(mut obj: Object) -> Self {
        let mut v = vec![];
        loop {
            match obj.kind {
                ObjectKind::Nil => return v,
                ObjectKind::Pair(pair) => {
                    let (car, cdr) = pair.into_inner();
                    v.push(car.into());
                    obj = cdr;
                }
                _ => panic!("Cannot convert improper list to Vec"),
            }
        }
    }
}

impl From<Object<'_>> for Vec<String> {
    fn from(mut obj: Object) -> Self {
        let mut v = vec![];
        loop {
            match obj.kind {
                ObjectKind::Nil => return v,
                ObjectKind::Pair(pair) => {
                    let (car, cdr) = pair.into_inner();
                    if let ObjectKind::Symbol(s) = car.kind {
                        v.push(s.to_string());
                    } else {
                        panic!("Expected list of symbols")
                    }
                    obj = cdr;
                }
                _ => panic!("Cannot convert improper list to Vec"),
            }
        }
    }
}

#[derive(Default)]
struct PrimitivePropagation<'a> {
    env: Vec<&'a str>,
}

impl<'a> PrimitivePropagation<'a> {
    pub fn pass(expr: Expr) -> Expr {
        let mut context = PrimitivePropagation::default();
        context.recurse(expr)
    }

    fn recurse(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::Int(i) => Expr::Int(i),
            Expr::Symbol(name) => {
                if self.lookup(&name) {
                    Expr::Symbol(name)
                } else if let Some(p) = self.lookup_primitive(&name) {
                    p
                } else {
                    Expr::Symbol(name)
                }
            }
            Expr::Apply(f, args) => Expr::Apply(
                Box::new(self.recurse(*f)),
                args.into_iter().map(|a| self.recurse(a)).collect(),
            ),
            Expr::Lambda(formals, mut body) => {
                unsafe {
                    // The borrow checker won't let us temporarily borrow formals in self.env.
                    // Thus, we eradicate the lifetime by casting to a raw pointer and back.
                    // This is only safe if all references to formals are removed again from env.
                    self.env.extend(
                        formals
                            .iter()
                            .map(String::as_str)
                            .map(|s| s as *const _)
                            .map(|s| std::mem::transmute::<_, &str>(s)),
                    );
                }
                let body = body.into_iter().map(|x| self.recurse(x)).collect();

                // remove formals borrowed above from env
                self.env.truncate(self.env.len() - formals.len());
                Expr::Lambda(formals, body)
            }
            Expr::Mul => Expr::Mul,
            //_ => expr,
        }
    }

    fn lookup(&self, name: &str) -> bool {
        self.env.contains(&name)
    }

    fn lookup_primitive(&self, name: &str) -> Option<Expr> {
        match name {
            "*" => Some(Expr::Mul),
            _ => None,
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
            Expr::Symbol(name) => {
                if let Some(var) = self.lookup(name) {
                    var
                } else if let Some(p) = self.lookup_primitive(name) {
                    self.compile_expr(&p, block)
                } else {
                    panic!("Unbound symbol: {}", name)
                }
            }
            Expr::Lambda(params, body) => {
                self.build_lambda_body(body, &params, self.free_vars(expr), &block)
            }
            Expr::Apply(func, args) => {
                if self.is_primitive_procedure(func) {
                    match **func {
                        Expr::Mul => block.mul(
                            &self.compile_expr(&args[0], block),
                            &self.compile_expr(&args[1], block),
                        ),
                        _ => unreachable!(),
                    }
                } else {
                    let closure = self.compile_expr(func, block);
                    let callee = block.get_rec(&closure, 0);
                    let mut compiled_args = vec![closure];
                    compiled_args.extend(args.iter().map(|a| self.compile_expr(a, block)));
                    block.call(&callee, &compiled_args.iter().collect::<Vec<_>>())
                }
            }
            Expr::Mul => self.build_primitive_body(block, |blk| {
                let a = blk.append_parameter();
                let b = blk.append_parameter();
                blk.mul(&a, &b)
            }),
        }
    }

    fn compile_sequence(&mut self, seq: &[Expr], block: &Block) -> Var {
        let mut result = block.constant(PrimitiveValue::Undefined);
        for expr in seq {
            result = self.compile_expr(expr, &block);
        }
        result
    }

    fn is_primitive_procedure(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Mul => true,
            _ => false,
        }
    }

    fn lookup(&self, name: &str) -> Option<Var> {
        for (n, v) in self.env.iter().rev() {
            if n == name {
                return Some(v.clone());
            }
        }
        None
    }

    fn lookup_primitive(&self, name: &str) -> Option<Expr> {
        match name {
            "*" => Some(Expr::Mul),
            _ => None,
        }
    }

    fn build_lambda_body<'a>(
        &mut self,
        body: &[Expr],
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

        let mut func_result = self.compile_sequence(body, &body_block);
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
            block.set_rec(&closure_record, i, &self.lookup(var).unwrap());
        }
        closure_record
    }

    fn free_vars<'a>(&self, expr: &'a Expr) -> HashSet<&'a str> {
        match expr {
            Expr::Mul => set![],
            Expr::Int(_) => set![],
            Expr::Symbol(v) if self.lookup_primitive(v).is_some() => set![],
            Expr::Symbol(v) => set![v.as_str()],
            Expr::Apply(func, args) => {
                let mut fvs = self.free_vars(func);
                for a in args {
                    fvs.extend(self.free_vars(a))
                }
                fvs
            }
            Expr::Lambda(params, body) => {
                let mut fvs = self.free_vars_in_sequence(body);
                for p in params {
                    fvs.remove(p.as_str());
                }
                fvs
            }
        }
    }

    fn free_vars_in_sequence<'a>(&self, seq: &'a [Expr]) -> HashSet<&'a str> {
        let mut fvs = set![];
        for expr in seq {
            fvs.extend(self.free_vars(expr));
        }
        fvs
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
            vec![body],
        )
    }

    fn apply(func: Expr, args: Vec<Expr>) -> Expr {
        Expr::Apply(Box::new(func), args)
    }

    fn var(name: &str) -> Expr {
        Expr::Symbol(name.to_string())
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

    #[test]
    fn eval_primitive() {
        let expr = read("*");
        let value = eval(&expr);

        if let PrimitiveValue::Record(r) = value {
            assert_eq!(r.len, 1)
        } else {
            panic!("Expected a closure record")
        }
    }

    #[test]
    fn eval_primitive_application() {
        let expr = read("(* 2 3)");
        let value = eval(&expr);
        assert_eq!(value, 6.into());
    }

    #[test]
    fn eval_lambda() {
        let expr = read("(lambda (x) (* x x))");
        let value = eval(&expr);

        if let PrimitiveValue::Record(r) = value {
            assert_eq!(r.len, 1)
        } else {
            panic!("Expected a closure record")
        }
    }

    #[test]
    fn eval_lambda_application() {
        let expr = read("((lambda (x) (* x x)) 5)");
        let value = eval(&expr);
        assert_eq!(value, 25.into());
    }

    #[test]
    fn eval_lambda_returns_last_body_expression() {
        let expr = read("((lambda () 1 2 3 4))");
        let value = eval(&expr);
        assert_eq!(value, 4.into());
    }

    #[test]
    fn optimized_primitive_application() {
        use vm::Op::*;
        use vm::Operand::*;
        use Expr::*;
        use PrimitiveValue::*;
        let expr = read("(* 2 3)");
        let expr = PrimitivePropagation::pass(expr);
        assert_eq!(expr, apply(Expr::Mul, vec![Int(2), Int(3)]));
        let code = verbose_build(&expr);
        println!("{:?}", code);
        match code.as_slice() {
            [Const(a, Integer(2)), Const(b, Integer(3)), vm::Op::Mul(_, x, R(y)), _] => {
                assert_eq!(a, x);
                assert_eq!(b, y);
            }
            _ => panic!("Unexpected output code"),
        }
    }

    #[test]
    fn optimized_primitive_application_nested_context() {
        use Expr::*;
        let expr = read("(lambda (x) (* x 99))");
        let expr = PrimitivePropagation::pass(expr);
        assert_eq!(expr, lambda(&["x"], apply(Mul, vec![var("x"), Int(99)])));
    }

    #[test]
    fn primitive_propagation_obeys_lambda_bound_names() {
        let expr = read("((lambda (*) *) 42)");
        let value = eval(&expr);
        assert_eq!(value, 42.into());
    }

    /*#[test]
    fn eval_quote() {
        let expr = read("'(1 2 3)");
        let value = eval(&expr);
        assert_eq!(value, 42.into());
    }*/
}
