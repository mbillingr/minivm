use crate::nolambda;

pub enum Expr {
    Begin(Vec<Expr>),
    Let(String, Cexp, Box<Expr>),
    Fix(Vec<FunctionDefinition>, Box<Expr>),
    Atomic(Aexp),
    Complex(Cexp),
}

pub enum Cexp {
    Atomic(Aexp),
    Apply(Aexp, Vec<Aexp>),
    If(Aexp, Box<Expr>, Box<Expr>),
}

pub enum Aexp {
    Undefined,
    Integer(i64),
    Var(String),
}

pub struct FunctionDefinition {
    name: String,
    params: Vec<String>,
    body: Expr,
}

/// Transform a fixlang Expression into a top-level nolambda Program.
fn transform_toplevel(expr: &Expr) -> nolambda::Prog {
    let mut context = Context::new();
    let mut function_definitions = vec![];
    let body = context.lift_closures(expr, &mut function_definitions);
    nolambda::Prog::new(function_definitions, body)
}

struct Context {}

impl Context {
    fn new() -> Self {
        Context {}
    }

    fn lift_closures(
        &mut self,
        expr: &Expr,
        top_funcs: &mut Vec<nolambda::FunctionDefinition>,
    ) -> nolambda::Expr {
        match expr {
            Expr::Begin(exprs) => nolambda::Expr::Begin(
                exprs
                    .iter()
                    .map(|x| self.lift_closures(x, top_funcs))
                    .collect(),
            ),
            Expr::Atomic(aexp) => nolambda::Expr::Atomic(self.aexp_to_nolambda(aexp)),
        }
    }

    fn cexp_to_nolambda(
        &mut self,
        cexp: &Cexp,
        top_funcs: &mut Vec<nolambda::FunctionDefinition>,
    ) -> nolambda::Cexp {
        match cexp {
            Cexp::Atomic(aexp) => nolambda::Cexp::Atomic(self.aexp_to_nolambda(aexp)),
            Cexp::Apply(func, args) => {
                let nl_args = args.iter().map(|a| self.aexp_to_nolambda(a).collect());
                if self.is_primitive(func) {
                    nolambda::Cexp::ApplyPrimitive()
                } else if self.is_function(func) {
                    nolambda::Cexp::ApplyStatic()
                } else {
                    let nl_func = self.aexp_to_nolambda(func);
                    nolambda::Cexp::Apply(nl_func, nl_args)
                }
            }
            Cexp::If(cond, yes, no) => nolambda::Cexp::If(
                self.aexp_to_nolambda(cond),
                Box::new(self.lift_closures(yes, top_funcs)),
                Box::new(self.lift_closures(no, top_funcs)),
            ),
        }
    }

    fn aexp_to_nolambda(&mut self, aexp: &Aexp) -> nolambda::Aexp {
        match aexp {
            Aexp::Undefined => nolambda::Aexp::Undefined,
            Aexp::Integer(i) => nolambda::Aexp::Integer(*i),
            Aexp::Var(v) if self.is_function(v) => nolambda::Aexp::Function(v.to_string()),
            Aexp::Var(v) => nolambda::Aexp::Var(v.to_string()),
        }
    }

    fn is_function(&self, var: &str) -> bool {
        unimplemented!()
    }
}
