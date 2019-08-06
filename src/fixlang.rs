/// Closure lifting assumes that all functions have unique names. Otherwise there will be name
/// clashes at the outer scope.
use crate::nolambda;
use std::collections::HashSet;

pub type PrimOp = nolambda::PrimOp;

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
    PrimOp(PrimOp),
}

pub struct FunctionDefinition {
    pub name: String,
    pub params: Vec<String>,
    pub body: Box<Expr>,
}

/// Transform a fixlang Expression into a top-level nolambda Program.
fn transform_toplevel(expr: &Expr) -> nolambda::Prog {
    let mut context = Context::new();
    let mut function_definitions = vec![];
    let body = context.lift_lambdas(expr, &mut function_definitions);
    nolambda::Prog::new(function_definitions, body)
}

struct Context {}

impl Context {
    fn new() -> Self {
        Context {}
    }

    fn lift_lambdas(
        &mut self,
        expr: &Expr,
        top_funcs: &mut Vec<nolambda::FunctionDefinition>,
    ) -> nolambda::Expr {
        match expr {
            Expr::Atomic(aexp) => nolambda::Expr::Atomic(self.aexp_to_nolambda(aexp, top_funcs)),
            Expr::Complex(cexp) => nolambda::Expr::Complex(self.cexp_to_nolambda(cexp, top_funcs)),
            Expr::Begin(exprs) => nolambda::Expr::Begin(
                exprs
                    .iter()
                    .map(|x| self.lift_lambdas(x, top_funcs))
                    .collect(),
            ),
            Expr::Let(varname, definition, body) => nolambda::Expr::Let(
                varname.clone(),
                self.cexp_to_nolambda(definition, top_funcs),
                Box::new(self.lift_lambdas(body, top_funcs)),
            ),
            Expr::Fix(fndefs, body) => {
                for fndef in fndefs {
                    self.add_top_func(fndef, top_funcs);
                }
                self.lift_lambdas(body, top_funcs)
            }
        }
    }

    fn add_top_func(
        &mut self,
        fndef: &FunctionDefinition,
        top_funcs: &mut Vec<nolambda::FunctionDefinition>,
    ) {
        let body = self.lift_lambdas(&fndef.body, top_funcs);
        let nl_func = nolambda::FunctionDefinition {
            name: fndef.name.clone(),
            params: fndef.params.clone(),
            body,
        };
        top_funcs.push(nl_func);
    }

    fn cexp_to_nolambda(
        &mut self,
        cexp: &Cexp,
        top_funcs: &mut Vec<nolambda::FunctionDefinition>,
    ) -> nolambda::Cexp {
        match cexp {
            Cexp::Atomic(aexp) => nolambda::Cexp::Atomic(self.aexp_to_nolambda(aexp, top_funcs)),
            Cexp::Apply(func, args) => {
                let nl_args = args
                    .iter()
                    .map(|a| self.aexp_to_nolambda(a, top_funcs))
                    .collect();
                if let Some(function) = self.as_known_function(func) {
                    nolambda::Cexp::ApplyStatic(function, nl_args)
                } else {
                    let nl_func = self.aexp_to_nolambda(func, top_funcs);
                    nolambda::Cexp::Apply(nl_func, nl_args)
                }
            }
            Cexp::If(cond, yes, no) => nolambda::Cexp::If(
                self.aexp_to_nolambda(cond, top_funcs),
                Box::new(self.lift_lambdas(yes, top_funcs)),
                Box::new(self.lift_lambdas(no, top_funcs)),
            ),
        }
    }

    fn aexp_to_nolambda(
        &mut self,
        aexp: &Aexp,
        top_funcs: &mut Vec<nolambda::FunctionDefinition>,
    ) -> nolambda::Aexp {
        match aexp {
            Aexp::Undefined => nolambda::Aexp::Undefined,
            Aexp::Integer(i) => nolambda::Aexp::Integer(*i),
            Aexp::Var(v) if self.is_function_by_name(v) => nolambda::Aexp::Function(v.to_string()),
            Aexp::Var(v) => nolambda::Aexp::Var(v.to_string()),
            Aexp::PrimOp(op) => nolambda::Aexp::PrimOp(*op),
        }
    }

    fn as_primitive(&self, aexp: &Aexp) -> Option<PrimOp> {
        match aexp {
            Aexp::PrimOp(op) => Some(*op),
            _ => None,
        }
    }

    fn as_known_function(&self, aexp: &Aexp) -> Option<String> {
        unimplemented!()
    }

    fn is_function_by_name(&self, var: &str) -> bool {
        unimplemented!()
    }
}
