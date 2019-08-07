use crate::fixlang;
use std::collections::HashSet;

pub type PrimOp = fixlang::PrimOp;

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
    pub body: Expr,
}

/// Transform a closure_lang Expression into a fixlang expression, where closures are
/// created explicitly for each function.
pub fn transform_closures(expr: &Expr) -> fixlang::Expr {
    let mut context = Context::new();
    context.convert_closures(expr)
}

struct Context {}

impl Context {
    fn new() -> Self {
        Context {}
    }

    fn convert_closures(&mut self, expr: &Expr) -> fixlang::Expr {
        match expr {
            Expr::Atomic(aexp) => fixlang::Expr::Atomic(self.aexp_to_fixlang(aexp)),
            Expr::Complex(cexp) => fixlang::Expr::Complex(self.cexp_to_fixlang(cexp)),
            Expr::Begin(exprs) => {
                fixlang::Expr::Begin(exprs.iter().map(|x| self.convert_closures(x)).collect())
            }
            Expr::Let(varname, definition, body) => fixlang::Expr::Let(
                varname.clone(),
                self.cexp_to_fixlang(definition),
                Box::new(self.convert_closures(body)),
            ),
            Expr::Fix(fndefs, body) => {
                // todo:
                //   1. identify free variables of function
                //   2. make them parameters of the function
                //   3. modify call sites to pass the variables as parameters
                //   4. if the function escapes
                //      4.1 create wrapper function that extracts free-var parameters from closure and calls the function
                //      4.2 create closure pointing to wrapper function and variables (eventually shared between all functions in this fixture)
                //      4.3 let closure escape instead of function
                //   5? What new Atomics do we need? Closures and Known functions?

                let closure_var_name = "unique closure name".to_string();
                unimplemented!();

                /*let mut closure_vars = vec![];
                let mut fix_fvs = set![];
                let mut fixdefs = vec![];
                for fndef in fndefs {
                    let fn_fvs = find_free_vars_in_function(fndef);

                    fix_fvs.extend(fn_fvs.iter().copied());
                    fix_fvs.insert(fndef.name.as_str());

                    let mut params = fndef.params.clone();
                    params.extend(fn_fvs.into_iter().map(str::to_string));

                    let body = self.convert_closures(&fndef.body);
                    let fix_func = fixlang::FunctionDefinition {
                        name: fndef.name.clone(), // todo: rename function
                        params,
                        body,
                    };

                    closure_vars.push(fndef.name.as_str());
                    fixdefs.push(fix_func);
                }

                for fndef in fndefs {
                    fix_fvs.remove(fndef.name.as_str());
                }
                closure_vars.extend(fix_fvs);

                for (i, fndef) in fndefs.iter().enumerate() {
                    let function = fixlang::Aexp::Var(fixdefs[i].name.clone());
                    let mut args: Vec<_> = fndef
                        .params
                        .iter()
                        .cloned()
                        .map(fixlang::Aexp::Var)
                        .collect();
                    args.extend(
                        fixdefs[i].params[args.len()..]
                            .iter()
                            .cloned()
                            .map(fixlang::Aexp::Var),
                    );
                    let mut body = fixlang::Expr::Complex(fixlang::Cexp::Apply(function, args));

                    for closure_arg in &fixdefs[i].params[args.len()..] {
                        let closure_record = fixlang::Aexp::Var(closure_var_name.clone());
                        let closure_index = closure_vars
                            .iter()
                            .position(|v| v == closure_arg)
                            .expect("Could not find closure variable");
                        body = fixlang::Expr::Let(
                            closure_arg.clone(),
                            fixlang::Cexp::Apply(
                                fixlang::Aexp::PrimOp(crate::nolambda::PrimOp::GetRec),
                                vec![closure_record],
                            ),
                            Box::new(body),
                        );
                    }

                    let fix_func = fixlang::FunctionDefinition {
                        name: fndef.name.clone(),
                        params: fndef.params.clone(),
                        body,
                    };
                    fixdefs.push(fix_func);
                }

                let body = self.convert_closures(body);

                let shared_closure = fixlang::Cexp::Apply(
                    fixlang::Aexp::PrimOp(crate::nolambda::PrimOp::MakeRec),
                    closure_vars
                        .into_iter()
                        .map(str::to_string)
                        .map(fixlang::Aexp::Var)
                        .collect(),
                );
                let closure_scope =
                    fixlang::Expr::Let(closure_var_name, shared_closure, Box::new(body));
                fixlang::Expr::Fix(fixdefs, Box::new(closure_scope))*/
            }
        }
    }

    fn cexp_to_fixlang(&mut self, cexp: &Cexp) -> fixlang::Cexp {
        match cexp {
            Cexp::Atomic(aexp) => fixlang::Cexp::Atomic(self.aexp_to_fixlang(aexp)),
            Cexp::Apply(func, args) => fixlang::Cexp::Apply(
                self.aexp_to_fixlang(func),
                args.iter().map(|a| self.aexp_to_fixlang(a)).collect(),
            ),
            Cexp::If(cond, yes, no) => fixlang::Cexp::If(
                self.aexp_to_fixlang(cond),
                Box::new(self.convert_closures(yes)),
                Box::new(self.convert_closures(no)),
            ),
        }
    }

    fn aexp_to_fixlang(&mut self, aexp: &Aexp) -> fixlang::Aexp {
        match aexp {
            Aexp::Undefined => fixlang::Aexp::Undefined,
            Aexp::Integer(i) => fixlang::Aexp::Integer(*i),
            Aexp::Var(v) => fixlang::Aexp::Var(v.to_string()),
            Aexp::PrimOp(op) => fixlang::Aexp::PrimOp(*op),
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

fn find_free_vars_in_expr<'a>(expr: &'a Expr) -> HashSet<&'a str> {
    match expr {
        Expr::Atomic(aexp) => find_free_vars_in_aexp(aexp).into_iter().collect(),
        Expr::Complex(cexp) => find_free_vars_in_cexp(cexp),
        Expr::Begin(exprs) => exprs
            .iter()
            .map(|x| find_free_vars_in_expr(x))
            .fold(set![], |acc, fvs| acc.union(&fvs).copied().collect()),
        Expr::Let(varname, definition, body) => {
            let mut fvs = find_free_vars_in_expr(body);
            fvs.remove(varname.as_str());
            fvs.union(&find_free_vars_in_cexp(definition))
                .copied()
                .collect()
        }
        Expr::Fix(fndefs, body) => {
            let mut fvs = find_free_vars_in_expr(body);
            for fndef in fndefs {
                fvs.extend(find_free_vars_in_function(fndef));
            }

            for fndef in fndefs {
                fvs.remove(fndef.name.as_str());
            }

            fvs
        }
    }
}

fn find_free_vars_in_function<'a>(fndef: &'a FunctionDefinition) -> HashSet<&'a str> {
    let mut fvs = find_free_vars_in_expr(&fndef.body);
    for param in &fndef.params {
        fvs.remove(param.as_str());
    }
    fvs
}

fn find_free_vars_in_cexp<'a>(cexp: &'a Cexp) -> HashSet<&'a str> {
    match cexp {
        Cexp::Atomic(aexp) => find_free_vars_in_aexp(aexp).into_iter().collect(),
        Cexp::Apply(func, args) => args
            .iter()
            .chain(std::iter::once(func))
            .filter_map(|aexp| find_free_vars_in_aexp(aexp))
            .collect(),
        Cexp::If(cond, yes, no) => {
            let mut fv: HashSet<_> = find_free_vars_in_expr(yes)
                .union(&find_free_vars_in_expr(no))
                .copied()
                .collect();
            fv.extend(find_free_vars_in_aexp(cond));
            fv
        }
    }
}

fn find_free_vars_in_aexp<'a>(aexp: &'a Aexp) -> Option<&'a str> {
    match aexp {
        Aexp::Undefined => None,
        Aexp::Integer(_) => None,
        Aexp::Var(v) => Some(v),
        Aexp::PrimOp(_) => None,
    }
}
