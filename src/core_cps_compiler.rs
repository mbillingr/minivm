use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

const CLOSURE_PARAM_POS: usize = 0;

pub type Variable = String;

#[derive(Debug, Clone)]
pub enum Value {
    Var(Variable),
    Int(i64),
}

/// Complex Expression
///
/// Semantics
/// ---------
///     - `Undefined` is used as a placeholder during code transformations. It should
///       never occur in the final result.
///     - `Primop(op, args, var, cont)` apply primitive operation `op` to `args` and
///       bind the result to variable `var` in scope of the continuation `cont`.
///     - `Apply(fnc, args)` apply function or closure `fnc` to args. This must always
///       happen in tail position; the function is expected to never return. If the
///       function takes a continuation it is passed as an ordinary argument.
///     - `Fix(funcdefs, cont)` defines a set of mutually recursive functions (these
///       functions can share a closure after closure conversion). The functions are
///       bound in the scope of the body continuation `cont`.
///     - `Record(values, var, cont)` allocates a new record containing `values` and
///       binds the record to `var` in `cont`.
///     - `Offset(idx, rec, var, cont)` creates a view into the record `rec` at offset
///       `idx`. The view behaves just like a normal record.
///     - `Select(idx, rec, var, cont)` fetch entry at index `idx` from `rec`.
#[derive(Debug, Clone)]
pub enum Cexp {
    Undefined,
    Unimplemented, // for testing
    PrimOp(PrimOp, Vec<Value>, Vec<Variable>, Box<Cexp>),
    Apply(Value, Vec<Value>),
    Fix(Vec<FixDef>, Box<Cexp>),
    Record(Vec<Value>, Variable, Box<Cexp>),
    Offset(isize, Value, Variable, Box<Cexp>),
    Select(usize, Value, Variable, Box<Cexp>),
    If(Value, Box<Cexp>, Box<Cexp>),
}

#[derive(Debug, Copy, Clone)]
pub enum PrimOp {
    Add,
    Mul,
}

#[derive(Debug, Clone)]
pub struct FixDef {
    name: Variable,
    params: Vec<Variable>,
    body: Cexp,
}

impl FixDef {
    fn close_free_variables(&mut self, vars: &[Variable], n_funcs: usize, offset: usize) {
        assert!(offset <= n_funcs);
        let closure_param = reserved_name("closure");
        self.params.insert(CLOSURE_PARAM_POS, closure_param.clone());
        for (i, var) in vars.iter().enumerate().rev() {
            if i < n_funcs {
                self.body = Cexp::Offset(
                    i as isize - offset as isize,
                    Value::Var(closure_param.clone()),
                    var.into(),
                    Box::new(std::mem::replace(&mut self.body, Cexp::Undefined)),
                );
            } else {
                self.body = Cexp::Select(
                    i - offset as usize,
                    Value::Var(closure_param.clone()),
                    var.into(),
                    Box::new(std::mem::replace(&mut self.body, Cexp::Undefined)),
                );
            }
        }
    }

    fn unique_rename(&mut self) -> String {
        self.rename(unique_varname(&self.name))
    }

    fn rename(&mut self, new_name: String) -> String {
        std::mem::replace(&mut self.name, new_name)
    }
}

pub trait FreeVariables {
    fn find_free_vars(&self) -> HashSet<Variable>;
}

impl<T: FreeVariables> FreeVariables for Vec<T> {
    fn find_free_vars(&self) -> HashSet<Variable> {
        self.iter().fold(HashSet::new(), |fv, x| {
            fv.union(&x.find_free_vars()).cloned().collect()
        })
    }
}

impl FreeVariables for Value {
    fn find_free_vars(&self) -> HashSet<Variable> {
        let mut fv = HashSet::new();
        match self {
            Value::Var(v) => {
                fv.insert(v.clone());
            }
            Value::Int(_) => {}
        };
        fv
    }
}

impl FreeVariables for Cexp {
    fn find_free_vars(&self) -> HashSet<Variable> {
        match self {
            Cexp::PrimOp(_, values, output, body) => body
                .find_free_vars()
                .difference(&output.iter().cloned().collect())
                .cloned()
                .collect::<HashSet<_>>()
                .union(&values.find_free_vars())
                .cloned()
                .collect(),
            Cexp::Apply(func, args) => args
                .find_free_vars()
                .union(&func.find_free_vars())
                .cloned()
                .collect(),
            Cexp::Fix(funcs, body) => funcs
                .find_free_vars()
                .union(&body.find_free_vars())
                .cloned()
                .collect::<HashSet<_>>()
                .difference(&funcs.iter().map(|f| f.name.clone()).collect())
                .cloned()
                .collect(),
            Cexp::Record(values, var, body) => {
                let mut fv = body.find_free_vars();
                fv.remove(var);
                fv.union(&values.find_free_vars()).cloned().collect()
            }
            Cexp::Offset(_, _, _, _) => unimplemented!(),
            Cexp::Select(_, _, _, _) => unimplemented!(),
            Cexp::If(_, _, _) => unimplemented!(),
            Cexp::Unimplemented => HashSet::new(),
            Cexp::Undefined => unreachable!(),
        }
    }
}

impl FreeVariables for FixDef {
    fn find_free_vars(&self) -> HashSet<Variable> {
        let mut fvs: HashSet<_> = self
            .body
            .find_free_vars()
            .difference(&self.params.iter().cloned().collect())
            .cloned()
            .collect();
        fvs.remove(&self.name);
        fvs
    }
}

pub fn closure_convert(cexp: Cexp) -> Cexp {
    match cexp {
        Cexp::Fix(mut funcs, fix_body) => {
            let mut new_body = closure_convert(*fix_body);
            for func in &mut funcs {
                func.body = closure_convert(std::mem::replace(&mut func.body, Cexp::Undefined));
            }

            let free_vars: Vec<_> = funcs
                .find_free_vars()
                .difference(&funcs.iter().map(|f| f.name.clone()).collect())
                .cloned()
                .collect();

            let func_names_old: Vec<_> = funcs.iter_mut().map(FixDef::unique_rename).collect();

            let mut var_names_new: Vec<_> = funcs.iter().map(|f| f.name.clone()).collect();
            var_names_new.extend(free_vars.iter().cloned());

            let mut var_names_old: Vec<_> = func_names_old.iter().cloned().collect();
            var_names_old.extend(free_vars.into_iter());

            let n_funcs = funcs.len();
            for (offset, func) in funcs.iter_mut().enumerate() {
                func.close_free_variables(&var_names_old, n_funcs, offset);
            }

            let closure_name = unique_varname("closure");

            for (i, old_name) in func_names_old.into_iter().enumerate().rev() {
                new_body = Cexp::Offset(
                    i as isize,
                    Value::Var(closure_name.clone()),
                    old_name,
                    Box::new(new_body),
                );
            }

            let fix_body = Cexp::Record(
                var_names_new.into_iter().map(Value::Var).collect(),
                closure_name,
                Box::new(new_body),
            );

            Cexp::Fix(funcs, Box::new(fix_body))
        }
        Cexp::PrimOp(op, args, vars, cont) => {
            Cexp::PrimOp(op, args, vars, Box::new(closure_convert(*cont)))
        }
        Cexp::Record(vals, var, cont) => Cexp::Record(vals, var, Box::new(closure_convert(*cont))),
        Cexp::Offset(idx, rec, var, cont) => {
            Cexp::Offset(idx, rec, var, Box::new(closure_convert(*cont)))
        }
        Cexp::Select(idx, rec, var, cont) => {
            Cexp::Select(idx, rec, var, Box::new(closure_convert(*cont)))
        }
        Cexp::If(cond, yes, no) => Cexp::If(
            cond,
            Box::new(closure_convert(*yes)),
            Box::new(closure_convert(*no)),
        ),
        Cexp::Apply(_, _) => cexp,
        Cexp::Unimplemented => cexp,
        Cexp::Undefined => unreachable!(),
    }
}

const INDENT: usize = 2;

trait PrettyPrint {
    fn to_pretty_string(&self, indent: usize) -> String;

    fn println(&self) {
        println!("{}", self.to_pretty_string(0));
    }
}

impl PrettyPrint for Value {
    fn to_pretty_string(&self, _: usize) -> String {
        match self {
            Value::Var(v) => v.clone(),
            Value::Int(i) => format!("{}", i),
        }
    }
}

impl PrettyPrint for PrimOp {
    fn to_pretty_string(&self, _: usize) -> String {
        match self {
            PrimOp::Add => "+".to_owned(),
            PrimOp::Mul => "*".to_owned(),
        }
    }
}

impl PrettyPrint for Cexp {
    fn to_pretty_string(&self, indent: usize) -> String {
        match self {
            Cexp::Undefined => format!("{}<undefined>", " ".repeat(indent)),
            Cexp::Unimplemented => format!("{}<unimplemented>", " ".repeat(indent)),
            Cexp::Fix(defs, cont) => {
                let mut s = " ".repeat(indent) + "(fix";
                if !defs.is_empty() {
                    s += " (";
                    for (i, def) in defs.iter().enumerate() {
                        if i > 0 {
                            s += "\n";
                            s += &" ".repeat(indent + 6);
                        }
                        s += &format!("({} ({})", def.name, def.params.join(" "));
                        s += "\n";
                        s += &def.body.to_pretty_string(indent + def.name.len() + 8);
                        s += ")";
                    }
                    s += ")";
                }
                s += "\n";
                s += &cont.to_pretty_string(indent + INDENT);
                s += ")";
                s
            }
            Cexp::PrimOp(op, args, var, cont) => {
                let opps = op.to_pretty_string(0);
                let mut s = format!(
                    "{}({} ({}) ({})\n",
                    " ".repeat(indent),
                    opps,
                    args.iter()
                        .map(|x| x.to_pretty_string(0))
                        .collect::<Vec<_>>()
                        .join(" "),
                    var.join(" ")
                );
                s += &cont.to_pretty_string(indent + opps.len() + 2);
                s += ")";
                s
            }
            Cexp::Apply(func, args) => {
                if args.is_empty() {
                    format!("{}({})", " ".repeat(indent), func.to_pretty_string(0))
                } else {
                    format!(
                        "{}({} {})",
                        " ".repeat(indent),
                        func.to_pretty_string(0),
                        args.iter()
                            .map(|x| x.to_pretty_string(0))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                }
            }
            Cexp::Record(vals, var, cont) => {
                let mut s = format!(
                    "{}(record [{}] {}\n",
                    " ".repeat(indent),
                    vals.iter()
                        .map(|x| x.to_pretty_string(0))
                        .collect::<Vec<_>>()
                        .join(" "),
                    var
                );
                s += &cont.to_pretty_string(indent + INDENT);
                s += ")";
                s
            }
            Cexp::Offset(idx, val, var, cont) => {
                let mut s = format!(
                    "{}(offset {} {} {}\n",
                    " ".repeat(indent),
                    idx,
                    val.to_pretty_string(0),
                    var
                );
                s += &cont.to_pretty_string(indent + INDENT);
                s += ")";
                s
            }
            Cexp::Select(idx, val, var, cont) => {
                let mut s = format!(
                    "{}(select {} {} {}\n",
                    " ".repeat(indent),
                    idx,
                    val.to_pretty_string(0),
                    var
                );
                s += &cont.to_pretty_string(indent + INDENT);
                s += ")";
                s
            }
            Cexp::If(cond, a, b) => format!(
                "{}(if {}\n{}\n{})",
                " ".repeat(indent),
                cond.to_pretty_string(0),
                a.to_pretty_string(indent + 4),
                b.to_pretty_string(indent + 4)
            ),
        }
    }
}

static VAR_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_varname(name: &str) -> String {
    let id = VAR_COUNTER.fetch_add(1, Ordering::SeqCst);
    reserved_name(&format!("{}-{}", name, id))
}

/// convert identifier name into something that cannot be created by the source language.
fn reserved_name(name: &str) -> String {
    format!("<{}>", name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_cps_program() {
        use PrimOp::*;
        use Value::*;
        Cexp::Fix(
            vec![FixDef {
                name: "f".to_string(),
                params: vec!["x".to_string(), "k".to_string()],
                body: Cexp::PrimOp(
                    Mul,
                    vec![Int(2), Var("x".to_string())],
                    vec!["u".to_string()],
                    Box::new(Cexp::PrimOp(
                        Add,
                        vec![Var("u".to_string()), Int(12)],
                        vec!["v".to_string()],
                        Box::new(Cexp::Apply(
                            Var("k".to_string()),
                            vec![Var("u".to_string())],
                        )),
                    )),
                ),
            }],
            Box::new(Cexp::Apply(
                Var("f".to_string()),
                vec![Int(15), Var("r".to_string())],
            )),
        );
    }

    #[test]
    fn closure_conversion() {
        use PrimOp::*;
        use Value::*;
        let prog = Cexp::Fix(
            vec![
                FixDef {
                    name: "f".to_string(),
                    params: vec!["x".to_string(), "k".to_string()],
                    body: Cexp::PrimOp(
                        Mul,
                        vec![Var("y".to_string()), Var("x".to_string())],
                        vec!["u".to_string()],
                        Box::new(Cexp::Apply(
                            Var("k".to_string()),
                            vec![Var("u".to_string())],
                        )),
                    ),
                },
                FixDef {
                    name: "g".to_string(),
                    params: vec!["k".to_string()],
                    body: Cexp::Apply(Var("k".to_string()), vec![Var("y".to_string())]),
                },
            ],
            Box::new(Cexp::Apply(
                Var("f".to_string()),
                vec![Int(15), Var("r".to_string())],
            )),
        );

        prog.println();
        let actual = closure_convert(prog);
        actual.println();
    }

    #[test]
    fn closure_conversion2() {
        use PrimOp::*;
        use Value::*;
        let prog = Cexp::PrimOp(
            Add,
            vec![Int(1), Int(2)],
            vec!["z".into()],
            Box::new(Cexp::Fix(
                vec![
                    FixDef {
                        name: "f".to_string(),
                        params: vec![],
                        body: Cexp::PrimOp(
                            Add,
                            vec![Var("z".into()), Var("f".into())],
                            vec![],
                            Box::new(Cexp::Unimplemented),
                        ),
                    },
                    FixDef {
                        name: "g".to_string(),
                        params: vec!["x".into(), "y".into()],
                        body: Cexp::Fix(
                            vec![],
                            Box::new(Cexp::Apply(
                                Var("g".to_string()),
                                vec![Var("z".into()), Var("w".into())],
                            )),
                        ),
                    },
                ],
                Box::new(Cexp::Apply(Var("f".to_string()), vec![Var("z".into())])),
            )),
        );

        prog.println();
        let actual = closure_convert(prog);
        actual.println();
        //panic!()
    }

    #[test]
    fn closure_recursion() {
        use Value::*;
        let prog = Cexp::Fix(
            vec![
                FixDef {
                    name: "f".to_string(),
                    params: vec![],
                    body: Cexp::Apply(Var("g".to_string()), vec![]),
                },
                FixDef {
                    name: "g".to_string(),
                    params: vec![],
                    body: Cexp::Apply(Var("f".to_string()), vec![]),
                },
            ],
            Box::new(Cexp::Apply(Var("f".to_string()), vec![])),
        );

        prog.println();
        let actual = closure_convert(prog);
        actual.println();
    }

    /*#[test]
    fn pretty() {
        use PrimOp::*;
        use Value::*;
        println!("{}", Cexp::Fix(vec![FixDef {
            name: "f".to_string(),
            params: vec!["x".to_string(), "k".to_string()],
            body: Cexp::PrimOp(Add, vec![Var("z".into()), Var("x".into())], vec![], Box::new(Cexp::Unimplemented)),
        }, FixDef {
            name: "g".to_string(),
            params: vec!["x".to_string(), "k".to_string()],
            body: Cexp::PrimOp(Add, vec![Var("z".into()), Var("x".into())], vec![], Box::new(Cexp::Unimplemented)),
        }], Box::new(Cexp::Undefined)).to_pretty_string(0));
        panic!();
    }*/
}
