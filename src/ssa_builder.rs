use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Var(usize);

#[derive(Debug, Copy, Clone, PartialEq)]
enum ConstValue {
    Integer(i64),
}

#[derive(Debug, Clone, PartialEq)]
enum Op {
    Const(Var, ConstValue),
    Add(Var, Var, Var),
    Mul(Var, Var, Var),
    Term(Var),
    Branch(Block, Vec<Var>),
    CondBranch(Var, Block, Vec<Var>, Block, Vec<Var>),
}

impl Op {
    fn is_terminal(&self) -> bool {
        match self {
            Op::Term(_) => true,
            Op::Branch(_, _) => true,
            Op::CondBranch(_, _, _, _, _) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
struct Block {
    data: Rc<RefCell<BlockData>>,
}

impl Block {
    fn new(n_args: usize) -> Self {
        Block {
            data: Rc::new(RefCell::new(BlockData::new(n_args))),
        }
    }

    fn n_args(&self) -> usize {
        self.data.borrow().n_args
    }

    fn parameter(&self, idx: usize) -> Var {
        self.data.borrow_mut().parameter(idx)
    }

    fn terminate(&self, x: Var) {
        self.data.borrow_mut().terminate(x)
    }

    fn constant(&self, i: i64) -> Var {
        self.data.borrow_mut().constant(i)
    }

    fn add(&self, a: Var, b: Var) -> Var {
        self.data.borrow_mut().add(a, b)
    }

    fn mul(&self, a: Var, b: Var) -> Var {
        self.data.borrow_mut().mul(a, b)
    }

    fn branch(&self, to: &Block, args: &[Var]) {
        to.data
            .borrow_mut()
            .preceding_blocks
            .push(Rc::downgrade(&self.data));
        self.data.borrow_mut().branch(to.clone(), args.to_vec())
    }

    fn cond_branch(
        &self,
        cond: Var,
        then_block: &Block,
        then_args: &[Var],
        else_block: &Block,
        else_args: &[Var],
    ) {
        then_block
            .data
            .borrow_mut()
            .preceding_blocks
            .push(Rc::downgrade(&self.data));
        else_block
            .data
            .borrow_mut()
            .preceding_blocks
            .push(Rc::downgrade(&self.data));
        self.data.borrow_mut().cond_branch(
            cond,
            then_block.clone(),
            then_args.to_vec(),
            else_block.clone(),
            else_args.to_vec(),
        )
    }
}

impl std::cmp::PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

#[derive(Debug, Clone)]
struct BlockData {
    n_args: usize,
    next_var: usize,
    ops: Vec<Op>,
    preceding_blocks: Vec<Weak<RefCell<BlockData>>>,
}

impl PartialEq for BlockData {
    fn eq(&self, other: &Self) -> bool {
        self.n_args == other.n_args
            && self.next_var == other.next_var
            && self.ops == other.ops
            && self.preceding_blocks.len() == other.preceding_blocks.len()
            && self
                .preceding_blocks
                .iter()
                .zip(other.preceding_blocks.iter())
                .all(|(a, b)| Rc::ptr_eq(&a.upgrade().unwrap(), &b.upgrade().unwrap()))
    }
}

impl BlockData {
    fn new(n_args: usize) -> Self {
        BlockData {
            n_args,
            next_var: n_args,
            ops: vec![],
            preceding_blocks: vec![],
        }
    }

    fn parameter(&mut self, idx: usize) -> Var {
        assert!(idx < self.n_args);
        Var(idx)
    }

    fn constant(&mut self, i: i64) -> Var {
        let result = self.new_var();
        self.push_op(Op::Const(result, ConstValue::Integer(i)));
        result
    }

    fn add(&mut self, a: Var, b: Var) -> Var {
        let result = self.new_var();
        self.push_op(Op::Add(result, a, b));
        result
    }

    fn mul(&mut self, a: Var, b: Var) -> Var {
        let result = self.new_var();
        self.push_op(Op::Mul(result, a, b));
        result
    }

    fn terminate(&mut self, x: Var) {
        self.push_op(Op::Term(x))
    }

    fn branch(&mut self, to: Block, args: Vec<Var>) {
        assert_eq!(args.len(), to.n_args());
        self.push_op(Op::Branch(to, args));
    }

    fn cond_branch(
        &mut self,
        cond: Var,
        then_block: Block,
        then_args: Vec<Var>,
        else_block: Block,
        else_args: Vec<Var>,
    ) {
        assert_eq!(then_args.len(), then_block.n_args());
        assert_eq!(else_args.len(), else_block.n_args());
        self.push_op(Op::CondBranch(
            cond, then_block, then_args, else_block, else_args,
        ));
    }

    fn new_var(&mut self) -> Var {
        let i = self.next_var;
        self.next_var += 1;
        Var(i)
    }

    fn push_op(&mut self, op: Op) {
        if let Some(o) = self.ops.last() {
            if o.is_terminal() {
                panic!("Cannot add operations after terminal instruction")
            }
        }
        self.ops.push(op);
    }
}

#[derive(Debug, PartialEq)]
struct LivenessGraph {
    edges: HashMap<Var, HashSet<Var>>,
}

impl LivenessGraph {
    fn new() -> Self {
        LivenessGraph {
            edges: HashMap::new(),
        }
    }

    fn build(block: &Block) -> Self {
        LivenessGraph::build_recursive(&block.data.borrow().ops).0
    }

    fn build_recursive(ops: &[Op]) -> (Self, HashSet<Var>) {
        if ops[0].is_terminal() {
            match &ops[0] {
                Op::Term(v) => {
                    let mut subgraph = LivenessGraph::new();
                    let mut liveset = HashSet::new();
                    liveset.insert(*v);
                    subgraph.add_liveset(&liveset);
                    (subgraph, liveset)
                }
                Op::Branch(_block, args) => {
                    let mut subgraph = LivenessGraph::new();
                    let liveset = args.iter().cloned().collect();
                    subgraph.add_liveset(&liveset);
                    (subgraph, liveset)
                }
                _ => unimplemented!("{:?}", ops[0]),
            }
        } else {
            let (mut subgraph, liveset) = LivenessGraph::build_recursive(&ops[1..]);
            let liveset = LivenessGraph::adjust_liveset(liveset, &ops[0]);
            subgraph.add_liveset(&liveset);
            (subgraph, liveset)
        }
    }

    fn adjust_liveset(mut set: HashSet<Var>, op: &Op) -> HashSet<Var> {
        match op {
            Op::Const(x, _) => {
                set.remove(x);
            }
            Op::Add(c, a, b) | Op::Mul(c, a, b) => {
                set.remove(c);
                set.insert(*a);
                set.insert(*b);
            }
            _ => unimplemented!("{:?}", op),
        }
        set
    }

    fn add_liveset<'a>(&mut self, vars: impl Copy + IntoIterator<Item = &'a Var>) {
        for v in vars {
            self.edges
                .entry(*v)
                .or_default()
                .extend(vars.into_iter().filter(|&w| w != v))
        }
    }

    fn greedy_assign(&self, mut assignment: HashMap<Var, usize>) -> HashMap<Var, usize> {
        let mut order: HashSet<_> = self.edges.keys().cloned().collect();

        while let Some(var) = self.next_var(&mut order, &assignment) {
            let neighbors = &self.edges[&var];

            let mut neighbor_registers: Vec<_> =
                neighbors.iter().filter_map(|r| assignment.get(r)).collect();
            neighbor_registers.sort();
            let n = neighbor_registers.len();

            let register = neighbor_registers
                .into_iter()
                .enumerate()
                .find(|(i, r)| i < r)
                .map(|(i, _)| i)
                .unwrap_or(n);

            assignment.insert(var, register);
        }
        assignment
    }

    fn next_var(
        &self,
        unassigned: &mut HashSet<Var>,
        assigned: &HashMap<Var, usize>,
    ) -> Option<Var> {
        if unassigned.is_empty() {
            return None;
        }

        let var = unassigned
            .iter()
            .map(|v| {
                let neighbors = &self.edges[v];
                let distinct_registers: HashSet<_> =
                    neighbors.iter().filter_map(|n| assigned.get(n)).collect();
                let sub_degree = neighbors
                    .iter()
                    .filter(|n| !assigned.contains_key(n))
                    .count();
                (distinct_registers.len(), sub_degree, *v)
            })
            .fold((0, 0, None), |best, (n_reg, sub_degree, v)| {
                match best.2 {
                    Some(bv)
                        if best.2.is_none()
                            || n_reg > best.0
                            || n_reg == best.0 && sub_degree > best.1
                            || n_reg == best.0 && sub_degree == best.1 && v > bv =>
                    {
                        (n_reg, sub_degree, Some(v))
                    }
                    None => (n_reg, sub_degree, Some(v)),
                    _ => best,
                }
            })
            .2
            .unwrap();

        assert!(unassigned.remove(&var));
        Some(var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ssa() {
        let block = Block::new(1);
        let a = block.parameter(0);
        let b = block.constant(2);
        let a2 = block.mul(a, a);
        let b2 = block.mul(b, b);
        let c = block.add(a2, b2);
        block.terminate(c);

        let d = block.data.borrow();
        assert_eq!(
            *d,
            BlockData {
                n_args: 1,
                next_var: 5,
                ops: vec![
                    Op::Const(Var(1), ConstValue::Integer(2)),
                    Op::Mul(Var(2), Var(0), Var(0)),
                    Op::Mul(Var(3), Var(1), Var(1)),
                    Op::Add(Var(4), Var(2), Var(3)),
                    Op::Term(Var(4)),
                ],
                preceding_blocks: vec![]
            }
        )
    }

    #[test]
    fn branch() {
        let entry = Block::new(1);
        let done = Block::new(2);
        let a = entry.parameter(0);
        let b = entry.constant(42);
        entry.branch(&done, &[a, b]);
        let c = done.parameter(1);
        done.terminate(c);

        let e = entry.data.borrow();
        assert_eq!(
            *e,
            BlockData {
                n_args: 1,
                next_var: 2,
                ops: vec![
                    Op::Const(Var(1), ConstValue::Integer(42)),
                    Op::Branch(done.clone(), vec![Var(0), Var(1)])
                ],
                preceding_blocks: vec![]
            }
        );

        let d = done.data.borrow();
        assert_eq!(
            *d,
            BlockData {
                n_args: 2,
                next_var: 2,
                ops: vec![Op::Term(Var(1))],
                preceding_blocks: vec![Rc::downgrade(&entry.data)]
            }
        );
    }

    #[test]
    fn cond_branch() {
        let entry = Block::new(2);
        let yes = Block::new(1);
        let no = Block::new(0);
        entry.cond_branch(entry.parameter(0), &yes, &[entry.parameter(1)], &no, &[]);
        yes.terminate(yes.parameter(0));
        no.terminate(no.constant(42));

        let e = entry.data.borrow();
        assert_eq!(
            *e,
            BlockData {
                n_args: 2,
                next_var: 2,
                ops: vec![Op::CondBranch(
                    Var(0),
                    yes.clone(),
                    vec![Var(1)],
                    no.clone(),
                    vec![]
                )],
                preceding_blocks: vec![]
            }
        );

        let y = yes.data.borrow();
        assert_eq!(
            *y,
            BlockData {
                n_args: 1,
                next_var: 1,
                ops: vec![Op::Term(Var(0))],
                preceding_blocks: vec![Rc::downgrade(&entry.data)]
            }
        );

        let n = no.data.borrow();
        assert_eq!(
            *n,
            BlockData {
                n_args: 0,
                next_var: 1,
                ops: vec![Op::Const(Var(0), ConstValue::Integer(42)), Op::Term(Var(0))],
                preceding_blocks: vec![Rc::downgrade(&entry.data)]
            }
        );
    }

    #[test]
    fn basic_ssa_liveness_graph() {
        let block = Block::new(1);
        let a = block.parameter(0);
        let b = block.constant(2);
        let a2 = block.mul(a, a);
        let b2 = block.mul(b, b);
        let c = block.add(a2, b2);
        block.terminate(c);

        let mut expected = LivenessGraph::new();
        expected.add_liveset(&[a, b]);
        expected.add_liveset(&[a2, b]);
        expected.add_liveset(&[a2, b2]);
        expected.add_liveset(&[c]);

        assert_eq!(LivenessGraph::build(&block), expected)
    }

    #[test]
    fn basic_ssa_register_assign() {
        let block = Block::new(1);
        let a = block.parameter(0);
        let b = block.constant(2);
        let a2 = block.mul(a, a);
        let b2 = block.mul(b, b);
        let c = block.add(a2, b2);
        block.terminate(c);

        let lg = LivenessGraph::build(&block);

        let mut expected = HashMap::new();
        expected.insert(a, 0);
        expected.insert(b, 1);
        expected.insert(a2, 0);
        expected.insert(b2, 1);
        expected.insert(c, 0);

        assert_eq!(lg.greedy_assign(HashMap::new()), expected)
    }

    #[test]
    fn branch_register_assign() {
        let entry = Block::new(1);
        let done = Block::new(2);
        let a = entry.parameter(0);
        let b = entry.constant(42);
        entry.branch(&done, &[a, b]);
        let c = done.parameter(1);
        done.terminate(c);
    }
}
