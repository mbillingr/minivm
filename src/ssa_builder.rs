use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

macro_rules! set {
    ( ) => { ::std::collections::HashSet::new() };

    ( $($key:expr),* ) => {set!($($key,)*)};

    ( $($key:expr),+ ,) => {
        {
            let mut set = ::std::collections::HashSet::new();
            $(set.insert($key);)*
            set
        }
    };
}

macro_rules! map {
    ( ) => { ::std::collections::HashMap::new() };

    ( $( $key:expr => $val:expr),* ) => {map!($($key => $val,)*)};

    ( $( $key:expr => $val:expr),+ , ) => {{
        let mut map = ::std::collections::HashMap::new();
        $(map.insert($key, $val);)*
        map
    }};
}

#[derive(Debug, Clone)]
pub struct TranslationUnit<V> {
    unit: Rc<RefCell<TransUnitData<V>>>,
}

#[derive(Debug)]
pub struct Block<V> {
    unit: WeakUnit<V>,
    block: Rc<RefCell<BlockData<V>>>,
}

#[derive(Debug, Clone)]
pub struct Var<V> {
    unit: WeakUnit<V>,
    block_id: BlockId,
    name: VarName,
}

impl<V> TranslationUnit<V> {
    pub fn new() -> Self {
        TranslationUnit {
            unit: Rc::new(RefCell::new(TransUnitData::new())),
        }
    }

    pub fn new_block(&self) -> Block<V> {
        let block = Block::new(self);
        self.unit.borrow_mut().push_block(block.clone());
        block
    }

    fn new_var_name(&mut self) -> VarName {
        self.unit.borrow_mut().new_var_name()
    }

    pub fn verify(&self, entry_block: &Block<V>) {
        let mut assigned_vars = HashSet::new();
        entry_block.verify(&mut assigned_vars);
    }

    pub fn allocate_registers(&self, entry_block: &Block<V>) {
        let liveness_graph = self.build_liveness_graph(entry_block);
        let mut data = self.unit.borrow_mut();
        let preassignment = std::mem::replace(&mut data.register_assignment, map![]);
        data.register_assignment = greedy_coloring(&liveness_graph, preassignment);
    }

    fn build_liveness_graph(&self, entry_block: &Block<V>) -> HashMap<VarName, HashSet<VarName>> {
        entry_block.build_liveness_graph().0
    }
}

impl<V> Block<V> {
    pub fn new(unit: &TranslationUnit<V>) -> Self {
        Block {
            unit: unit.into(),
            block: Rc::new(RefCell::new(BlockData::new())),
        }
    }

    pub fn id(&self) -> BlockId {
        BlockId(self.block.as_ptr() as usize)
    }

    pub fn append_parameter(&self) -> Var<V> {
        let var = self.new_var();
        self.block.borrow_mut().append_parameter(var.name);
        var
    }

    pub fn n_params(&self) -> usize {
        self.block.borrow().params.len()
    }

    pub fn constant(&self, c: impl Into<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Const(var.name, c.into()));
        var
    }

    pub fn add(&self, a: &Var<V>, b: &Var<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Add(var.name, a.name, b.name));
        var
    }

    pub fn mul(&self, a: &Var<V>, b: &Var<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Mul(var.name, a.name, b.name));
        var
    }

    pub fn terminate(&self, x: &Var<V>) {
        self.block.borrow_mut().append_op(Op::Return(x.name));
    }

    pub fn branch(&self, target: &Block<V>, args: &[&Var<V>]) {
        let arg_names = args.iter().map(|a| a.name).collect();
        self.block
            .borrow_mut()
            .append_op(Op::Branch(target.clone(), arg_names));
    }

    pub fn branch_conditionally(
        &self,
        cond: &Var<V>,
        then_block: &Block<V>,
        then_args: &[&Var<V>],
        else_block: &Block<V>,
        else_args: &[&Var<V>],
    ) {
        let then_names = then_args.iter().map(|a| a.name).collect();
        let else_names = else_args.iter().map(|a| a.name).collect();
        self.block.borrow_mut().append_op(Op::CondBranch(
            cond.name,
            then_block.clone(),
            then_names,
            else_block.clone(),
            else_names,
        ));
    }

    fn new_var(&self) -> Var<V> {
        let name = self.unit.upgrade().new_var_name();
        Var::new(self.unit.clone(), self, name)
    }

    fn verify(&self, assigned_vars: &mut HashSet<VarName>) {
        let block = self.block.borrow();
        assert!(block.ops.last().map(Op::is_terminal).unwrap_or(true));
        assigned_vars.extend(&block.params);
        for op in &block.ops {
            op.verify(assigned_vars);
        }
    }

    fn build_liveness_graph(&self) -> (HashMap<VarName, HashSet<VarName>>, HashSet<VarName>) {
        let block = self.block.borrow();

        let mut liveset = set![];
        let mut subgraph = map![];

        for op in block.ops.iter().rev() {
            let (g, l) = op.update_liveness_graph((subgraph, liveset));
            liveset = l;
            subgraph = g;
            println!("{:?}: {:?}", self.id(), liveset);
        }

        for p in &block.params {
            liveset.remove(p);
        }

        (subgraph, liveset)
    }
}

impl<V> Clone for Block<V> {
    fn clone(&self) -> Self {
        Block {
            unit: self.unit.clone(),
            block: self.block.clone(),
        }
    }
}

impl<V> Var<V> {
    fn new(unit: WeakUnit<V>, block: &Block<V>, name: VarName) -> Self {
        Var {
            unit,
            block_id: block.id(),
            name,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BlockId(usize);

#[derive(Debug)]
struct WeakUnit<V> {
    data: Weak<RefCell<TransUnitData<V>>>,
}

impl<V> Clone for WeakUnit<V> {
    fn clone(&self) -> Self {
        WeakUnit {
            data: self.data.clone(),
        }
    }
}

impl<V> From<&TranslationUnit<V>> for WeakUnit<V> {
    fn from(tu: &TranslationUnit<V>) -> Self {
        WeakUnit {
            data: Rc::downgrade(&tu.unit),
        }
    }
}

impl<V> WeakUnit<V> {
    fn upgrade(&self) -> TranslationUnit<V> {
        TranslationUnit {
            unit: self
                .data
                .upgrade()
                .expect("Translation unit has been deallocated"),
        }
    }
}

#[derive(Debug)]
struct TransUnitData<V> {
    blocks: Vec<Block<V>>,
    next_var: VarName,
    register_assignment: HashMap<VarName, usize>,
}

impl<V> TransUnitData<V> {
    fn new() -> Self {
        TransUnitData {
            blocks: vec![],
            next_var: VarName::default(),
            register_assignment: HashMap::new(),
        }
    }

    fn push_block(&mut self, block: Block<V>) {
        self.blocks.push(block)
    }

    fn new_var_name(&mut self) -> VarName {
        self.next_var.new_name()
    }
}

#[derive(Debug)]
struct BlockData<V> {
    ops: Vec<Op<V>>,
    params: Vec<VarName>,
}

impl<V> BlockData<V> {
    fn new() -> Self {
        BlockData {
            ops: vec![],
            params: vec![],
        }
    }

    fn append_parameter(&mut self, varname: VarName) {
        self.params.push(varname);
    }

    fn append_op(&mut self, op: Op<V>) {
        assert!(self.ops.last().map(Op::is_nonterminal).unwrap_or(true));
        self.ops.push(op);
    }
}

#[derive(Debug, Copy, Clone, Default, PartialOrd, PartialEq, Eq, Hash)]
struct VarName(usize);

impl VarName {
    fn new_name(&mut self) -> Self {
        let old_name = *self;
        self.0 += 1;
        old_name
    }
}

#[derive(Debug)]
struct Function {}

impl Function {
    fn n_params(&self) -> usize {
        unimplemented!()
    }

    fn get_argument_registers(&self) -> Option<&[usize]> {
        unimplemented!()
    }
    
    fn set_argument_registers(&mut self, registers: &[usize]) {
        unimplemented!()
    }
}

#[derive(Debug)]
enum Op<V> {
    Const(VarName, V),
    Add(VarName, VarName, VarName),
    Mul(VarName, VarName, VarName),

    Branch(Block<V>, Vec<VarName>),
    CondBranch(VarName, Block<V>, Vec<VarName>, Block<V>, Vec<VarName>),

    Return(VarName),
    Call(VarName, Function, Vec<VarName>),
    TailCall(Function, Vec<VarName>),

    CallDynamic(VarName, VarName, Vec<VarName>),
    TailCallDynamic(VarName, Vec<VarName>),
}

impl<V> Op<V> {
    fn is_terminal(&self) -> bool {
        match self {
            Op::Return(_)
            | Op::Branch(_, _)
            | Op::CondBranch(_, _, _, _, _)
            | Op::TailCall(_, _)
            | Op::TailCallDynamic(_, _) => true,
            _ => false,
        }
    }

    fn is_nonterminal(&self) -> bool {
        !self.is_terminal()
    }

    fn verify(&self, assigned_vars: &mut HashSet<VarName>) {
        match self {
            Op::Const(v, _) => {
                assigned_vars.insert(*v);
            }
            Op::Add(z, a, b) | Op::Mul(z, a, b) => {
                assert!(assigned_vars.contains(a));
                assert!(assigned_vars.contains(b));
                assigned_vars.insert(*z);
            }
            Op::Return(a) => assert!(assigned_vars.contains(a)),
            Op::Branch(block, args) => {
                for a in args {
                    assert!(assigned_vars.contains(a));
                }
                assert_eq!(args.len(), block.n_params());
                block.verify(assigned_vars)
            }
            Op::CondBranch(cond, then_block, then_args, else_block, else_args) => {
                assert!(assigned_vars.contains(cond));
                for a in then_args {
                    assert!(assigned_vars.contains(a));
                }
                for a in else_args {
                    assert!(assigned_vars.contains(a));
                }
                then_block.verify(&mut assigned_vars.clone());
                else_block.verify(assigned_vars);
            }
            Op::Call(z, func, args) => {
                for a in args {
                    assert!(assigned_vars.contains(a));
                }
                assert_eq!(args.len(), func.n_params());
                assigned_vars.insert(*z);
            }
            Op::CallDynamic(z, func, args) => {
                assert!(assigned_vars.contains(func));
                for a in args {
                    assert!(assigned_vars.contains(a));
                }
                assigned_vars.insert(*z);
            }
            Op::TailCall(func, args) => {
                for a in args {
                    assert!(assigned_vars.contains(a));
                }
                assert_eq!(args.len(), func.n_params());
            }
            Op::TailCallDynamic(func, args) => {
                assert!(assigned_vars.contains(func));
                for a in args {
                    assert!(assigned_vars.contains(a));
                }
            }
        }
    }

    fn update_liveness_graph(
        &self,
        (mut graph, mut liveset): (HashMap<VarName, HashSet<VarName>>, HashSet<VarName>),
    ) -> (HashMap<VarName, HashSet<VarName>>, HashSet<VarName>) {
        match self {
            Op::Const(z, _) => {
                liveset.remove(z);
            }
            Op::Add(z, a, b) | Op::Mul(z, a, b) => {
                liveset.remove(z);
                liveset.insert(*a);
                liveset.insert(*b);
            }
            Op::Return(a) => {
                liveset.insert(*a);
            }
            Op::Branch(blk, args) => {
                let (g, l) = blk.build_liveness_graph();
                graph = g;
                liveset = l;
                liveset.extend(args);
            }
            Op::CondBranch(cond, blk1, args1, blk2, args2) => {
                let (g1, l1) = blk1.build_liveness_graph();
                let (g2, l2) = blk2.build_liveness_graph();
                liveset = l1.union(&l2).cloned().collect();
                graph = join_edges(g1, g2);
                liveset.insert(*cond);
                liveset.extend(args1);
                liveset.extend(args2);
            }
            Op::Call(z, _, args) => {
                liveset.remove(z);
                liveset.extend(args);
            }
            Op::CallDynamic(z, func, args) => {
                liveset.remove(z);
                liveset.insert(*func);
                liveset.extend(args);
            }
            Op::TailCall(_, args) => {
                liveset.extend(args);
            }
            Op::TailCallDynamic(func, args) => {
                liveset.insert(*func);
                liveset.extend(args);
            }
        }

        for a in &liveset {
            graph
                .entry(*a)
                .or_default()
                .extend(liveset.iter().filter(|&b| a != b));
        }

        (graph, liveset)
    }
}

impl<V> PartialEq for Op<V> {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Op::Const(z1, _), Op::Const(z2, _)) => z1 == z2,
            (Op::Add(z1, a1, b1), Op::Add(z2, a2, b2))
            | (Op::Mul(z1, a1, b1), Op::Mul(z2, a2, b2)) => (z1, a1, b1) == (z2, a2, b2),
            (Op::Return(z1), Op::Return(z2)) => z1 == z2,
            (Op::Branch(bl1, args1), Op::Branch(bl2, args2)) => {
                bl1.id() == bl2.id() && args1 == args2
            }
            (Op::CondBranch(co1, a1, b1, c1, d1), Op::CondBranch(co2, a2, b2, c2, d2)) => {
                co1 == co2 && a1.id() == a2.id() && b1 == b2 && c1.id() == c2.id() && d1 == d2
            }
            _ => false,
        }
    }
}

fn join_edges<K: std::hash::Hash + Eq>(
    mut g1: HashMap<K, HashSet<K>>,
    g2: HashMap<K, HashSet<K>>,
) -> HashMap<K, HashSet<K>> {
    for (node, neighbors) in g2 {
        g1.entry(node).or_default().extend(neighbors)
    }
    g1
}

fn greedy_coloring<K: std::hash::Hash + Eq + PartialOrd + Clone>(
    graph: &HashMap<K, HashSet<K>>,
    mut assignment: HashMap<K, usize>,
) -> HashMap<K, usize> {
    let mut remaining_nodes: HashSet<_> = graph.keys().cloned().collect();

    while let Some(node) = next_node(graph, &mut remaining_nodes, &assignment) {
        let neighbors = &graph[&node];

        let neighbor_colors: Vec<_> = neighbors
            .iter()
            .filter_map(|n| assignment.get(n))
            .cloned()
            .collect();

        assignment.insert(node, find_smallest_color(neighbor_colors));
    }

    assignment
}

fn next_node<K: std::hash::Hash + Eq + PartialOrd + Clone>(
    graph: &HashMap<K, HashSet<K>>,
    remaining_nodes: &mut HashSet<K>,
    assignment: &HashMap<K, usize>,
) -> Option<K> {
    if remaining_nodes.is_empty() {
        return None;
    }

    let node = remaining_nodes
        .iter()
        .cloned()
        .map(|node| {
            let neighbors = &graph[&node];
            let distinct_colors: HashSet<_> =
                neighbors.iter().filter_map(|n| assignment.get(n)).collect();
            let unassigned_degree = neighbors
                .iter()
                .filter(|n| !assignment.contains_key(n))
                .count();
            (distinct_colors.len(), unassigned_degree, node)
        })
        .fold(
            (0, 0, None),
            |best, (n_colors, sub_degree, node)| match best.2 {
                Some(ref bv)
                    if best.2.is_none()
                        || n_colors > best.0
                        || n_colors == best.0 && sub_degree > best.1
                        || n_colors == best.0 && sub_degree == best.1 && node > *bv =>
                {
                    (n_colors, sub_degree, Some(node))
                }
                None => (n_colors, sub_degree, Some(node)),
                _ => best,
            },
        )
        .2
        .unwrap();

    remaining_nodes.remove(&node);
    Some(node)
}

fn find_smallest_color(mut neighbor_colors: Vec<usize>) -> usize {
    let n_colors = neighbor_colors.len();
    neighbor_colors.sort();
    neighbor_colors
        .into_iter()
        .enumerate()
        .find(|(i, r)| i < r)
        .map(|(i, _)| i)
        .unwrap_or(n_colors)
}

#[cfg(test)]
mod tests {
    use super::*;

    impl<V> PartialEq<Vec<Op<V>>> for Block<V> {
        fn eq(&self, rhs: &Vec<Op<V>>) -> bool {
            self.block.borrow_mut().ops == *rhs
        }
    }

    #[test]
    fn create_blocks() {
        let tu = TranslationUnit::<()>::new();
        let entry = tu.new_block();
        let exit = tu.new_block();
        entry.terminate(&entry.constant(()));
        exit.terminate(&exit.constant(()));

        tu.verify(&entry);
        assert_eq!(
            entry,
            vec![Op::Const(VarName(0), ()), Op::Return(VarName(0))]
        );
        assert_eq!(
            exit,
            vec![Op::Const(VarName(1), ()), Op::Return(VarName(1))]
        );
    }

    #[test]
    fn add_parameters() {
        let tu = TranslationUnit::<i32>::new();
        let entry = tu.new_block();
        let x = entry.append_parameter();
        let y = entry.append_parameter();
        let k = entry.constant(2);
        let z = entry.mul(&x, &y);
        let z = entry.add(&z, &k);
        entry.terminate(&z);

        tu.verify(&entry);
        assert_eq!(
            entry,
            vec![
                Op::Const(VarName(2), 2),
                Op::Mul(VarName(3), VarName(0), VarName(1)),
                Op::Add(VarName(4), VarName(3), VarName(2)),
                Op::Return(VarName(4))
            ]
        );
    }

    #[test]
    #[should_panic]
    fn invalid_variable_usage() {
        let tu = TranslationUnit::<()>::new();
        let entry = tu.new_block();
        let exit = tu.new_block();
        let x = exit.append_parameter();
        exit.terminate(&x);
        entry.terminate(&x);
        tu.verify(&entry);
    }

    #[test]
    fn branch() {
        let tu = TranslationUnit::<()>::new();
        let entry = tu.new_block();
        let exit = tu.new_block();
        let x = entry.append_parameter();
        entry.branch(&exit, &[]);
        exit.terminate(&x);

        tu.verify(&entry);

        assert_eq!(exit, vec![Op::Return(VarName(0)),]);

        assert_eq!(entry, vec![Op::Branch(exit, vec![]),]);
    }

    #[test]
    fn branch_with_args() {
        let tu = TranslationUnit::<()>::new();
        let entry = tu.new_block();
        let exit = tu.new_block();
        let x = entry.append_parameter();
        entry.branch(&exit, &[&x]);
        let y = exit.append_parameter();
        exit.terminate(&y);

        tu.verify(&entry);

        assert_eq!(exit, vec![Op::Return(VarName(1)),]);

        assert_eq!(entry, vec![Op::Branch(exit, vec![VarName(0)]),]);
    }

    #[test]
    #[should_panic]
    fn branch_with_missing_args() {
        let tu = TranslationUnit::<()>::new();
        let entry = tu.new_block();
        let exit = tu.new_block();
        entry.branch(&exit, &[]);
        let y = exit.append_parameter();
        exit.terminate(&y);
        tu.verify(&entry);
    }

    #[test]
    #[should_panic]
    fn branch_with_too_many_args() {
        let tu = TranslationUnit::<()>::new();
        let entry = tu.new_block();
        let exit = tu.new_block();
        let x = entry.append_parameter();
        entry.branch(&exit, &[&x]);
        exit.terminate(&x);
        tu.verify(&entry);
    }

    #[test]
    fn conditional_branch() {
        let tu = TranslationUnit::<()>::new();
        let entry = tu.new_block();
        let yes = tu.new_block();
        let no = tu.new_block();
        let c = entry.append_parameter();
        entry.branch_conditionally(&c, &yes, &[], &no, &[]);
        yes.terminate(&yes.constant(()));
        no.terminate(&no.constant(()));

        tu.verify(&entry);

        assert_eq!(
            yes,
            vec![Op::Const(VarName(1), ()), Op::Return(VarName(1)),]
        );

        assert_eq!(no, vec![Op::Const(VarName(2), ()), Op::Return(VarName(2)),]);

        assert_eq!(
            entry,
            vec![Op::CondBranch(VarName(0), yes, vec![], no, vec![]),]
        );
    }

    #[test]
    fn conditional_branch_with_args() {
        let tu = TranslationUnit::<i64>::new();
        let entry = tu.new_block();
        let yes = tu.new_block();
        let exit = tu.new_block();
        let c = entry.append_parameter();
        let x = entry.append_parameter();
        entry.branch_conditionally(&c, &yes, &[&x], &exit, &[&x]);
        let y = yes.append_parameter();
        let y = yes.add(&y, &yes.constant(100));
        yes.branch(&exit, &[&y]);
        let z = exit.append_parameter();
        exit.terminate(&z);

        tu.verify(&entry);

        assert_eq!(exit, vec![Op::Return(VarName(5)),]);

        assert_eq!(
            yes,
            vec![
                Op::Const(VarName(3), 100),
                Op::Add(VarName(4), VarName(2), VarName(3)),
                Op::Branch(exit.clone(), vec![VarName(4)]),
            ]
        );

        assert_eq!(
            entry,
            vec![Op::CondBranch(
                VarName(0),
                yes,
                vec![VarName(1)],
                exit,
                vec![VarName(1)]
            ),]
        );
    }

    #[test]
    #[should_panic]
    fn conditional_branch_assignment_error() {
        let tu = TranslationUnit::<()>::new();
        let entry = tu.new_block();
        let yes = tu.new_block();
        let no = tu.new_block();
        let exit = tu.new_block();
        let c = entry.append_parameter();
        entry.branch_conditionally(&c, &yes, &[], &no, &[]);
        let _ = yes.constant(());
        yes.branch(&exit, &[]);
        let x = no.constant(());
        no.branch(&exit, &[]);
        exit.terminate(&x);
        tu.verify(&entry);
    }

    #[test]
    fn conditional_branch_outer_variable() {
        let tu = TranslationUnit::<()>::new();
        let entry = tu.new_block();
        let yes = tu.new_block();
        let no = tu.new_block();
        let exit = tu.new_block();
        let c = entry.append_parameter();
        let x = entry.append_parameter();
        entry.branch_conditionally(&c, &yes, &[], &no, &[]);
        yes.branch(&exit, &[]);
        no.branch(&exit, &[]);
        exit.terminate(&x);

        tu.verify(&entry);

        assert_eq!(exit, vec![Op::Return(VarName(1)),]);

        assert_eq!(yes, vec![Op::Branch(exit.clone(), vec![]),]);

        assert_eq!(no, vec![Op::Branch(exit.clone(), vec![]),]);

        assert_eq!(
            entry,
            vec![Op::CondBranch(VarName(0), yes, vec![], no, vec![]),]
        );
    }

    #[test]
    fn simple_liveness_graph() {
        let tu = TranslationUnit::<i32>::new();
        let entry = tu.new_block();
        let x = entry.append_parameter();
        let y = entry.append_parameter();
        // alive: 0, 1
        let k = entry.constant(2);
        // alive: 2, 0, 1
        let z = entry.mul(&x, &y);
        // alive: 3, 2
        let z = entry.add(&z, &k);
        // alive: 4
        entry.terminate(&z);

        tu.verify(&entry);

        assert_eq!(
            tu.build_liveness_graph(&entry),
            map![
                    VarName(4) => set![],
                    VarName(3) => set![VarName(2)],
                    VarName(2) => set![VarName(0), VarName(1), VarName(3)],
                    VarName(1) => set![VarName(0), VarName(2)],
                    VarName(0) => set![VarName(1), VarName(2)],
            ]
        );
    }

    #[test]
    fn branch_liveness_graph() {
        let tu = TranslationUnit::<i32>::new();
        let entry = tu.new_block();
        let exit = tu.new_block();

        let x = entry.append_parameter();
        let y = entry.append_parameter();
        // alive: 0, 1
        let z = entry.add(&x, &y);
        // alive: 1, 2
        entry.branch(&exit, &[&z]);

        let z = exit.append_parameter();
        // alive: 1, 3
        let w = exit.add(&z, &y);
        //alive: 4
        exit.terminate(&w);

        tu.verify(&entry);

        assert_eq!(
            tu.build_liveness_graph(&entry),
            map![
                VarName(4) => set![],
                VarName(3) => set![VarName(1)],
                VarName(2) => set![VarName(1)],
                VarName(1) => set![VarName(0), VarName(2), VarName(3)],
                VarName(0) => set![VarName(1)],
            ]
        );
    }

    #[test]
    fn conditional_branch_with_args_liveness_graph() {
        let tu = TranslationUnit::<i64>::new();
        let entry = tu.new_block();
        let yes = tu.new_block();
        let exit = tu.new_block();
        let c = entry.append_parameter();
        let x = entry.append_parameter();
        // alive: 0, 1
        entry.branch_conditionally(&c, &yes, &[&x], &exit, &[&x]);

        let y = yes.append_parameter();
        // alive: 0, 2
        let k = yes.constant(100);
        // alive: 0, 2, 3
        let y = yes.add(&y, &k);
        // alive: 0, 4
        yes.branch(&exit, &[&y]);

        let z = exit.append_parameter();
        // alive: 0, 5
        let w = exit.add(&z, &c);
        // alive: 6
        exit.terminate(&w);

        tu.verify(&entry);

        assert_eq!(
            tu.build_liveness_graph(&entry),
            map![
                VarName(6) => set![],
                VarName(5) => set![VarName(0)],
                VarName(4) => set![VarName(0)],
                VarName(3) => set![VarName(0), VarName(2)],
                VarName(2) => set![VarName(0), VarName(3)],
                VarName(1) => set![VarName(0)],
                VarName(0) => set![VarName(5), VarName(4), VarName(3), VarName(2), VarName(1)],
            ]
        );
    }

    #[test]
    fn assign_registers_fully_connected() {
        let graph = map![
            'A' => set!['B', 'C'],
            'B' => set!['A', 'C'],
            'C' => set!['A', 'B']
        ];
        assert_eq!(
            greedy_coloring(&graph, HashMap::new()),
            map!['A' => 2, 'B' => 1, 'C' => 0]
        );
    }

    #[test]
    fn assign_registers_fully_unconnected() {
        let graph = map![
            'A' => set![],
            'B' => set![],
            'C' => set![]
        ];
        assert_eq!(
            greedy_coloring(&graph, HashMap::new()),
            map!['A' => 0, 'B' => 0, 'C' => 0]
        );
    }

    #[test]
    fn assign_registers_simple_cycle() {
        let graph = map![
            'A' => set!['E', 'B'],
            'B' => set!['A', 'C'],
            'C' => set!['B', 'D'],
            'D' => set!['C', 'E'],
            'E' => set!['D', 'A']
        ];
        assert_eq!(
            greedy_coloring(&graph, HashMap::new()),
            map!['A' => 2, 'B' => 1, 'C' => 0, 'D' => 1, 'E' => 0]
        );
    }
}
