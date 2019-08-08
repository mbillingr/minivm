use crate::assembler::Assembler;
use crate::memory::RecordStorage;
use crate::primitive_value::{CodePos, PrimitiveValue};
use crate::uid::global_uid;
use crate::virtual_machine as vm;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::iter::once;
use std::rc::{Rc, Weak};

pub const RETURN_TARGET_REGISTER: vm::Register = 0;
pub const STACK_POINTER_REGISTER: vm::Register = 1;
pub const STACK_REGISTER: vm::Register = 2;
const FIRST_GENERAL_PURPOSE_REGISTER: vm::Register = 3;
pub const RETURN_VALUE_REGISTER: vm::Register = 3;
pub const FIRST_ARG_REGISTER: vm::Register = 4;

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

pub fn eval(vm_code: CodePos) -> PrimitiveValue {
    let func = PrimitiveValue::CodeBlock(vm_code);
    let storage = RecordStorage::new(0);
    vm::run(&SSA_MAIN, &storage, vec![func])
}

static SSA_MAIN: [vm::Op; 7] = [
    vm::Op::Copy(FIRST_GENERAL_PURPOSE_REGISTER, 0),
    vm::Op::Alloc(STACK_REGISTER, 100),
    vm::Op::Const(STACK_POINTER_REGISTER, PrimitiveValue::Integer(0)),
    vm::Op::LoadLabel(RETURN_TARGET_REGISTER, 2),
    vm::Op::Jmp(vm::Operand::R(FIRST_GENERAL_PURPOSE_REGISTER)),
    vm::Op::Copy(0, RETURN_VALUE_REGISTER),
    vm::Op::Term,
];

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

impl<V: std::fmt::Debug> TranslationUnit<V> {
    pub fn new() -> Self {
        TranslationUnit {
            unit: Rc::new(RefCell::new(TransUnitData::new())),
        }
    }

    pub fn new_function(&self, entry_block: &Block<V>) -> Block<V> {
        let param_block = self.new_block();
        let eb = entry_block.block.borrow();

        let func_params: Vec<_> = eb
            .params
            .iter()
            .zip(FIRST_ARG_REGISTER..)
            .map(|(_, r)| {
                let param = param_block.new_var();
                self.set_allocation(&param, r);
                param
            })
            .collect();

        let tmp_vars: Vec<_> = func_params.iter().map(|p| param_block.copy(p)).collect();

        param_block.branch(entry_block, &tmp_vars[..]);

        param_block
    }

    pub fn new_block(&self) -> Block<V> {
        let block = Block::new(self);
        self.unit.borrow_mut().push_block(block.clone());
        block
    }

    fn new_var_name(&self) -> VarName {
        self.unit.borrow_mut().new_var_name()
    }

    pub fn verify(&self, entry_block: &Block<V>) {
        let mut assigned_vars = HashSet::new();
        entry_block.verify(&mut assigned_vars);
    }

    pub fn allocate_registers(&self, entry_block: &Block<V>) {
        let mut preassignment =
            std::mem::replace(&mut self.unit.borrow_mut().register_assignment, map![]);
        entry_block.preassign_function_arg_registers(&mut preassignment);
        //println!("{:?}", entry_block.block.borrow().ops);
        let liveness_graph = LivenessGraph::build(entry_block);
        //println!("{:?}", preassignment);
        //println!("{:?}", liveness_graph.edges);
        self.unit.borrow_mut().register_assignment = greedy_coloring(
            &liveness_graph.edges,
            preassignment,
            &liveness_graph.preference_pairs,
            FIRST_GENERAL_PURPOSE_REGISTER,
        );
    }

    pub fn get_allocation(&self, var: impl Into<VarName>) -> Option<vm::Register> {
        self.unit
            .borrow()
            .register_assignment
            .get(&var.into())
            .cloned()
    }

    pub fn set_allocation(&self, var: impl Into<VarName>, reg: vm::Register) {
        self.unit
            .borrow_mut()
            .register_assignment
            .insert(var.into(), reg);
    }

    pub fn preassign_function_arg_registers(&self, entry_block: &Block<V>) {
        let mut data = self.unit.borrow_mut();
        entry_block.preassign_function_arg_registers(&mut data.register_assignment);
    }
}

impl<V> From<&Var<V>> for VarName {
    fn from(var: &Var<V>) -> Self {
        var.name
    }
}

impl<V> From<&&Var<V>> for VarName {
    fn from(var: &&Var<V>) -> Self {
        var.name
    }
}

impl<V: std::fmt::Debug> Block<V> {
    pub fn new(unit: &TranslationUnit<V>) -> Self {
        Block {
            unit: unit.into(),
            block: Rc::new(RefCell::new(BlockData::new())),
        }
    }

    pub fn id(&self) -> BlockId {
        BlockId(self.block.as_ptr() as usize)
    }

    pub fn create_sibling(&self) -> Block<V> {
        self.unit.upgrade().new_block()
    }

    pub fn into_function(self) -> Block<V> {
        self.unit.upgrade().new_function(&self)
    }

    pub fn append_parameter(&self) -> Var<V> {
        let var = self.new_var();
        self.block.borrow_mut().append_parameter(var.name);
        var
    }

    pub fn n_params(&self) -> usize {
        self.block.borrow().params.len()
    }

    pub fn params(&self) -> Vec<Var<V>> {
        self.block
            .borrow()
            .params
            .iter()
            .map(|&name| Var::new(self.unit.clone(), self, name))
            .collect()
    }

    pub fn nop(&self) {
        self.block.borrow_mut().append_op(Op::Nop);
    }

    pub fn constant(&self, c: impl Into<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Const(var.name, c.into()));
        var
    }

    pub fn copy(&self, x: &Var<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Copy(var.name, x.name));
        var
    }

    pub fn add(&self, a: &Var<V>, b: &Var<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Add(var.name, a.name, b.name));
        var
    }

    pub fn sub(&self, a: &Var<V>, b: &Var<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Sub(var.name, a.name, b.name));
        var
    }

    pub fn mul(&self, a: &Var<V>, b: &Var<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Mul(var.name, a.name, b.name));
        var
    }

    pub fn div(&self, a: &Var<V>, b: &Var<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Div(var.name, a.name, b.name));
        var
    }

    pub fn get_cell(&self, cell: &Var<V>) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::GetCell(var.name, cell.name));
        var
    }

    pub fn make_rec(&self, n: usize) -> Var<V> {
        let var = self.new_var();
        self.block.borrow_mut().append_op(Op::Alloc(var.name, n));
        var
    }

    pub fn get_rec(&self, rec: &Var<V>, idx: usize) -> Var<V> {
        let var = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::GetRec(var.name, rec.name, idx));
        var
    }

    pub fn set_rec(&self, rec: &Var<V>, idx: usize, val: &Var<V>) {
        self.block
            .borrow_mut()
            .append_op(Op::SetRec(rec.name, idx, val.name));
    }

    pub fn equals(&self, a: &Var<V>, b: &Var<V>) -> Var<V> {
        let result = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Equal(result.name, a.name, b.name));
        result
    }

    pub fn terminate(&self, x: &Var<V>) {
        self.block.borrow_mut().append_op(Op::Return(x.name));
    }

    pub fn branch<'a, T>(&self, target: &Block<V>, args: &'a [T])
    where
        &'a T: Into<VarName>,
    {
        let arg_names = args.iter().map(|a| a.into()).collect();
        self.block
            .borrow_mut()
            .append_op(Op::Branch(target.clone(), arg_names));
    }

    pub fn branch_conditionally(
        &self,
        cond: &Var<V>,
        then_block: &Block<V>,
        //then_args: &[&Var<V>],
        else_block: &Block<V>,
        //else_args: &[&Var<V>],
    ) {
        //let then_names = then_args.iter().map(|a| a.name).collect();
        //let else_names = else_args.iter().map(|a| a.name).collect();
        self.block.borrow_mut().append_op(Op::CondBranch(
            cond.name,
            then_block.clone(),
            //then_names,
            else_block.clone(),
            //else_names,
        ));
    }

    pub fn return_(&self, var: &Var<V>) {
        let retvar = self.copy(var);
        self.block.borrow_mut().append_op(Op::Return(retvar.name));
    }

    pub fn tail_call(&self, func: &Var<V>, args: &[&Var<V>]) {
        let arg_names = args.iter().map(|a| self.copy(a).name).collect();
        self.block
            .borrow_mut()
            .append_op(Op::TailCallDynamic(func.name, arg_names));
    }

    pub fn call(&self, func: &Var<V>, args: &[&Var<V>]) -> Var<V> {
        let ret = self.new_var();
        let arg_names = args.iter().map(|a| self.copy(a).name).collect();
        self.block
            .borrow_mut()
            .append_op(Op::CallDynamic(ret.name, func.name, arg_names));
        self.copy(&ret)
    }

    pub fn tail_call_static(&self, func: &Block<V>, args: &[&Var<V>]) {
        let arg_names = args.iter().map(|a| self.copy(a).name).collect();
        self.block
            .borrow_mut()
            .append_op(Op::TailCall(func.clone(), arg_names));
    }

    pub fn call_static(&self, func: &Block<V>, args: &[&Var<V>]) -> Var<V> {
        let ret = self.new_var();
        let arg_names = args.iter().map(|a| self.copy(a).name).collect();
        self.block
            .borrow_mut()
            .append_op(Op::Call(ret.name, func.clone(), arg_names));
        self.copy(&ret)
    }

    pub fn label(&self, func: &Block<V>) -> Var<V> {
        let ret = self.new_var();
        self.block
            .borrow_mut()
            .append_op(Op::Label(ret.name, func.clone()));
        ret
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

    fn preassign_function_arg_registers(&self, assignment: &mut HashMap<VarName, vm::Register>) {
        loop {
            let mut conflicts = vec![];
            {
                let block = self.block.borrow();
                'conflict: for (i, op) in block.ops.iter().enumerate() {
                    match op {
                        Op::CallDynamic(retval, _, args) | Op::Call(retval, _, args) => {
                            for (var, reg) in args
                                .iter()
                                .cloned()
                                .zip(FIRST_ARG_REGISTER..)
                                .chain(once((*retval, RETURN_VALUE_REGISTER)))
                            {
                                if let Some(&r) = assignment.get(&var) {
                                    if r != reg {
                                        conflicts.push((var, i));
                                    }
                                }
                                if conflicts.is_empty() {
                                    assignment.insert(var, reg);
                                }
                            }
                        }
                        Op::TailCallDynamic(_, args) | Op::TailCall(_, args) => {
                            for (var, reg) in args.iter().cloned().zip(FIRST_ARG_REGISTER..) {
                                if let Some(&r) = assignment.get(&var) {
                                    if r != reg {
                                        conflicts.push((var, i));
                                    }
                                }
                                if conflicts.is_empty() {
                                    assignment.insert(var, reg);
                                }
                            }
                        }
                        Op::Return(var) => {
                            if let Some(&r) = assignment.get(var) {
                                if r != RETURN_VALUE_REGISTER {
                                    conflicts.push((*var, i));
                                }
                            }
                            if conflicts.is_empty() {
                                assignment.insert(*var, RETURN_VALUE_REGISTER);
                            }
                        }
                        Op::Branch(blk, _) => blk.preassign_function_arg_registers(assignment),
                        Op::CondBranch(_, blk1, blk2) => {
                            blk1.preassign_function_arg_registers(assignment);
                            blk2.preassign_function_arg_registers(assignment);
                        }
                        _ => {}
                    };
                }
            }

            if conflicts.is_empty() {
                return;
            } else {
                let mut idx_shift = 0;
                for (var, op_idx) in conflicts {
                    let op_idx = op_idx + idx_shift * 2;
                    let tmp_var1 = self.new_var();
                    let tmp_var2 = self.new_var();
                    let mut block = self.block.borrow_mut();
                    block.ops[op_idx].replace_var(var, tmp_var2.name);
                    block
                        .ops
                        .insert(op_idx - idx_shift, Op::Copy(tmp_var2.name, tmp_var1.name));
                    block
                        .ops
                        .insert(op_idx - idx_shift, Op::Copy(tmp_var1.name, var));
                    idx_shift += 1;
                }
            }
        }
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

impl<V: std::fmt::Debug> Var<V> {
    fn new(unit: WeakUnit<V>, block: &Block<V>, name: VarName) -> Self {
        Var {
            unit,
            block_id: block.id(),
            name,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
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
    register_assignment: HashMap<VarName, vm::Register>,
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

impl<V: std::fmt::Debug> BlockData<V> {
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
pub struct VarName(usize);

impl VarName {
    fn new_name(&mut self) -> Self {
        let old_name = *self;
        self.0 += 1;
        old_name
    }
}

#[derive(Debug)]
enum Op<V> {
    Nop,
    Const(VarName, V),
    Add(VarName, VarName, VarName),
    Sub(VarName, VarName, VarName),
    Mul(VarName, VarName, VarName),
    Div(VarName, VarName, VarName),

    Alloc(VarName, usize),
    GetRec(VarName, VarName, usize),
    SetRec(VarName, usize, VarName),
    GetCell(VarName, VarName),

    Equal(VarName, VarName, VarName),

    Copy(VarName, VarName),

    Branch(Block<V>, Vec<VarName>),
    CondBranch(VarName, Block<V>, Block<V>),

    Return(VarName),
    Call(VarName, Block<V>, Vec<VarName>),
    TailCall(Block<V>, Vec<VarName>),
    Label(VarName, Block<V>),

    CallDynamic(VarName, VarName, Vec<VarName>),
    TailCallDynamic(VarName, Vec<VarName>),
}

impl<V: std::fmt::Debug> Op<V> {
    fn is_terminal(&self) -> bool {
        match self {
            Op::Return(_)
            | Op::Branch(_, _)
            | Op::CondBranch(_, _, _)
            | Op::TailCall(_, _)
            | Op::TailCallDynamic(_, _) => true,
            _ => false,
        }
    }

    fn is_nonterminal(&self) -> bool {
        !self.is_terminal()
    }

    fn replace_var(&mut self, old: VarName, new: VarName) {
        match self {
            Op::Nop => {}
            Op::Const(v, _) | Op::Return(v) | Op::Label(v, _) | Op::Alloc(v, _) => {
                if v == &old {
                    *v = new
                }
            }
            Op::Add(a, b, c)
            | Op::Sub(a, b, c)
            | Op::Mul(a, b, c)
            | Op::Div(a, b, c)
            | Op::Equal(a, b, c) => {
                if a == &old {
                    *a = new
                }
                if b == &old {
                    *b = new
                }
                if c == &old {
                    *c = new
                }
            }
            Op::Copy(a, b) | Op::GetCell(a, b) | Op::GetRec(a, b, _) | Op::SetRec(a, _, b) => {
                if a == &old {
                    *a = new
                }
                if b == &old {
                    *b = new
                }
            }
            Op::Branch(_, args) | Op::TailCall(_, args) => {
                for a in args {
                    if a == &old {
                        *a = new
                    }
                }
            }
            Op::CondBranch(c, _, _) => {
                if c == &old {
                    *c = new
                }
            }
            Op::Call(a, _, args) | Op::TailCallDynamic(a, args) => {
                if a == &old {
                    *a = new
                }
                for a in args {
                    if a == &old {
                        *a = new
                    }
                }
            }
            Op::CallDynamic(a, b, args) => {
                if a == &old {
                    *a = new
                }
                if b == &old {
                    *b = new
                }
                for a in args {
                    if a == &old {
                        *a = new
                    }
                }
            }
        }
    }

    fn verify(&self, assigned_vars: &mut HashSet<VarName>) {
        match self {
            Op::Nop => {}
            Op::Const(v, _) | Op::Label(v, _) | Op::Alloc(v, _) => {
                assigned_vars.insert(*v);
            }
            Op::Add(z, a, b)
            | Op::Sub(z, a, b)
            | Op::Mul(z, a, b)
            | Op::Div(z, a, b)
            | Op::Equal(z, a, b) => {
                assert!(assigned_vars.contains(a));
                assert!(assigned_vars.contains(b));
                assigned_vars.insert(*z);
            }
            Op::Copy(z, a) | Op::GetCell(z, a) | Op::GetRec(z, a, _) => {
                assert!(assigned_vars.contains(a));
                assigned_vars.insert(*z);
            }
            Op::SetRec(a, _, b) => {
                assert!(assigned_vars.contains(a));
                assert!(assigned_vars.contains(b));
            }
            Op::Return(a) => assert!(assigned_vars.contains(a)),
            Op::Branch(block, args) => {
                for a in args {
                    assert!(assigned_vars.contains(a));
                }
                assert_eq!(args.len(), block.n_params());
                block.verify(assigned_vars)
            }
            Op::CondBranch(cond, then_block, else_block) => {
                assert!(assigned_vars.contains(cond));
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
}

impl<V: std::fmt::Debug> PartialEq for Op<V> {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Op::Const(z1, _), Op::Const(z2, _)) => z1 == z2,
            (Op::Add(z1, a1, b1), Op::Add(z2, a2, b2))
            | (Op::Mul(z1, a1, b1), Op::Mul(z2, a2, b2)) => (z1, a1, b1) == (z2, a2, b2),
            (Op::Return(z1), Op::Return(z2)) => z1 == z2,
            (Op::Branch(bl1, args1), Op::Branch(bl2, args2)) => {
                bl1.id() == bl2.id() && args1 == args2
            }
            (Op::CondBranch(co1, a1, c1), Op::CondBranch(co2, a2, c2)) => {
                co1 == co2 && a1.id() == a2.id() && c1.id() == c2.id()
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
struct LivenessGraph {
    edges: HashMap<VarName, HashSet<VarName>>,
    liveset: HashSet<VarName>,

    /// variables that would like to be assigned to the same register, if possible
    preference_pairs: Vec<(VarName, VarName)>,
}

impl LivenessGraph {
    fn new() -> Self {
        LivenessGraph {
            liveset: set![],
            edges: map![],
            preference_pairs: vec![],
        }
    }

    fn join(self, other: Self) -> Self {
        LivenessGraph {
            liveset: self.liveset.union(&other.liveset).cloned().collect(),
            edges: join_edges(self.edges, other.edges),
            preference_pairs: {
                let mut pairs = self.preference_pairs;
                pairs.extend(other.preference_pairs.into_iter());
                pairs
            },
        }
    }

    fn build<V: std::fmt::Debug>(block: &Block<V>) -> Self {
        let block_data = block.block.borrow();

        let mut subgraph = Self::new();

        for op in block_data.ops.iter().rev() {
            subgraph.update(op);
        }

        for p in &block_data.params {
            subgraph.remove_from_liveset(p);
        }

        subgraph
    }

    fn update<V: std::fmt::Debug>(&mut self, op: &Op<V>) {
        match op {
            Op::Nop => {}
            Op::Const(z, _) | Op::Label(z, _) | Op::Alloc(z, _) => {
                self.remove_from_liveset(z);
            }
            Op::Add(z, a, b)
            | Op::Sub(z, a, b)
            | Op::Mul(z, a, b)
            | Op::Div(z, a, b)
            | Op::Equal(z, a, b) => {
                self.remove_from_liveset(z);
                self.liveset.insert(*a);
                self.liveset.insert(*b);
            }
            Op::Copy(z, a) => {
                self.remove_from_liveset(z);
                self.liveset.insert(*a);
                // The register allocator should prefer assignments that make copies redundant
                // TODO: evaluate if making this explicit really has benefits
                //       (the allocator heuristically minimizes the number of registers used,
                //       which might actually have the same effect.)
                self.preference_pairs.push((*a, *z));
            }
            Op::GetCell(z, a) | Op::GetRec(z, a, _) => {
                self.remove_from_liveset(z);
                self.liveset.insert(*a);
            }
            Op::SetRec(a, _, b) => {
                self.liveset.insert(*a);
                self.liveset.insert(*b);
            }
            Op::Return(a) => {
                self.liveset.insert(*a);
            }
            Op::Branch(blk, args) => {
                *self = Self::build(blk);
                self.liveset.extend(args);
                for (a, p) in args.iter().zip(blk.params()) {
                    self.preference_pairs.push((*a, p.name));
                }
            }
            Op::CondBranch(cond, blk1, blk2) => {
                let subgraph1 = Self::build(blk1);
                let subgraph2 = Self::build(blk2);
                *self = subgraph1.join(subgraph2);

                self.liveset.insert(*cond);
                //self.liveset.extend(args1);
                //self.liveset.extend(args2);

                /*for (a, p) in args1
                    .iter()
                    .zip(blk1.params().into_iter().chain(blk2.params()))
                {
                    self.preference_pairs.push((*a, p.name));
                }*/
            }
            Op::Call(z, _, args) => {
                self.remove_from_liveset(z);
                self.liveset.extend(args);
            }
            Op::CallDynamic(z, func, args) => {
                self.remove_from_liveset(z);
                self.liveset.insert(*func);
                self.liveset.extend(args);
            }
            Op::TailCall(_, args) => {
                self.liveset.extend(args);
            }
            Op::TailCallDynamic(func, args) => {
                self.liveset.insert(*func);
                self.liveset.extend(args);
            }
        }

        for a in &self.liveset {
            self.edges
                .entry(*a)
                .or_default()
                .extend(self.liveset.iter().filter(|&b| a != b));
        }
    }

    fn remove_from_liveset(&mut self, var: &VarName) {
        self.liveset.remove(var);
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
    mut assignment: HashMap<K, vm::Register>,
    preferences: &Vec<(K, K)>,
    smallest_color: vm::Register,
) -> HashMap<K, vm::Register> {
    let mut remaining_nodes: HashSet<_> = graph.keys().cloned().collect();

    while let Some(node) = next_node(graph, &mut remaining_nodes, &assignment) {
        let neighbors = &graph[&node];

        let mut neighbor_colors = neighbors.iter().filter_map(|n| assignment.get(n));

        if let Some(color) = assignment.get(&node) {
            if neighbor_colors.any(|nc| nc == color) {
                panic!("Can't find valid coloring")
            }
        } else {
            let neighbor_colors: BTreeSet<_> = neighbor_colors.cloned().collect();
            let color = preferences
                .iter()
                .filter_map(|(a, b)| match (a, b) {
                    (a, b) if a == &node => Some(b),
                    (a, b) if b == &node => Some(a),
                    _ => None,
                })
                .filter_map(|x| assignment.get(x))
                .filter(|&c| !neighbor_colors.contains(c))
                .cloned()
                .next();
            let color =
                color.unwrap_or_else(|| find_smallest_color(neighbor_colors, smallest_color));
            assignment.insert(node, color);
        }
    }

    assignment
}

fn next_node<K: std::hash::Hash + Eq + PartialOrd + Clone>(
    graph: &HashMap<K, HashSet<K>>,
    remaining_nodes: &mut HashSet<K>,
    assignment: &HashMap<K, vm::Register>,
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

fn find_smallest_color(
    neighbor_colors: BTreeSet<vm::Register>,
    start_color: vm::Register,
) -> vm::Register {
    let mut i = start_color;
    for c in neighbor_colors {
        if i < c {
            return i;
        }
        if i == c {
            i += 1;
        }
    }
    i
}

impl TranslationUnit<PrimitiveValue> {
    pub fn compile_function(&mut self, entry_block: &Block<PrimitiveValue>) -> Compiler {
        self.allocate_registers(entry_block);
        let mut compiler = Compiler::new(self.unit.borrow().register_assignment.clone());
        compiler.compile_block(&entry_block);
        compiler
    }
}

pub struct Compiler {
    assembly: Assembler,
    compiled_blocks: HashSet<BlockId>,
    register_assignment: HashMap<VarName, vm::Register>,
    counter: usize,
}

impl Compiler {
    pub fn new(register_assignment: HashMap<VarName, vm::Register>) -> Self {
        Compiler {
            assembly: Assembler::new(),
            compiled_blocks: set![],
            register_assignment,
            counter: 0,
        }
    }

    pub fn get_assembly(&self) -> &Assembler {
        &self.assembly
    }

    pub fn compile_named(&mut self, name: String, block: &Block<PrimitiveValue>) {
        assert!(!self.is_compiled(block));
        self.assembly.label(name);
        self.compile_block(block);
    }

    fn compile_block(&mut self, block: &Block<PrimitiveValue>) -> String {
        let label = self.block_label(block);

        if self.is_compiled(&block) {
            return label;
        }

        self.compiled_blocks.insert(block.id());
        self.assembly.label(label.clone());
        for op in &block.block.borrow().ops {
            self.compile_op(op);
        }
        label
    }

    fn compile_op(&mut self, op: &Op<PrimitiveValue>) {
        use vm::Operand::*;

        macro_rules! r {
            ($var:expr) => {
                *self
                    .register_assignment
                    .get($var)
                    .expect("unassigned register") as vm::Register
            };
        }

        macro_rules! is_assigned {
            ($var:expr) => {
                self.register_assignment.contains_key($var)
            };
        }

        match op {
            Op::Nop => self.assembly.op(vm::Op::Nop),
            Op::Const(z, c) => {
                if is_assigned!(z) {
                    self.assembly.op(vm::Op::Const(r!(z), *c))
                }
            }
            Op::Copy(z, a) => {
                if is_assigned!(z) && r!(z) != r!(a) {
                    self.assembly.op(vm::Op::Copy(r!(z), r!(a)))
                }
            }
            Op::Add(z, a, b) => {
                if is_assigned!(z) {
                    self.assembly.op(vm::Op::Add(r!(z), r!(a), R(r!(b))))
                }
            }
            Op::Sub(z, a, b) => {
                if is_assigned!(z) {
                    self.assembly.op(vm::Op::Sub(r!(z), r!(a), R(r!(b))))
                }
            }
            Op::Mul(z, a, b) => {
                if is_assigned!(z) {
                    self.assembly.op(vm::Op::Mul(r!(z), r!(a), R(r!(b))))
                }
            }
            Op::Div(z, a, b) => {
                if is_assigned!(z) {
                    self.assembly.op(vm::Op::Div(r!(z), R(r!(a)), R(r!(b))))
                }
            }
            Op::Alloc(z, n) => {
                if is_assigned!(z) {
                    self.assembly.op(vm::Op::Alloc(r!(z), *n))
                }
            }
            Op::GetCell(z, a) => {
                if is_assigned!(z) {
                    self.assembly.op(vm::Op::GetCell(r!(z), r!(a)))
                }
            }
            Op::GetRec(z, a, idx) => {
                if is_assigned!(z) {
                    self.assembly.op(vm::Op::GetRec(r!(z), r!(a), I(*idx)))
                }
            }
            Op::SetRec(a, idx, b) => self.assembly.op(vm::Op::SetRec(r!(a), I(*idx), R(r!(b)))),
            Op::Equal(z, a, b) => {
                if is_assigned!(z) {
                    self.assembly.op(vm::Op::Equal(r!(z), r!(a), R(r!(b))))
                }
            }
            Op::Branch(blk, args) => {
                for (a, p) in args.iter().zip(&blk.block.borrow().params) {
                    self.compile_op(&Op::Copy(*p, *a));
                }
                if self.is_compiled(blk) {
                    self.assembly.op(vm::Op::Jmp(I(self.block_label(&blk))))
                } else {
                    self.compile_block(blk);
                }
            }
            Op::CondBranch(cond, yes, no) => {
                let yes_label = self.block_label(yes);
                let after_if = self.unique_label("__after_if");

                self.assembly.op(vm::Op::JmpCond(I(yes_label), r!(cond)));

                if self.is_compiled(no) {
                    self.assembly.op(vm::Op::Jmp(I(self.block_label(no))))
                } else {
                    self.compile_block(no);
                }

                if !self.is_compiled(yes) {
                    self.compile_block(yes);
                }

                self.assembly.label(after_if);
            }
            Op::Return(a) => {
                assert_eq!(r!(a), RETURN_VALUE_REGISTER as vm::Register);
                self.assembly
                    .op(vm::Op::Jmp(R(RETURN_TARGET_REGISTER as vm::Register)));
            }
            Op::CallDynamic(_, f, _) => {
                let after_call = self.unique_label("__after_call");
                self.compile_push(RETURN_TARGET_REGISTER);
                self.assembly.op(vm::Op::LoadLabel(
                    RETURN_TARGET_REGISTER,
                    after_call.clone(),
                ));
                self.assembly.op(vm::Op::Jmp(R(r!(f))));
                self.assembly.label(after_call);
                self.compile_pop(RETURN_TARGET_REGISTER);
            }
            Op::Call(_, f, _) => {
                let func = self.block_label(f);
                let after_call = self.unique_label("__after_call");
                self.compile_push(RETURN_TARGET_REGISTER);
                self.assembly.op(vm::Op::LoadLabel(
                    RETURN_TARGET_REGISTER,
                    after_call.clone(),
                ));
                self.assembly.op(vm::Op::Jmp(I(func)));
                self.assembly.label(after_call);
                self.compile_pop(RETURN_TARGET_REGISTER);
            }
            Op::TailCallDynamic(f, _) => self.assembly.op(vm::Op::Jmp(R(r!(f)))),
            Op::TailCall(f, _) => {
                let func = self.block_label(f);
                self.assembly.op(vm::Op::Jmp(I(func)));
            }
            Op::Label(z, f) => {
                if is_assigned!(z) {
                    let func = self.block_label(f);
                    self.assembly.op(vm::Op::LoadLabel(r!(z), func));
                }
            }
        }
    }

    fn compile_push(&mut self, register: vm::Register) {
        use vm::Operand::R;
        self.assembly.op(vm::Op::SetRec(
            STACK_REGISTER,
            R(STACK_POINTER_REGISTER),
            R(register),
        ));
        self.assembly.op(vm::Op::Inc(STACK_POINTER_REGISTER));
    }

    fn compile_pop(&mut self, register: vm::Register) {
        use vm::Operand::R;
        self.assembly.op(vm::Op::Dec(STACK_POINTER_REGISTER));
        self.assembly.op(vm::Op::GetRec(
            register,
            STACK_REGISTER,
            R(STACK_POINTER_REGISTER),
        ));
    }

    fn block_label(&self, block: &Block<PrimitiveValue>) -> String {
        format!("__blk-{:?}", block.id())
    }

    fn is_compiled(&self, block: &Block<PrimitiveValue>) -> bool {
        self.compiled_blocks.contains(&block.id())
    }

    fn unique_label(&mut self, name: &str) -> String {
        format!("{}-{}", name, global_uid())
    }
}

pub fn link(compilers: &[Compiler]) -> (Vec<vm::Op>, HashMap<String, usize>) {
    let mut assembly = Assembler::new();

    for comp in compilers {
        assembly.append(&comp.assembly)
    }

    let (code, labels) = assembly.assemble().unwrap();

    (code, labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::store_code_block;
    use crate::primitive_value::CodePos;
    use crate::virtual_machine::eval;

    impl<V: std::fmt::Debug> PartialEq<Vec<Op<V>>> for Block<V> {
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
        entry.branch::<Var<()>>(&exit, &[]);
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
        entry.branch::<Var<()>>(&exit, &[]);
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
        entry.branch_conditionally(&c, &yes, &no);
        yes.terminate(&yes.constant(()));
        no.terminate(&no.constant(()));

        tu.verify(&entry);

        assert_eq!(
            yes,
            vec![Op::Const(VarName(1), ()), Op::Return(VarName(1)),]
        );

        assert_eq!(no, vec![Op::Const(VarName(2), ()), Op::Return(VarName(2)),]);

        assert_eq!(entry, vec![Op::CondBranch(VarName(0), yes, no),]);
    }

    #[test]
    fn conditional_branch_with_args() {
        let tu = TranslationUnit::<i64>::new();
        let entry = tu.new_block();
        let yes = tu.new_block();
        let exit = tu.new_block();
        let c = entry.append_parameter();
        let _ = entry.append_parameter();
        entry.branch_conditionally(&c, &yes, &exit);
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

        assert_eq!(entry, vec![Op::CondBranch(VarName(0), yes, exit,),]);
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
        entry.branch_conditionally(&c, &yes, &no);
        let _ = yes.constant(());
        yes.branch::<Var<()>>(&exit, &[]);
        let x = no.constant(());
        no.branch::<Var<()>>(&exit, &[]);
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
        entry.branch_conditionally(&c, &yes, &no);
        yes.branch::<Var<()>>(&exit, &[]);
        no.branch::<Var<()>>(&exit, &[]);
        exit.terminate(&x);

        tu.verify(&entry);

        assert_eq!(exit, vec![Op::Return(VarName(1)),]);

        assert_eq!(yes, vec![Op::Branch(exit.clone(), vec![]),]);

        assert_eq!(no, vec![Op::Branch(exit.clone(), vec![]),]);

        assert_eq!(entry, vec![Op::CondBranch(VarName(0), yes, no),]);
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
            LivenessGraph::build(&entry).edges,
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
            LivenessGraph::build(&entry).edges,
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
    fn assign_registers_fully_connected() {
        let graph = map![
            'A' => set!['B', 'C'],
            'B' => set!['A', 'C'],
            'C' => set!['A', 'B']
        ];
        assert_eq!(
            greedy_coloring(&graph, map![], &vec![], 0),
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
            greedy_coloring(&graph, map![], &vec![], 0),
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
            greedy_coloring(&graph, map![], &vec![], 0),
            map!['A' => 2, 'B' => 1, 'C' => 0, 'D' => 1, 'E' => 0]
        );
    }

    #[test]
    fn assign_call_registers() {
        let tu = TranslationUnit::<i64>::new();
        let entry = tu.new_block();
        let f = entry.append_parameter();
        let x = entry.append_parameter();
        let y = entry.append_parameter();
        let z = entry.append_parameter();

        let r = entry.call(&f, &[&z, &x, &y]);
        entry.terminate(&r);

        tu.verify(&entry);

        tu.preassign_function_arg_registers(&entry);

        tu.allocate_registers(&entry);

        assert_eq!(
            tu.unit.borrow().register_assignment,
            map![
            VarName(0) => FIRST_GENERAL_PURPOSE_REGISTER + 0,
            VarName(1) => FIRST_GENERAL_PURPOSE_REGISTER + 2,
            VarName(2) => FIRST_GENERAL_PURPOSE_REGISTER + 3,
            VarName(3) => FIRST_GENERAL_PURPOSE_REGISTER + 1,
            VarName(4) => RETURN_VALUE_REGISTER,
            VarName(5) => FIRST_ARG_REGISTER + 0,
            VarName(6) => FIRST_ARG_REGISTER + 1,
            VarName(7) => FIRST_ARG_REGISTER + 2,
            VarName(8) => RETURN_VALUE_REGISTER]
        );
    }

    #[test]
    fn assign_registers_conflicts() {
        let tu = TranslationUnit::<i64>::new();
        let entry = tu.new_block();
        let f = entry.append_parameter();
        let x = entry.append_parameter();
        let _ = entry.call(&f, &[&x]); // conflict 1: x goes into 1st argument slot
        entry.return_(&x); // conflict 2: x goes into return value slot

        tu.verify(&entry);

        tu.preassign_function_arg_registers(&entry);

        tu.allocate_registers(&entry);

        assert_eq!(
            tu.unit.borrow().register_assignment,
            map![
            f.name => FIRST_GENERAL_PURPOSE_REGISTER + 0,
            x.name => FIRST_GENERAL_PURPOSE_REGISTER + 2,
            VarName(2) => FIRST_GENERAL_PURPOSE_REGISTER + 0,
            VarName(3) => FIRST_ARG_REGISTER + 0,
            // VarName(4) is unused
            VarName(5) => RETURN_VALUE_REGISTER]
        );
    }

    /*#[test]
    fn assign_call_register_conflict() {
        let tu = TranslationUnit::<i64>::new();
        let entry = tu.new_block();
        let v0 = entry.append_parameter();
        let v1 = entry.append_parameter();
        let v2 = entry.append_parameter();
        let v3 = entry.append_parameter();

        let _4 = entry.call(&v0, &[&v1, &v2, &v3]);

        entry.tail_call(&v0, &[&v3, &v2, &v1]);

        tu.verify(&entry);

        tu.allocate_registers(&entry);

        tu.verify(&entry);

        assert_eq!(
            tu.unit.borrow().register_assignment,
            map![
                VarName(0) => FIRST_GENERAL_PURPOSE_REGISTER + 4,
                VarName(1) => FIRST_ARG_REGISTER + 0,
                VarName(2) => FIRST_ARG_REGISTER + 1,
                VarName(3) => FIRST_ARG_REGISTER + 2,
                VarName(4) => RETURN_VALUE_REGISTER,
                VarName(5) => FIRST_GENERAL_PURPOSE_REGISTER + 0,
                VarName(6) => FIRST_ARG_REGISTER + 0,
                VarName(7) => FIRST_GENERAL_PURPOSE_REGISTER + 3,
                VarName(8) => FIRST_ARG_REGISTER + 2,
                VarName(9) => FIRST_ARG_REGISTER + 2,
                VarName(10) => FIRST_ARG_REGISTER + 2,
            ]
        );
    }*/

    #[test]
    fn compile_branch() {
        use vm::Operand::R;

        let tu = TranslationUnit::<PrimitiveValue>::new();
        let entry = tu.new_block();
        let exit = tu.new_block();
        let x = entry.append_parameter();
        let z = entry.append_parameter();
        entry.branch(&exit, &[&x]);
        let y = exit.append_parameter();
        exit.terminate(&exit.add(&y, &z));

        tu.verify(&entry);
        tu.allocate_registers(&entry);

        let mut compiler = Compiler::new(tu.unit.borrow().register_assignment.clone());
        compiler.compile_block(&entry);

        let (code, _) = compiler.assembly.assemble().unwrap();

        assert_eq!(
            code,
            vec![
                vm::Op::Add(
                    RETURN_VALUE_REGISTER,
                    FIRST_GENERAL_PURPOSE_REGISTER + 1,
                    R(FIRST_GENERAL_PURPOSE_REGISTER)
                ),
                vm::Op::Jmp(R(RETURN_TARGET_REGISTER))
            ]
        );
    }

    #[test]
    fn compile_branch_to_existing_block() {
        use vm::Operand::*;

        let tu = TranslationUnit::<PrimitiveValue>::new();
        let entry = tu.new_block();
        let exit = tu.new_block();
        let x = entry.append_parameter();
        let z = entry.append_parameter();
        entry.branch(&exit, &[&x]);
        let y = exit.append_parameter();
        exit.terminate(&exit.add(&y, &z));

        tu.verify(&entry);
        tu.allocate_registers(&entry);

        let mut compiler = Compiler::new(tu.unit.borrow().register_assignment.clone());
        compiler.compile_block(&exit);
        compiler.compile_block(&entry);

        let (code, _) = compiler.assembly.assemble().unwrap();
        assert_eq!(
            code,
            vec![
                vm::Op::Add(
                    RETURN_VALUE_REGISTER,
                    FIRST_GENERAL_PURPOSE_REGISTER + 1,
                    R(FIRST_GENERAL_PURPOSE_REGISTER)
                ),
                vm::Op::Jmp(R(RETURN_TARGET_REGISTER)),
                vm::Op::Jmp(I(-2))
            ]
        );
    }

    #[test]
    fn compile_function() {
        let tu = TranslationUnit::<PrimitiveValue>::new();
        let entry = tu.new_block();
        let a = entry.append_parameter();
        let b = entry.append_parameter();

        let c = entry.add(&a, &b);
        entry.return_(&c);

        let func = tu.new_function(&entry);

        tu.allocate_registers(&func);

        let mut compiler = Compiler::new(tu.unit.borrow().register_assignment.clone());
        compiler.compile_block(&func);

        let (code, _) = compiler.assembly.assemble().unwrap();

        let code = store_code_block(code);
        let main = store_code_block(vec![
            vm::Op::Const(
                FIRST_ARG_REGISTER as vm::Register,
                PrimitiveValue::Integer(24),
            ),
            vm::Op::Const(
                1 + FIRST_ARG_REGISTER as vm::Register,
                PrimitiveValue::Integer(18),
            ),
            vm::Op::LoadLabel(RETURN_TARGET_REGISTER as vm::Register, 2),
            vm::Op::JmpFar(CodePos::new(code, 0)),
            vm::Op::Copy(0, RETURN_VALUE_REGISTER as vm::Register),
            vm::Op::Term,
        ]);

        assert_eq!(eval(&main), PrimitiveValue::Integer(42));
    }

    #[test]
    fn compile_function_dynamic_call() {
        use vm::Operand::R;
        let tu = TranslationUnit::<PrimitiveValue>::new();
        let entry = tu.new_block();
        let a = entry.append_parameter();
        let b = entry.append_parameter();

        let c = entry.call(&a, &[&b]);
        entry.return_(&c);

        let func = tu.new_function(&entry);

        tu.allocate_registers(&func);

        let mut compiler = Compiler::new(tu.unit.borrow().register_assignment.clone());
        compiler.compile_block(&func);

        let (code, _) = compiler.assembly.assemble().unwrap();

        assert_eq!(
            code,
            vec![
                vm::Op::Copy(FIRST_GENERAL_PURPOSE_REGISTER, FIRST_ARG_REGISTER),
                vm::Op::Copy(FIRST_ARG_REGISTER, FIRST_ARG_REGISTER + 1),
                vm::Op::SetRec(
                    STACK_REGISTER,
                    R(STACK_POINTER_REGISTER),
                    R(RETURN_TARGET_REGISTER)
                ),
                vm::Op::Inc(STACK_POINTER_REGISTER),
                vm::Op::LoadLabel(RETURN_TARGET_REGISTER, 2),
                vm::Op::Jmp(R(FIRST_GENERAL_PURPOSE_REGISTER)),
                vm::Op::Dec(STACK_POINTER_REGISTER),
                vm::Op::GetRec(
                    RETURN_TARGET_REGISTER,
                    STACK_REGISTER,
                    R(STACK_POINTER_REGISTER)
                ),
                vm::Op::Jmp(R(RETURN_TARGET_REGISTER)),
            ]
        );
    }

    #[test]
    fn compile_function_static_call() {
        use vm::Operand::*;
        let tu1 = TranslationUnit::<PrimitiveValue>::new();
        let entry1 = tu1.new_block();
        let a = entry1.append_parameter();
        entry1.return_(&a);
        let func1 = tu1.new_function(&entry1);

        let tu2 = TranslationUnit::<PrimitiveValue>::new();
        let entry2 = tu2.new_block();
        let b = entry2.constant(PrimitiveValue::Integer(42));
        entry2.tail_call_static(&func1, &[&b]);
        let func2 = tu2.new_function(&entry2);

        tu2.allocate_registers(&func2);
        let mut compiler2 = Compiler::new(tu2.unit.borrow().register_assignment.clone());
        compiler2.compile_named("f2".to_string(), &func2);

        tu1.allocate_registers(&func1);
        let mut compiler1 = Compiler::new(tu1.unit.borrow().register_assignment.clone());
        compiler1.compile_named("f1".to_string(), &func1);

        let (code, labels) = link(&[compiler1, compiler2]);

        assert_eq!(
            code,
            vec![
                vm::Op::Copy(FIRST_GENERAL_PURPOSE_REGISTER, FIRST_ARG_REGISTER),
                vm::Op::Jmp(R(0)),
                vm::Op::Const(FIRST_ARG_REGISTER, 42.into()),
                vm::Op::Jmp(I(-3))
            ]
        );

        assert_eq!(labels["f1"], 0);
        assert_eq!(labels["f2"], 2);
    }

    /*#[test]
    fn compile_mutual_recursion() {
        use vm::Operand::R;
        let tu = TranslationUnit::<PrimitiveValue>::new();
        let even_entry = tu.new_block();
        let is_even = tu.new_block();
        let not_even = tu.new_block();
        let odd_entry = tu.new_block();
        let is_odd = tu.new_block();
        let not_odd = tu.new_block();
        let even = Function::new(even_entry.clone());
        let odd = Function::new(odd_entry.clone());

        let x = even_entry.append_parameter();
        let e = even_entry.equals(&x, &even_entry.constant(PrimitiveValue::Integer(0)));
        even_entry.branch_conditionally(&e, &is_even, &[], &not_even, &[]);
        is_even.return_(&is_even.constant(PrimitiveValue::True));
        not_even.tail_call_static(&odd, &[&not_even.add(&x, &not_even.constant(PrimitiveValue::Integer(-1)))]);
        //not_even.terminate(&not_even.add(&x, &not_even.constant(PrimitiveValue::Integer(-1))));

        let y = odd_entry.append_parameter();
        let o = odd_entry.equals(&y, &odd_entry.constant(PrimitiveValue::Integer(0)));
        odd_entry.branch_conditionally(&o, &is_odd, &[], &not_odd, &[]);
        is_odd.return_(&is_odd.constant(PrimitiveValue::False));
        not_odd.tail_call_static(&odd, &[&not_odd.add(&y, &not_odd.constant(PrimitiveValue::Integer(-1)))]);
        //not_odd.terminate(&y);

        tu.verify(&even_entry);
        tu.verify(&odd_entry);

        let mut compiler = Compiler::new();
        even.compile(&mut compiler);
        //odd.compile(&mut compiler);
    }*/
}
