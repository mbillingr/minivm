use crate::primitive_value::PrimitiveValue;
use crate::virtual_machine::Op;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem::replace;

//const INITIAL_RECORD_STORAGE_SIZE: usize = 16;

thread_local! {
    static CODE: RefCell<Vec<Vec<Op>>> = RefCell::new(vec![]);

    //static RECORDS: RecordStorage<'static> = RecordStorage::new(INITIAL_RECORD_STORAGE_SIZE);
}

pub fn store_code_block(code: Vec<Op>) -> &'static [Op] {
    CODE.with(|c| {
        let n = code.len();
        c.borrow_mut().push(code);
        let ptr = c.borrow().last().unwrap().as_ptr();
        unsafe { std::slice::from_raw_parts(ptr, n) }
    })
}

/*pub fn allocate_record(n_fields: usize) -> Record {
    RECORDS.with(|recs|{
        recs.allocate_record(n_fields)
    })
}

pub fn init_record(items: &[PrimitiveValue]) -> Record {
    RECORDS.with(|recs|{
        recs.init_record(items)
    })
}*/

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Record {
    pub start_idx: usize,
    pub len: usize,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Pair {
    pub start_idx: usize,
}

impl From<Record> for Pair {
    fn from(r: Record) -> Self {
        if r.len != 2 {
            panic!("Length {} record is not a pair", r.len)
        }
        Pair {
            start_idx: r.start_idx,
        }
    }
}

impl From<Pair> for Record {
    fn from(p: Pair) -> Self {
        Record {
            start_idx: p.start_idx,
            len: 2,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Cell {
    pub start_idx: usize,
}

impl From<Record> for Cell {
    fn from(r: Record) -> Self {
        if r.len != 1 {
            panic!("Length {} record is not a cell", r.len)
        }
        Cell {
            start_idx: r.start_idx,
        }
    }
}

impl From<Cell> for Record {
    fn from(p: Cell) -> Self {
        Record {
            start_idx: p.start_idx,
            len: 1,
        }
    }
}

pub struct RecordStorage<'a> {
    entries: Vec<PrimitiveValue>,
    _p: PhantomData<&'a ()>,
}

impl<'a> RecordStorage<'a> {
    pub fn new(capacity: usize) -> Self {
        RecordStorage {
            entries: Vec::with_capacity(capacity),
            _p: PhantomData,
        }
    }

    pub fn free_slots(&self) -> usize {
        self.entries.capacity() - self.entries.len()
    }

    pub fn total_size(&self) -> usize {
        self.entries.capacity()
    }

    unsafe fn as_mut(&self) -> &mut Self {
        &mut *(self as *const _ as *mut Self)
    }

    pub fn get_record(&self, r: Record) -> &[PrimitiveValue] {
        &self.entries[r.start_idx..r.start_idx + r.len]
    }

    pub fn get_pair(&self, r: Pair) -> [&PrimitiveValue; 2] {
        [&self.entries[r.start_idx], &self.entries[r.start_idx + 1]]
    }

    pub fn set_element(&self, r: Record, idx: usize, value: PrimitiveValue) {
        unsafe { *(&self.entries[r.start_idx + idx] as *const _ as *mut _) = value }
    }

    pub fn allocate_record(&'a self, n_fields: usize, roots: &mut [PrimitiveValue]) -> Record {
        if self.free_slots() < n_fields {
            let new_roots = self.collect_garbage(&roots, 0);
            for (r, n) in roots.iter_mut().zip(new_roots) {
                *r = *n;
            }
        }
        if self.free_slots() < n_fields {
            let new_roots = self.collect_garbage(&roots, n_fields.max(self.total_size()));
            for (r, n) in roots.iter_mut().zip(new_roots) {
                *r = *n;
            }
        }
        self.allocate(n_fields)
    }

    fn allocate(&'a self, n_fields: usize) -> Record {
        let start_idx = self.entries.len();
        if start_idx + n_fields > self.entries.capacity() {
            panic!("Out of Record storage capacity")
        }
        unsafe {
            // We shamelessly mutate the record storage although there are certainly live
            // references to its entries in existence. Thus, it is important that
            //     - `RecordStorage::allocate_record` does not invalidate existing references (no
            //       reallocation of the backing storage!)
            //     - records do not alias
            // We also pretend that the returned record has static lifetime. This is certainly not
            // true. The garbage collector (tbd.) must make sure that all references are updated
            // whenever stuff is moved around.
            self.as_mut()
                .entries
                .resize(start_idx + n_fields, PrimitiveValue::Undefined);
            //std::slice::from_raw_parts(&self.entries[start], n_fields)
        }
        Record {
            start_idx,
            len: n_fields,
        }
    }

    pub fn collect_garbage(&self, roots: &[PrimitiveValue], extend: usize) -> &[PrimitiveValue] {
        let mut to_space = Vec::with_capacity(self.entries.capacity() + extend);
        let mut scan_idx = 0;

        for root in roots {
            to_space.push(*root);
        }

        while scan_idx < to_space.len() {
            if let Some(r) = to_space[scan_idx].try_as_record() {
                let rec = self.get_record(r);
                match rec[0] {
                    PrimitiveValue::Relocated(l) => {
                        to_space[scan_idx] = PrimitiveValue::Record(Record { start_idx: l, ..r })
                    }
                    _ => {
                        let new_loc = to_space.len();
                        to_space[scan_idx] = PrimitiveValue::Record(Record {
                            start_idx: new_loc,
                            ..r
                        });
                        let first = rec[0];
                        unsafe {
                            *(&rec[0] as *const _ as *mut _) = PrimitiveValue::Relocated(new_loc);
                        }
                        to_space.push(first);
                        to_space.extend_from_slice(&rec[1..]);
                    }
                }
            }
            scan_idx += 1;
        }

        unsafe {
            replace(&mut self.as_mut().entries, to_space);
        }

        &self.entries[0..roots.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_init_record(rs: &RecordStorage, items: &[PrimitiveValue]) -> Record {
        let record = rs.allocate(items.len());
        for (r, i) in rs.get_record(record).iter().zip(items) {
            unsafe {
                *(r as *const _ as *mut _) = *i;
            }
        }
        record
    }

    #[test]
    fn code_blocks_are_statically_preserved() {
        let code = {
            let owned = vec![Op::Nop, Op::Term];
            let static_ref = store_code_block(owned);
            static_ref
        };
        assert_eq!(code, &[Op::Nop, Op::Term]);
    }

    #[test]
    fn stored_blocks_do_not_invalidate_each_other() {
        let codes: Vec<_> = (0..1000)
            .rev()
            .map(|n| vec![Op::Nop; n])
            .map(store_code_block)
            .collect();
        for (n, code) in codes.into_iter().rev().enumerate() {
            assert_eq!(code, vec![Op::Nop; n].as_slice());
        }
    }

    #[test]
    fn simple_record_allocation_works() {
        let rs = RecordStorage::new(5);
        let record = rs.allocate(5);
        assert_eq!(
            rs.get_record(record),
            vec![PrimitiveValue::Undefined; 5].as_slice()
        );
    }

    #[test]
    fn records_do_not_alias() {
        const N: usize = 100;
        let storage = RecordStorage::new(N);
        let pairs: Vec<_> = (0..N / 2)
            .map(|i| {
                raw_init_record(
                    &storage,
                    &[(2 * i as i64).into(), (1 + 2 * i as i64).into()],
                )
            })
            .collect();
        for (i, p) in pairs.into_iter().enumerate() {
            let i = i as i64;
            assert_eq!(storage.get_record(p), [(2 * i).into(), (2 * i + 1).into()]);
        }
    }

    #[test]
    fn garbage_collection_frees_unreachable_slots() {
        let rs = RecordStorage::new(50);
        raw_init_record(&rs, &[1.into(), 2.into(), 3.into()]);
        raw_init_record(&rs, &[1.into(), 2.into(), 3.into()]);
        raw_init_record(&rs, &[1.into(), 2.into(), 3.into()]);
        assert_eq!(rs.free_slots(), 41);

        rs.collect_garbage(&[], 50);
        assert_eq!(rs.free_slots(), 100);
    }

    #[test]
    fn garbage_collection_preserves_roots() {
        let rs = RecordStorage::new(5);
        let record = raw_init_record(&rs, &[1.into(), 2.into(), 3.into()]);

        let roots = [PrimitiveValue::Record(record)];
        let updated_roots = rs.collect_garbage(&roots, 0);

        let record = updated_roots[0].as_record();

        assert_eq!(
            rs.get_record(record),
            vec![1.into(), 2.into(), 3.into()].as_slice()
        );
    }

    #[test]
    fn garbage_collection_preserves_nested_records() {
        let rs = RecordStorage::new(20);
        let a = raw_init_record(&rs, &[1.into(), 2.into(), 3.into()]);
        let b = raw_init_record(&rs, &[4.into(), 5.into(), 6.into()]);
        let _ = raw_init_record(&rs, &[7.into(), 8.into(), 9.into()]);
        let record = raw_init_record(
            &rs,
            &[
                PrimitiveValue::Record(a),
                7.into(),
                PrimitiveValue::Record(b),
            ],
        );

        let roots = [PrimitiveValue::Record(record)];
        let updated_roots = rs.collect_garbage(&roots, 0);

        let record = updated_roots[0].as_record();
        let data = rs.get_record(record);

        assert_eq!(
            rs.get_record(data[0].as_record()),
            vec![1.into(), 2.into(), 3.into()].as_slice()
        );
        assert_eq!(data[1], 7.into());
        assert_eq!(
            rs.get_record(data[2].as_record()),
            vec![4.into(), 5.into(), 6.into()].as_slice()
        );

        assert_eq!(rs.free_slots(), 10);
    }
}
