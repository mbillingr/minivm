pub mod bytecode_builder;
pub mod core_compiler;
pub mod core_cps_compiler;
pub mod memory;
pub mod primitive_value;
pub mod scheme_parser;
pub mod ssa_builder;
pub mod test_utils;
pub mod virtual_machine;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
