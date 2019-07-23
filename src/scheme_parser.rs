use pest::error::{InputLocation, LineColLocation};
use pest::{iterators::Pair, Parser, RuleType, Span};
use pest_derive::*;
use std::cell::RefCell;
use std::marker::PhantomData;

pub type Result<T, R> = std::result::Result<T, Error<R>>;

#[derive(Debug)]
pub struct Error<R> {
    pub kind: ErrorKind<R>,
    pub location: InputLocation,
    pub line_col: LineColLocation,
}

#[derive(Debug)]
pub enum ErrorKind<R> {
    PestError(pest::error::Error<R>),
    InvalidNumericConstant,
}

impl<R> From<pest::error::Error<R>> for Error<R> {
    fn from(pe: pest::error::Error<R>) -> Self {
        Error {
            location: pe.location.clone(),
            line_col: pe.line_col.clone(),
            kind: ErrorKind::PestError(pe),
        }
    }
}

impl<R> From<(ErrorKind<R>, Pair<'_, R>)> for Error<R> {
    fn from((kind, pair): (ErrorKind<R>, Pair<R>)) -> Self {
        unimplemented!()
    }
}

#[derive(Debug)]
pub struct Object<'i> {
    kind: ObjectKind<'i>,
    span: Span<'i>,
}

#[derive(Debug)]
pub enum ObjectKind<'i> {
    Nil,
    Exact(i64),
    Inexact(f64),
    Symbol(&'i str),
    String(&'i str),
    Pair(Box<RefCell<(Object<'i>, Object<'i>)>>),
}

impl<'i, T: Into<ObjectKind<'i>>, R: RuleType> From<(T, Pair<'i, R>)> for Object<'i> {
    fn from((x, pair): (T, Pair<'i, R>)) -> Self {
        Object {
            kind: x.into(),
            span: pair.as_span(),
        }
    }
}

impl<'i, T: Into<ObjectKind<'i>>> From<(T, Span<'i>)> for Object<'i> {
    fn from((x, span): (T, Span<'i>)) -> Self {
        Object {
            kind: x.into(),
            span,
        }
    }
}

impl<'i> From<i64> for ObjectKind<'i> {
    fn from(x: i64) -> Self {
        ObjectKind::Exact(x)
    }
}

impl<'i> From<f64> for ObjectKind<'i> {
    fn from(x: f64) -> Self {
        ObjectKind::Inexact(x)
    }
}

impl<'i> Object<'i> {
    fn nil() -> Self {
        (ObjectKind::Nil, Span::new("", 0, 0).unwrap()).into()
    }

    fn cons(a: Object<'i>, b: Object<'i>, span: Span<'i>) -> Self {
        Object {
            kind: ObjectKind::Pair(Box::new(RefCell::new((a, b)))),
            span,
        }
    }

    fn set_cdr(&self, item: Object<'i>) {
        match self.kind {
            ObjectKind::Pair(ref pair) => pair.borrow_mut().1 = item,
            _ => panic!("Not a pair"),
        }
    }

    fn set_list_cdr(self, item: Object<'i>) -> Object<'i> {
        match self.kind {
            ObjectKind::Nil => item,
            ObjectKind::Pair(pair) => {
                let (a, b) = pair.into_inner();
                Object::cons(a, b.set_list_cdr(item), self.span)
            }
            _ => panic!("Not a list"),
        }
    }
}

struct ListBuilder<'i> {
    start: Object<'i>,
}

impl<'i> ListBuilder<'i> {
    fn new() -> Self {
        ListBuilder {
            start: Object::nil(),
        }
    }

    fn set_cdr(&mut self, item: Object<'i>) {
        self.start = std::mem::replace(&mut self.start, Object::nil()).set_list_cdr(item);
    }

    fn append(&mut self, item: Object<'i>) {
        let span = item.span.clone();
        self.set_cdr(Object::cons(item, Object::nil(), span));
    }

    fn build(self) -> Object<'i> {
        self.start
    }
}

#[derive(Parser)]
#[grammar = "r7rs.pest"]
pub struct R7rsGrammar;

pub fn parse_datum(input: &str) -> Result<Object, Rule> {
    let mut datum = R7rsGrammar::parse(Rule::datum, input)?;
    walk_datum(datum.next().unwrap())
}

fn walk_datum(pair: Pair<Rule>) -> Result<Object, Rule> {
    match pair.as_rule() {
        Rule::list => walk_list(pair),
        Rule::number => walk_number(pair),
        Rule::symbol => walk_symbol(pair),
        Rule::string_content => walk_string(pair),
        Rule::abbreviation => walk_abbreviation(pair),
        _ => unimplemented!("{:?}", pair),
    }
}

fn walk_list(pair: Pair<Rule>) -> Result<Object, Rule> {
    let mut parse_list = pair.into_inner();
    let mut list_builder = ListBuilder::new();
    while let Some(list_item) = parse_list.next() {
        if list_item.as_rule() == Rule::dot {
            let item = walk_datum(parse_list.next().unwrap())?;
            list_builder.set_cdr(item);
        } else {
            let item = walk_datum(list_item)?;
            list_builder.append(item);
        }
    }
    Ok(list_builder.build())
}

fn walk_number(pair: Pair<Rule>) -> Result<Object, Rule> {
    let number = pair.into_inner().next().unwrap();
    match number.as_rule() {
        Rule::num_2 => walk_num_with_radix(number, 2),
        Rule::num_8 => walk_num_with_radix(number, 8),
        Rule::num_10 => walk_num_with_radix(number, 10),
        Rule::num_16 => walk_num_with_radix(number, 16),
        _ => unreachable!(),
    }
}

fn walk_num_with_radix(pair: Pair<Rule>, radix: u32) -> Result<Object, Rule> {
    let mut inner = pair.clone().into_inner();
    let exactness = inner.next().unwrap();
    let value = inner.next().unwrap();
    let integer_result = i64::from_str_radix(value.as_str(), radix);
    match (exactness.as_rule(), integer_result) {
        (Rule::exact, Ok(i)) | (Rule::empty, Ok(i)) => Ok((i, pair).into()),
        (Rule::exact, Err(_)) => Err((ErrorKind::InvalidNumericConstant, pair).into()),
        (Rule::inexact, Ok(i)) => Ok((i as f64, pair).into()),
        (Rule::inexact, Err(_)) | (Rule::empty, Err(_)) => value
            .as_str()
            .parse::<f64>()
            .map(|f| (f, pair.clone()).into())
            .map_err(|_| (ErrorKind::InvalidNumericConstant, pair).into()),
        _ => unreachable!(),
    }
}

fn walk_symbol(pair: Pair<Rule>) -> Result<Object, Rule> {
    let identifier = pair.clone().into_inner().next().unwrap();
    match identifier.as_rule() {
        Rule::delimited_identifier | Rule::normal_identifier | Rule::peculiar_identifier => {
            Ok((ObjectKind::Symbol(identifier.as_str()), pair).into())
        }
        _ => unreachable!(),
    }
}

fn walk_string(pair: Pair<Rule>) -> Result<Object, Rule> {
    Ok((ObjectKind::String(pair.as_str()), pair).into())
}

fn walk_abbreviation(pair: Pair<Rule>) -> Result<Object, Rule> {
    let mut inner = pair.clone().into_inner();
    let prefix = inner.next().unwrap();
    let datum = inner.next().unwrap();

    match prefix.as_str() {
        "'" => Ok(Object::cons(
            (ObjectKind::Symbol("quote"), prefix).into(),
            Object::cons(walk_datum(datum.clone())?, Object::nil(), datum.as_span()),
            pair.as_span(),
        )),
        _ => unimplemented!("{:?}", prefix),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        println!("{:?}", R7rsGrammar::parse(Rule::prefix_10, "#i"));
        println!(
            "{:?}",
            parse_datum("(#e1 |x y| #i2 \"foo\" bar #b10 4.7 . 5)").unwrap()
        );
        println!(
            "{:?}",
            parse_datum("(define (two-sqr x) (* 2 x x))").unwrap()
        );
        println!(
            "{:?}",
            parse_datum("(define (two-sqr x) (* 2 x x))").unwrap()
        );
        println!("{:?}", parse_datum("'(1 2 3)").unwrap());
        panic!()
    }
}
