
use pyo3::prelude::*;
use rust_bpe::BpeTokenizer as RustBpeTokenizer;

/// Python exposed wrapper for the BPE Tokenizer
/// Uses the #[pyclass] macro to make this struct accessible in python
#[pyclass]
pub struct BpeTokenizer {
    inner: RustBpeTokenizer,
}