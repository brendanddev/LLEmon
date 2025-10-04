
use pyo3::prelude::*;
use ::rust_bpe_tokenizer::BpeTokenizer as RustBpeTokenizer;

/// Python exposed wrapper for the BPE Tokenizer
/// Uses the #[pyclass] macro to make this struct accessible in python
#[pyclass]
pub struct BpeTokenizer {
    inner: RustBpeTokenizer,
}

/// Implementation of methods that will be exposed to Python
/// Uses the #[pymethods] macro to mark these methods as callable from Python
#[pymethods]
impl BpeTokenizer {

    /// Constructor for the BpeTokenizer
    #[new]
    pub fn new(num_merges: usize) -> Self {
        BpeTokenizer {
            inner: RustBpeTokenizer::new(num_merges),
        }
    }

    /// Train the tokenizer on the provided text
    pub fn fit(&mut self, text: &str) {
        self.inner.fit(text);
    }

    /// Encode input text to a list of token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encode(text)
    }

    /// Decode a list of token IDs back to text
    pub fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }

    /// Get the size of the vocabulary
    pub fn get_vocab_size(&self) -> usize {
        self.inner.token2id.len()
    }
}