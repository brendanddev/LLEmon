
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

    /// Get the number of merges used
    pub fn get_num_merges(&self) -> usize {
        self.inner.num_merges
    }
}

/// Save the learned vocabulary to a JSON file
#[pyfunction]
pub fn save_vocab(tokenizer: &BpeTokenizer, path: &str) {
    ::rust_bpe_tokenizer::save_vocab(&tokenizer.inner, path);
}

/// Save tokenized text (list of token IDs) to a JSON file
#[pyfunction]
pub fn save_tokenized_text(tokens: Vec<usize>, path: &str) {
    ::rust_bpe_tokenizer::save_tokenized_text(&tokens, path);
}

/// Module initialization
#[pymodule]
fn rust_bpe_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BpeTokenizer>()?;
    m.add_function(wrap_pyfunction!(save_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(save_tokenized_text, m)?)?;
    Ok(())
}