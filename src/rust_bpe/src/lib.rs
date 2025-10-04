
use std::collections::HashMap;


/// A rust implementation of a Byte Pair Encoding (BPE) tokenizer
pub struct BpeTokenizer {

    // Vocab mapping: token string to token ID
    pub token2id: HashMap<String, usize>,

    // Reverse vocab mapping: token ID to token string
    pub id2token: HashMap<usize, String>,

    // Learned merges as tuples of strings
    pub merges: Vec<(String, String)>,

    // Number of merges to perform during training
    pub num_merges: usize,
    
}

impl BpeTokenizer {

    // Constructs a new BpeTokenizer instance
    pub fn new(num_merges: usize) -> Self {
        BpeTokenizer {
            token2id: HashMap::new(),
            id2token: HashMap::new(),
            merges: Vec::new(),
            num_merges,
        }
    }

    pub fn fit() { }

    pub fn get_stats() { }
    
    fn merge_vocab() { }

    pub fn tokenize() { }

    pub fn encode() { }

    pub fn decode() { }




}