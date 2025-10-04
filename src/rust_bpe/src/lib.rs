
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

    /// Train the BPE tokenizer on the input text
    pub fn fit() {
        // Build initial vocab from words with end of word token
        let mut vocab: HashMap<String, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let chars_with_end: Vec<String> = word.chars().map(|c| c.to_string()).chain(std::iter::once("</w>".to_string())).collect();
            let key = chars_with_end.join(" ");
            *vocab.entry(key).or_insert(0) += 1;
        }

        // Perform BPE merges
        for _ in 0..self.num_merges {
            let pairs = Self::get_stats(&vocab);
            if pairs.is_empty() {
                break;
            }
            let best = pairs.iter().max_by_key(|(_, count)| *count).unwrap().0.clone();
            vocab = Self::merge_vocab(best.clone(), &vocab);
            self.merges.push(best);
        }

        // Extract final token set
        let mut vocab_set: HashSet<String> = HashSet::new();
        for word in vocab.keys() {
            for token in word.split_whitespace() {
                vocab_set.insert(token.to_string());
            }
        }
    }


    pub fn get_stats() { }
    
    fn merge_vocab() { }

    pub fn tokenize() { }

    pub fn encode() { }

    pub fn decode() { }




}