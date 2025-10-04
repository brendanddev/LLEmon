
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

        // Assign token IDs
        self.token2id = vocab_set.iter().enumerate().map(|(i, tok)| (tok.clone(), i)).collect();
        self.id2token = self.token2id.iter().map(|(k, &v)| (v, k.clone())).collect();
    }
    
    /// Count frequency of all adjacent pairs (bigrams) in the vocabulary
    fn get_stats(vocab: &HashMap<String, usize>) -> HashMap<(String, String), usize> {
        let mut pairs: HashMap<(String, String), usize> = HashMap::new();
        for (word, freq) in vocab {
            let symbols: Vec<&str> = word.split_whitespace().collect();
            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].to_string(), symbols[i + 1].to_string());
                *pairs.entry(pair).or_insert(0) += freq;
            }
        }
        pairs
    }
    
    fn merge_vocab() { }

    pub fn tokenize() { }

    pub fn encode() { }

    pub fn decode() { }




}