
use std::collections::HashSet;
use std::collections::HashMap;
use regex::Regex;


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
    pub fn fit(&mut self, text: &str) {
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
    
    /// Performs one merge operation on the vocabulary
    fn merge_vocab(pair: (String, String), vocab: &HashMap<String, usize>) -> HashMap<String, usize> {
        let mut new_vocab: HashMap<String, usize> = HashMap::new();
        let pattern = Regex::new(&format!(r"(?<!\S){}\s{}(?!\S)", regex::escape(&pair.0), regex::escape(&pair.1))).unwrap();

        for (word, freq) in vocab {
            let new_word = pattern.replace_all(word, format!("{}{}", pair.0, pair.1)).to_string();
            new_vocab.insert(new_word, *freq);
        }
        new_vocab
    }

    /// Tokenize a single word into subword using learned merges
    pub fn tokenize(&self, word: &str) -> Vec<String> {
        let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        chars.push("</w>".to_string());
        let mut i = 0;

        while i < chars.len() - 1 {
            let pair = (chars[i].clone(), chars[i + 1].clone());
            if self.merges.contains(&pair) {
                chars.splice(i..=i+1, std::iter::once(format!("{}{}", pair.0, pair.1)));
            } else {
                i += 1;
            }
        }
        
        // Remove end of word token before returning
        chars.pop();
        chars
    }

    /// Encode a string into token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            let subwords = self.tokenize(word);
            for sw in subwords {
                // Default to 0 if token not found
                tokens.push(*self.token2id.get(&sw).unwrap_or(&0));
            }
        }
        tokens
    }

    /// Decode a sequence of token IDs back into a string
    pub fn decode(&self, ids: Vec<usize>) -> String {
        ids.iter()
            .map(|id| self.id2token.get(id).unwrap_or(&"<unk>".to_string()).clone())
            .collect::<Vec<String>>()
            .join("")
            .replace("</w>", " ")
            .trim()
            .to_string()
    }




}