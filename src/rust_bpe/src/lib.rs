
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
        // Each character becomes a token intially, with a special </w> token to mark word boundaries
        let mut vocab: HashMap<String, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let chars_with_end: Vec<String> = word.chars().map(|c| c.to_string()).chain(std::iter::once("</w>".to_string())).collect();
            // Store as space-separated string for easy merging and count word frequency
            let key = chars_with_end.join(" ");
            *vocab.entry(key).or_insert(0) += 1;
        }

        // Perform BPE merges
        for _ in 0..self.num_merges {
            // Count adjacent pairs
            let pairs = Self::get_stats(&vocab);
            if pairs.is_empty() {
                break;
            }
            // Find most frequent pair
            let best = pairs.iter().max_by_key(|(_, count)| *count).unwrap().0.clone();
            vocab = Self::merge_vocab(best.clone(), &vocab);
            self.merges.push(best);
        }

        // Collect final unique tokens
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

        for (word, freq) in vocab {
            // Split the word into tokens seperated by space
            let mut tokens: Vec<String> = word.split_whitespace().map(|s| s.to_string()).collect();
            let mut i = 0;

            // Merge adjacent pairs matching 'pair'
            while i < tokens.len() - 1 {
                if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    tokens.splice(i..=i+1, [format!("{}{}", pair.0, pair.1)]);
                } else {
                    i += 1;
                }
            }

            // Join tokens back into a space seperated string
            let new_word = tokens.join(" ");
            new_vocab.insert(new_word, *freq);
        }
        new_vocab
    }

    /// Tokenize a single word into subword using learned merges
    pub fn tokenize(&self, word: &str) -> Vec<String> {
        // Start with characters plus end of word token
        let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        chars.push("</w>".to_string());
        let mut i = 0;

        while i < chars.len() - 1 {
            let pair = (chars[i].clone(), chars[i + 1].clone());
            if self.merges.contains(&pair) {
                // Merge the pair into a single token
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
                // Map subword to its ID, defaulting to 0 if not found
                tokens.push(*self.token2id.get(&sw).unwrap_or(&0));
            }
        }
        tokens
    }

    /// Decode a sequence of token IDs back into a string
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|id| self.id2token.get(id).unwrap_or(&"<unk>".to_string()).clone())
            .collect::<Vec<String>>()
            .join("")
            .replace("</w>", " ")
            .trim()
            .to_string()
    }


}