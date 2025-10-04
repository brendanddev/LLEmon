
import std::collections::HashMap;

impl BpeTokenizer {

    // Constructs a new BpeTokenizer instance
    pub fn new() -> Self {
        BpeTokenizer {
            vocab: HashMap::new(),
            merges: Vec::new(),
        }
    }




}