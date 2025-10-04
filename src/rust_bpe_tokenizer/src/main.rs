
use rust_bpe::BpeTokenizer;
use std::fs;
use rust_bpe::{save_vocab, save_tokenized_text};

fn main() {

    let text = fs::read_to_string("data/training.txt")
        .expect("Failed to read training file");
    
    let mut tokenizer = BpeTokenizer::new(100);
    tokenizer.fit(&text);

    let tokens = tokenizer.encode(&text);
    save_vocab(&tokenizer, "vocab.json");
    save_tokenized_text(&tokens, "tokens.json");

    println!("Saved vocab.json and tokens.json");
}