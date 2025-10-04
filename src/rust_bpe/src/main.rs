
use rust_bpe::BpeTokenizer;

/// Test for the BPE tokenizer
fn main() {
    let text = "hello world hello byte pair encoding";
    let mut tokenizer = BpeTokenizer::new(50);
    tokenizer.fit(text);

    println!("Token IDs for 'hello world': {:?}", tokenizer.encode("hello world"));
    println!("Decoded back: {}", tokenizer.decode(&tokenizer.encode("hello world")));
}