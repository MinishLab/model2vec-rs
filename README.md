# model2vec-rs

This crate provides a lightweight Rust implementation for loading and running inference on Model2Vec static embedding models (including quantized formats: float32, float16, int8) from either local folders or the Hugging Face Hub.

## Quick Start

Install the crate:

```bash
git clone https://github.com/minishlab/model2vec-rust.git
cd model2vec-rs

# Build
cargo build --release
```

Make embeddings:

```rust
use anyhow::Result;
use model2vec_rust::inference::StaticModel;

fn main() -> Result<()> {
    // Load a model from the Hugging Face Hub or a local path
    let model = StaticModel::from_pretrained("minishlab/potion-base-8M", None)?;

    // Prepare a list of sentences
    let texts = vec![
        "Hello world".to_string(),
        "Rust is awesome".to_string(),
    ];

    // Create embeddings
    let embeddings = model.encode(&texts);
    println!("Embeddings: {:?}", embeddings);

    Ok(())
}
```


Make embeddings with the CLI:

```rust
# Single sentence
cargo run -- encode "Hello world" minishlab/potion-base-8M

# Multiple lines from a file
echo -e "Hello world\nRust is awesome" > input.txt
cargo run -- encode input.txt minishlab/potion-base-8M --output embeds.json

