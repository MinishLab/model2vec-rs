use clap::{Parser, Subcommand};
use anyhow::Result;
use std::path::Path;

mod inference;

#[derive(Parser)]
#[command(author, version, about = "Model2Vec Rust CLI")] 
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference with a pre-trained Model2Vec model
    Encode {
        /// Input text or file path
        input: String,
        /// Hugging Face repo ID
        model: String,
        /// Output file to save embeddings (JSON)
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Encode { input, model, output } => {
            // Read input texts
            let texts = if Path::new(&input).exists() {
                std::fs::read_to_string(&input)?.lines()
                    .map(str::to_string).collect()
            } else { vec![input.clone()] };

            // Load model and encode
            let m = inference::StaticModel::from_pretrained(&model)?;
            let embeds = m.encode(&texts);

            // Output
            if let Some(path) = output {
                let json = serde_json::to_string(&embeds)?;
                std::fs::write(path, json)?;
            } else {
                println!("{:?}", embeds);
            }
        }
    }
    Ok(())
}