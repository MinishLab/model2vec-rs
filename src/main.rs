use clap::{Parser, Subcommand};
use anyhow::Result;
use std::path::Path;

mod inference;
use inference::StaticModel;

#[derive(Parser)]
#[command(author, version, about = "Model2Vec Rust CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode input texts into embeddings
    Encode {
        /// Input text or path to file (one sentence per line)
        input: String,
        /// Hugging Face repo ID or local path
        model: String,
        /// Optional output file (JSON) for embeddings
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Show token ID sequences for input texts
    Tokens {
        /// Input text or path to file
        input: String,
        /// Hugging Face repo ID or local path
        model: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Encode { input, model, output } => {
            let texts = if Path::new(&input).exists() {
                std::fs::read_to_string(&input)?
                    .lines()
                    .map(str::to_string)
                    .collect()
            } else {
                vec![input]
            };

            let m = StaticModel::from_pretrained(&model, None)?;
            let embs = m.encode(&texts);

            if let Some(path) = output {
                let json = serde_json::to_string(&embs)?;
                std::fs::write(path, json)?;
            } else {
                println!("{:#?}", embs);
            }
        }

        Commands::Tokens { input, model } => {
            let texts = if Path::new(&input).exists() {
                std::fs::read_to_string(&input)?
                    .lines()
                    .map(str::to_string)
                    .collect()
            } else {
                vec![input]
            };

            let m = StaticModel::from_pretrained(&model, None)?;
            let ids = m.tokenize(&texts);
            println!("Token ID sequences: {:#?}", ids);
        }
    }
    Ok(())
}
