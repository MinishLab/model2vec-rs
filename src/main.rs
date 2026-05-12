use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

mod model;
use model::StaticModel;

fn write_output<T: serde::Serialize + std::fmt::Debug>(data: &T, path: Option<String>) -> Result<()> {
    match path {
        Some(p) => {
            let file = File::create(&p).context("failed to create output file")?;
            serde_json::to_writer(BufWriter::new(file), data).context("failed to write JSON")
        }
        None => {
            println!("{data:#?}");
            Ok(())
        }
    }
}

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
    /// Encode a single sentence  
    EncodeSingle {
        /// The sentence to embed
        sentence: String,
        /// HF repo ID or local dir
        model: String,
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Encode { input, model, output } => {
            let texts = if Path::new(&input).exists() {
                std::fs::read_to_string(&input)?.lines().map(str::to_string).collect()
            } else {
                vec![input]
            };
            let embs = StaticModel::from_pretrained(&model, None, None, None)?.encode(&texts);
            write_output(&embs, output)?;
        }
        Commands::EncodeSingle { sentence, model, output } => {
            let embedding = StaticModel::from_pretrained(&model, None, None, None)?.encode_single(&sentence);
            write_output(&embedding, output)?;
        }
    }
    Ok(())
}
