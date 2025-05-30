use clap::{Parser, Subcommand};
use anyhow::{Context, Result};
use std::path::Path;
use std::fs::File;
use std::io::BufWriter;

mod model;
use model::StaticModel;

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

            let m = StaticModel::from_pretrained(&model, None, None, None)?;
            let embs = m.encode(&texts);
            
            if let Some(output) = output {
                let file = File::create(&output).context("failed to create output file")?;
                let writer = BufWriter::new(file);
                serde_json::to_writer(writer, &embs).context("failed to write embeddings to JSON")?;
            } else {
                println!("{:?}", embs);
            }
        }
    }
    Ok(())
}
