use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

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
        // Encode multiple sentences from a file or input string
        Commands::Encode { input, model, output } => {
            let texts = if Path::new(&input).exists() {
                std::fs::read_to_string(&input)?.lines().map(str::to_string).collect()
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
        // Encode a single sentence
        Commands::EncodeSingle {
            sentence,
            model,
            output,
        } => {
            let m = StaticModel::from_pretrained(&model, None, None, None)?;
            let embedding = m.encode_single(&sentence);

            if let Some(path) = output {
                let file = File::create(path).context("creating output file failed")?;
                serde_json::to_writer(BufWriter::new(file), &embedding).context("writing JSON failed")?;
            } else {
                println!("{embedding:#?}");
            }
        }
    }
    Ok(())
}
