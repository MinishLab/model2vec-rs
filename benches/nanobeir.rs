// NanoBEIR retrieval quality (NDCG@10) of one model, driven by
// `scripts/nanobeir.py`, which downloads the data and gates releases on the
// scores. Uses only the public API so the same file builds against any
// revision.
//
//   cargo bench --bench nanobeir -- data=target/nanobeir/data model=minishlab/potion-base-8M \
//       [quantize=int8]
//
// `data` holds one directory per dataset with `corpus.jsonl` and
// `queries.jsonl` (`{"_id", "text"}`) and `qrels.jsonl`
// (`{"query-id", "corpus-id"}`, binary relevance). Documents are ranked by
// cosine similarity; a document with the same id as the query is never
// retrieved, as in BEIR, and ties go to the larger document id, as in
// trec_eval. Prints one `SCORE <dataset> <ndcg@10>` line per
// dataset.

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::Path;

use model2vec_rs::model::StaticModel;
use serde_json::Value;

const K: usize = 10;

fn read_jsonl(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad line in {}: {e}", path.display())))
        .collect()
}

fn field(v: &Value, key: &str) -> String {
    v[key]
        .as_str()
        .unwrap_or_else(|| panic!("missing string field {key:?} in {v}"))
        .to_string()
}

fn read_texts(path: &Path) -> (Vec<String>, Vec<String>) {
    read_jsonl(path)
        .iter()
        .map(|v| (field(v, "_id"), field(v, "text")))
        .unzip()
}

fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn ndcg_at_k(ranked: &[usize], doc_ids: &[String], relevant: &HashSet<String>) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .enumerate()
        .filter(|(_, d)| relevant.contains(&doc_ids[**d]))
        .map(|(rank, _)| 1.0 / (rank as f64 + 2.0).log2())
        .sum();
    let idcg: f64 = (0..relevant.len().min(K))
        .map(|rank| 1.0 / (rank as f64 + 2.0).log2())
        .sum();
    dcg / idcg
}

fn evaluate(model: &StaticModel, dir: &Path) -> f64 {
    let (doc_ids, docs) = read_texts(&dir.join("corpus.jsonl"));
    let (query_ids, queries) = read_texts(&dir.join("queries.jsonl"));
    let mut qrels: HashMap<String, HashSet<String>> = HashMap::new();
    for v in read_jsonl(&dir.join("qrels.jsonl")) {
        qrels
            .entry(field(&v, "query-id"))
            .or_default()
            .insert(field(&v, "corpus-id"));
    }

    let doc_emb: Vec<Vec<f32>> = model.encode(&docs).into_iter().map(normalized).collect();
    let query_emb: Vec<Vec<f32>> = model.encode(&queries).into_iter().map(normalized).collect();

    let mut total = 0.0;
    let mut count = 0;
    for (qid, q) in query_ids.iter().zip(&query_emb) {
        let Some(relevant) = qrels.get(qid) else { continue };
        let mut scored: Vec<(f32, usize)> = doc_emb
            .iter()
            .enumerate()
            .filter(|(d, _)| &doc_ids[*d] != qid)
            .map(|(d, e)| (q.iter().zip(e).map(|(a, b)| a * b).sum(), d))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| doc_ids[b.1].cmp(&doc_ids[a.1])));
        let ranked: Vec<usize> = scored.iter().take(K).map(|&(_, d)| d).collect();
        total += ndcg_at_k(&ranked, &doc_ids, relevant);
        count += 1;
    }
    assert!(count > 0, "no queries with relevance judgements in {}", dir.display());
    total / count as f64
}

fn main() {
    let args: HashMap<String, String> = env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
        .collect();
    let data = args
        .get("data")
        .expect("usage: nanobeir data=<dir> [model=..] [quantize=none|int8]");
    let model_id = args.get("model").map_or("minishlab/potion-base-8M", String::as_str);
    let quantize = args.get("quantize").map_or("none", String::as_str);

    let model = StaticModel::from_pretrained(model_id, None, None, None)
        .unwrap_or_else(|e| panic!("failed to load {model_id}: {e}"));
    let model = match quantize {
        "none" => model,
        "int8" => model.quantize_int8(),
        other => panic!("unknown quantize mode {other:?}"),
    };

    let mut datasets: Vec<_> = fs::read_dir(data)
        .unwrap_or_else(|e| panic!("failed to read {data}: {e}"))
        .map(|e| e.expect("failed to read directory entry").path())
        .filter(|p| p.join("corpus.jsonl").exists())
        .collect();
    datasets.sort();
    assert!(!datasets.is_empty(), "no datasets found in {data}");

    for dir in datasets {
        let name = dir.file_name().unwrap().to_string_lossy();
        println!("SCORE {name} {:.6}", evaluate(&model, &dir));
    }
}
