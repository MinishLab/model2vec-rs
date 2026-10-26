//! A purpose-built, IDs-only Unigram tokenizer.
//!
//! `encode_batch_fast` runs the `tokenizers` crate's Unigram model, which
//! walks a trie with a hash map per node, returns every token as a `String`,
//! looks each one up again in the vocabulary, and builds a full `Encoding`
//! (offsets, alignment, masks, type IDs) of which the static-embedding path
//! uses only `get_ids()`. This module keeps the crate's own added-token
//! extraction, normalizer and pre-tokenizer, and replaces only the model: the
//! same Viterbi search as `Unigram::encode_optimized`, over a flat byte trie
//! (CSR layout: each node's edges contiguous and sorted by byte), emitting IDs
//! directly. The IDs are identical to the `tokenizers` crate (verified by the
//! parity tests below).
//!
//! Exactness rests on doing what `encode_optimized` does in the same order:
//! candidates are visited by start position, then by length, and replace the
//! best path only on a strictly greater score, so ties break the same way;
//! a character with no single-character piece gets the unknown score
//! (`min_score - 10`); consecutive unknown pieces are fused into one string,
//! which then resolves as `Unigram::tokenize` resolves it (a piece if it spells
//! one, else its bytes under byte fallback, else the unknown id).
//!
//! Tokenizers whose output this module would not reproduce (truncation or
//! padding configured, or no `unk_id`) return `None` from
//! [`FastUnigram::from_tokenizer`] so the caller falls back to the crate.

use tokenizers::{Model, OffsetReferential, OffsetType, PreTokenizer, Tokenizer, models::ModelWrapper};

/// `tokenizers`' `K_UNK_PENALTY`: an unknown character scores this much below
/// the lowest-scoring piece.
const UNK_PENALTY: f64 = 10.0;

/// Marks an absent child node, an absent node token, or an unset path start.
const NONE: u32 = u32::MAX;

/// A byte trie over the vocabulary in CSR layout.
struct ByteTrie {
    /// Index of each node's first edge; node `n`'s edges are
    /// `edge_start[n]..edge_start[n + 1]`.
    edge_start: Vec<u32>,
    /// Each edge's byte, sorted within a node.
    edge_byte: Vec<u8>,
    /// Each edge's child node.
    edge_child: Vec<u32>,
    /// The token ID a node spells, or `NONE`.
    node_token: Vec<u32>,
}

impl ByteTrie {
    /// Build from `(piece, id)` pairs, breadth first, so every node's edges are
    /// appended together and in byte order.
    fn build(mut pieces: Vec<(&[u8], u32)>) -> ByteTrie {
        pieces.sort_unstable();
        pieces.dedup_by(|a, b| a.0 == b.0);
        let mut trie = ByteTrie {
            edge_start: vec![0],
            edge_byte: Vec::new(),
            edge_child: Vec::new(),
            node_token: vec![NONE],
        };
        // Each queued node covers the sorted pieces in `lo..hi` that share its
        // `depth`-byte prefix.
        let mut queue = std::collections::VecDeque::from([(0usize, 0usize, pieces.len())]);
        while let Some((depth, mut lo, hi)) = queue.pop_front() {
            let node = trie.edge_start.len() - 1;
            if lo < hi && pieces[lo].0.len() == depth {
                trie.node_token[node] = pieces[lo].1;
                lo += 1;
            }
            while lo < hi {
                let byte = pieces[lo].0[depth];
                let mut end = lo + 1;
                while end < hi && pieces[end].0[depth] == byte {
                    end += 1;
                }
                trie.edge_byte.push(byte);
                trie.edge_child.push(trie.node_token.len() as u32);
                trie.node_token.push(NONE);
                queue.push_back((depth + 1, lo, end));
                lo = end;
            }
            trie.edge_start.push(trie.edge_byte.len() as u32);
        }
        trie
    }

    /// The child of `node` along `byte`, or `NONE`.
    #[inline(always)]
    fn child(&self, node: u32, byte: u8) -> u32 {
        let lo = self.edge_start[node as usize] as usize;
        let hi = self.edge_start[node as usize + 1] as usize;
        let bytes = &self.edge_byte[lo..hi];
        // Most nodes have a handful of edges, where a scan beats a search.
        let found = if bytes.len() <= 8 {
            bytes.iter().position(|&b| b == byte)
        } else {
            bytes.binary_search(&byte).ok()
        };
        found.map_or(NONE, |i| self.edge_child[lo + i])
    }

    /// The token ID `key` spells, or `NONE`.
    fn get(&self, key: &[u8]) -> u32 {
        let mut node = 0;
        for &byte in key {
            node = self.child(node, byte);
            if node == NONE {
                return NONE;
            }
        }
        self.node_token[node as usize]
    }
}

/// The best path ending at a byte position of a pre-token.
#[derive(Clone, Copy)]
struct BestPath {
    score: f64,
    /// Byte position the last token starts at, or `NONE` while unset.
    starts_at: u32,
    token_id: u32,
}

const UNSET: BestPath = BestPath {
    score: 0.0,
    starts_at: NONE,
    token_id: 0,
};

/// IDs-only Unigram tokenizer over a byte trie of the model's vocabulary.
pub(crate) struct FastUnigram {
    trie: ByteTrie,
    /// Score of each token ID.
    scores: Vec<f64>,
    unk_id: u32,
    unk_score: f64,
    byte_fallback: bool,
    /// Token ID of each `<0xXX>` byte piece, or `NONE`.
    byte_ids: [u32; 256],
}

impl std::fmt::Debug for FastUnigram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("FastUnigram")
            .field("vocab", &self.scores.len())
            .field("trie_nodes", &self.trie.node_token.len())
            .field("unk_id", &self.unk_id)
            .field("byte_fallback", &self.byte_fallback)
            .finish()
    }
}

impl FastUnigram {
    /// Build from a loaded tokenizer and its `unk_id`, or `None` if the model
    /// is not Unigram or the pipeline does something this module skips.
    pub(crate) fn from_tokenizer(tok: &Tokenizer, unk_id: Option<usize>) -> Option<FastUnigram> {
        let ModelWrapper::Unigram(model) = tok.get_model() else {
            return None;
        };
        // Truncation and padding change the IDs after the model; without an
        // `unk_id` the crate fails on an unknown character.
        if tok.get_truncation().is_some() || tok.get_padding().is_some() {
            return None;
        }
        let unk_id = u32::try_from(unk_id?).ok()?;

        // A piece listed twice resolves to its last ID, as the crate's
        // `token_to_ids` map does, and scores as that ID.
        let mut scores = Vec::with_capacity(model.get_vocab_size());
        let mut pieces = Vec::with_capacity(model.get_vocab_size());
        for (piece, score) in model.iter() {
            scores.push(*score);
            pieces.push((piece.as_bytes(), model.token_to_id(piece)?));
        }
        let unk_score = model.min_score - UNK_PENALTY;
        let trie = ByteTrie::build(pieces);

        let mut byte_ids = [NONE; 256];
        for (byte, slot) in byte_ids.iter_mut().enumerate() {
            if let Some(id) = model.token_to_id(&format!("<0x{byte:02X}>")) {
                *slot = id;
            }
        }

        Some(FastUnigram {
            trie,
            scores,
            unk_id,
            unk_score,
            byte_fallback: model.byte_fallback(),
            byte_ids,
        })
    }

    /// Encode text into token IDs, matching `encode_batch_fast(...).get_ids()`
    /// (unknown pieces emit the unk id, which the caller filters out).
    /// `tok` must be the tokenizer this was built from.
    pub(crate) fn encode_ids(&self, tok: &Tokenizer, text: &str) -> tokenizers::Result<Vec<u32>> {
        // Added tokens, normalization and pre-tokenization exactly as
        // `encode_single_sequence` does them before the model.
        let mut pretokenized = tok
            .get_added_vocabulary()
            .extract_and_normalize(tok.get_normalizer(), text);
        if let Some(pre_tokenizer) = tok.get_pre_tokenizer() {
            pre_tokenizer.pre_tokenize(&mut pretokenized)?;
        }
        let mut out = Vec::new();
        let mut best = Vec::new();
        for (piece, _, tokens) in pretokenized.get_splits(OffsetReferential::Normalized, OffsetType::None) {
            match tokens {
                Some(tokens) => out.extend(tokens.iter().map(|t| t.id)),
                None => self.viterbi(piece, &mut best, &mut out),
            }
        }
        Ok(out)
    }

    // --- Unigram -----------------------------------------------------------

    /// Append the IDs of pre-token `text` to `out`, as `Unigram::tokenize`
    /// gives them. `best` is scratch space reused across pre-tokens.
    fn viterbi(&self, text: &str, best: &mut Vec<BestPath>, out: &mut Vec<u32>) {
        let bytes = text.as_bytes();
        let size = bytes.len();
        if size == 0 {
            return;
        }
        best.clear();
        best.resize(size + 1, UNSET);
        best[0].starts_at = 0;

        let mut starts_at = 0;
        while starts_at < size {
            let score_till_here = best[starts_at].score;
            let char_len = utf8_len(bytes[starts_at]);
            let mut has_single_node = false;
            // Every piece starting here, shortest first, as the crate's
            // `common_prefix_search` yields them.
            let (mut node, mut end) = (0, starts_at);
            while end < size {
                node = self.trie.child(node, bytes[end]);
                if node == NONE {
                    break;
                }
                end += 1;
                let id = self.trie.node_token[node as usize];
                if id != NONE {
                    self.relax(
                        &mut best[end],
                        self.scores[id as usize] + score_till_here,
                        starts_at,
                        id,
                    );
                    has_single_node |= end - starts_at == char_len;
                }
            }
            if !has_single_node {
                let target = &mut best[starts_at + char_len];
                self.relax(target, self.unk_score + score_till_here, starts_at, self.unk_id);
            }
            starts_at += char_len;
        }

        // Walk back from the end, fusing runs of unknown pieces. IDs are
        // pushed in reverse and the new tail of `out` is flipped at the end.
        let mark = out.len();
        let mut ends_at = size;
        let mut unk_run_end = None;
        while ends_at > 0 {
            let path = best[ends_at];
            let starts_at = path.starts_at as usize;
            if path.token_id == self.unk_id {
                unk_run_end.get_or_insert(ends_at);
            } else {
                if let Some(run_end) = unk_run_end.take() {
                    self.push_unknown_rev(&text[ends_at..run_end], out);
                }
                out.push(path.token_id);
            }
            ends_at = starts_at;
        }
        if let Some(run_end) = unk_run_end {
            self.push_unknown_rev(&text[..run_end], out);
        }
        out[mark..].reverse();
    }

    /// Take `candidate` as the best path into `target` if `target` is unset or
    /// `candidate` scores strictly higher.
    #[inline(always)]
    fn relax(&self, target: &mut BestPath, candidate: f64, starts_at: usize, token_id: u32) {
        if target.starts_at == NONE || candidate > target.score {
            *target = BestPath {
                score: candidate,
                starts_at: starts_at as u32,
                token_id,
            };
        }
    }

    /// Append, in reverse, the IDs of a fused unknown run: its piece if it
    /// spells one, else its bytes under byte fallback, else the unknown id.
    fn push_unknown_rev(&self, run: &str, out: &mut Vec<u32>) {
        let id = self.trie.get(run.as_bytes());
        if id != NONE {
            out.push(id);
            return;
        }
        if self.byte_fallback && run.bytes().all(|b| self.byte_ids[b as usize] != NONE) {
            out.extend(run.bytes().rev().map(|b| self.byte_ids[b as usize]));
            return;
        }
        out.push(self.unk_id);
    }
}

/// Length of the UTF-8 sequence that starts with `lead`.
#[inline(always)]
fn utf8_len(lead: u8) -> usize {
    match lead {
        0x00..=0x7f => 1,
        0x80..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::str::FromStr;

    fn corpus() -> Vec<String> {
        let mut texts: Vec<String> = [
            "The quick brown fox jumps over the lazy dog.",
            "It's 3:45 p.m. on 2026-10-05; the price is $1,299.99 (incl. 18% GST).",
            "Les élèves ont étudié l’histoire de la Révolution française.",
            "Größere Äpfel schmecken süßer als kleinere, sagt Jürgen.",
            "Москва — столица России, крупнейший город Европы.",
            "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας.",
            "İstanbul’da çay içip simit yedik.",
            "Hà Nội là thủ đô của Việt Nam.",
            "مرحبا بالعالم، كيف حالك اليوم؟",
            "שלום עולם, מה שלומך?",
            "नमस्ते दुनिया, आज मौसम अच्छा है।",
            "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।",
            "வணக்கம் உலகம், இன்று வானிலை நன்றாக உள்ளது.",
            "สวัสดีชาวโลก วันนี้อากาศดี",
            "你好，世界！今天天气很好。",
            "日本語の文章を正しく分割できますか？",
            "안녕하세요, 세계! 오늘 날씨가 좋네요.",
            "def f(x): return x**2  # squares\nprint(f(3))",
            "https://example.com/a/b?c=d&e=f#g user@example.org",
            "E = mc², ∑ᵢ xᵢ² ≤ ∞, ∀x∈ℝ ¯\\_(ツ)_/¯",
            // --- whitespace, control characters, empty input ---
            "",
            " ",
            "   ",
            "\t\n",
            "\u{0}\u{1}\u{1f}\u{7f}",
            "tab\tsep\u{b}vt\u{c}ff CRLF\r\nline\rcr",
            "  leading",
            "trailing  ",
            "a  b   c",
            "\u{a0}nbsp\u{3000}ideographic\u{2009}thin",
            "\u{200b}\u{200c}\u{200d}\u{feff}bom",
            // --- special and added tokens ---
            "<unk>",
            "<s>x</s>",
            "<pad></s>",
            "a<s>b <unk> c",
            "Hello World and hello world",
            "▁",
            "▁▁x",
            // --- normalization and unknown runs ---
            "\u{1F600} emoji 👩‍👩‍👧‍👦 👍🏽 🇮🇳",
            "ﬁ ﬂ Ⅻ ① ㎏ ㍿",
            "e\u{301} é ñ n\u{303}",
            "z̷̢̛a̴l̸g̵o̶",
            "ＦＵＬＬＷＩＤＴＨ ｶﾀｶﾅ",
            "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟙𝟚𝟛",
            "\u{10FFFF}",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        texts.push("x".repeat(3000));
        texts.push("नम".repeat(2000));
        texts.push(texts[..20].join(" "));
        texts.extend(noise(600));
        texts
    }

    /// Reproducible random strings drawn from one to three Unicode blocks.
    fn noise(n: usize) -> Vec<String> {
        const BLOCKS: [(u32, u32); 16] = [
            (0x20, 0x7e),
            (0xa0, 0xff),
            (0x300, 0x36f),
            (0x370, 0x3ff),
            (0x400, 0x4ff),
            (0x600, 0x6ff),
            (0x900, 0x97f),
            (0xb80, 0xbff),
            (0xe00, 0xe7f),
            (0x2000, 0x206f),
            (0x3040, 0x30ff),
            (0x4e00, 0x4fff),
            (0xac00, 0xadff),
            (0xff00, 0xffef),
            (0x1f300, 0x1f6ff),
            (0x0, 0x1f),
        ];
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        (0..n)
            .map(|_| {
                let blocks: Vec<_> = (0..1 + next(3)).map(|_| BLOCKS[next(16) as usize]).collect();
                (0..next(60))
                    .map(|_| {
                        if next(100) < 15 {
                            return ' ';
                        }
                        let (lo, hi) = blocks[next(blocks.len() as u64) as usize];
                        char::from_u32(lo + next((hi - lo + 1) as u64) as u32).unwrap_or(' ')
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_matches_crate(tok: &Tokenizer, texts: &[String]) {
        let unk_id = match tok.get_model() {
            ModelWrapper::Unigram(model) => serde_json::to_value(model).unwrap()["unk_id"].as_u64(),
            _ => panic!("not a Unigram tokenizer"),
        };
        let fast = FastUnigram::from_tokenizer(tok, unk_id.map(|id| id as usize)).expect("fast path applies");
        for text in texts {
            let want = tok
                .encode_fast(text.as_str(), false)
                .expect("crate encode")
                .get_ids()
                .to_vec();
            let got = fast.encode_ids(tok, text).expect("fast encode");
            assert_eq!(got, want, "token-id mismatch for {text:?}");
        }
    }

    /// A small Unigram tokenizer over pieces cut from `texts`, with scores
    /// drawn from a handful of values so that tied paths are common.
    fn small_unigram(texts: &[String], byte_fallback: bool) -> Tokenizer {
        let mut vocab = vec![json!(["<unk>", 0.0]), json!(["<s>", 0.0]), json!(["</s>", 0.0])];
        let mut seen = std::collections::HashSet::new();
        let mut i = 0u64;
        for text in texts.iter().take(60) {
            let chars: Vec<char> = format!("▁{}", text.replace(' ', "▁")).chars().collect();
            for width in 1..=4 {
                for window in chars.windows(width).step_by(width) {
                    let piece: String = window.iter().collect();
                    i += 1;
                    // Leave some single characters out so unknown runs occur.
                    if (width == 1 && i.is_multiple_of(5)) || !seen.insert(piece.clone()) {
                        continue;
                    }
                    vocab.push(json!([piece, -(((i * 7919) % 6) as f64) - 1.0]));
                }
            }
        }
        // A duplicate piece resolves to its last ID.
        vocab.push(json!(["▁the", -0.5]));
        if byte_fallback {
            for byte in 0..=255u8 {
                if !byte.is_multiple_of(3) {
                    vocab.push(json!([format!("<0x{byte:02X}>"), -20.0]));
                }
            }
        }
        let added = |id: usize, content: &str, normalized: bool, special: bool, single_word: bool| {
            json!({"id": id, "content": content, "single_word": single_word, "lstrip": false,
                   "rstrip": false, "normalized": normalized, "special": special})
        };
        let n = vocab.len();
        let json = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                added(0, "<unk>", false, true, false),
                added(1, "<s>", false, true, false),
                added(2, "</s>", false, true, false),
                added(n, "hello world", true, false, false),
                added(n + 1, "and", false, false, true),
            ],
            "normalizer": {"type": "NFKC"},
            "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "Unigram", "unk_id": 0, "vocab": vocab, "byte_fallback": byte_fallback},
        });
        Tokenizer::from_str(&json.to_string()).expect("build tokenizer")
    }

    #[test]
    fn fast_matches_crate() {
        let texts = corpus();
        assert_matches_crate(&small_unigram(&texts, false), &texts);
    }

    #[test]
    fn fast_matches_crate_with_byte_fallback() {
        let texts = corpus();
        assert_matches_crate(&small_unigram(&texts, true), &texts);
    }

    #[test]
    fn truncation_falls_back() {
        let texts = corpus();
        let mut tok = small_unigram(&texts, false);
        tok.with_truncation(Some(tokenizers::TruncationParams::default()))
            .unwrap();
        assert!(FastUnigram::from_tokenizer(&tok, Some(0)).is_none());
    }

    #[test]
    fn missing_unk_id_falls_back() {
        let tok = small_unigram(&corpus(), false);
        assert!(FastUnigram::from_tokenizer(&tok, None).is_none());
    }

    /// The multilingual POTION tokenizer, when it is already in the local
    /// Hugging Face cache (the test does not download it).
    #[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
    #[test]
    fn fast_matches_crate_on_cached_multilingual_model() {
        let cache = hf_hub::Cache::from_env();
        let Some(path) = cache
            .model("minishlab/potion-multilingual-128M".to_string())
            .get("tokenizer.json")
        else {
            eprintln!("skipped: minishlab/potion-multilingual-128M is not in the local cache");
            return;
        };
        let tok = Tokenizer::from_file(path).expect("load cached tokenizer");
        assert_matches_crate(&tok, &corpus());
    }
}
