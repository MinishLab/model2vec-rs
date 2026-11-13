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
//! Tokenizers whose output this module would not reproduce (no `unk_id`)
//! return `None` from
//! [`FastUnigram::from_tokenizer`] so the caller falls back to the crate.
//!
//! The crate's Unigram model is only needed to build this one, and it is
//! large: its trie keeps a hash map per node. [`FastUnigram::from_json`]
//! builds the trie straight from `tokenizer.json` instead, for the common
//! layout where every added token is one of the first pieces of the
//! vocabulary, so the crate's model is never built.

use crate::model::median_token_length;
use crate::trie::{DoubleArrayTrie, NONE};
use serde::Deserialize;
use serde_json::value::RawValue;
use std::collections::HashMap;
use tokenizers::{Model, OffsetReferential, OffsetType, PreTokenizer, Tokenizer, models::ModelWrapper};

/// `tokenizers`' `K_UNK_PENALTY`: an unknown character scores this much below
/// the lowest-scoring piece.
const UNK_PENALTY: f64 = 10.0;

/// The most pieces the tokenizer returned by [`FastUnigram::from_json`] keeps.
const MAX_KEPT_PIECES: usize = 4096;

/// The part of an `added_tokens` entry that [`FastUnigram::from_json`] reads.
#[derive(Deserialize)]
struct AddedTokenJson {
    content: String,
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
#[derive(Clone)]
pub(crate) struct FastUnigram {
    trie: DoubleArrayTrie,
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
            .field("trie_nodes", &self.trie.len())
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
        // Without an `unk_id` the crate fails on an unknown character.
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
        let trie = DoubleArrayTrie::build(pieces);

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

    /// Build from `tokenizer.json` bytes without the crate's Unigram model, or
    /// `None` when [`FastUnigram::from_tokenizer`] has to do it. Returns the
    /// tokenizer to pass to `encode_ids`, the fast model and the median token
    /// length of the vocabulary.
    ///
    /// `encode_ids` takes the added vocabulary, the normalizer and the
    /// pre-tokenizer from its tokenizer, never the model. So the vocabulary
    /// goes straight into the trie, and the returned tokenizer is built from
    /// the same JSON with the vocabulary cut to its first pieces: enough for
    /// every added token to get the ID it has in the full vocabulary. That
    /// tokenizer is of no use for anything but `encode_ids`.
    ///
    /// Applies to a Unigram model with an `unk_id` when the unknown piece and
    /// every added token are among the first `MAX_KEPT_PIECES` pieces.
    pub(crate) fn from_json(bytes: &[u8]) -> Option<(Tokenizer, FastUnigram, usize)> {
        // Fields stay unparsed until the model is known to be Unigram, so
        // other tokenizers only pay for a scan, and of the model only the
        // vocabulary is parsed.
        let mut json: HashMap<&str, &RawValue> = serde_json::from_slice(bytes).ok()?;
        let mut model: HashMap<&str, &RawValue> = serde_json::from_str(json.get("model")?.get()).ok()?;
        if serde_json::from_str::<String>(model.get("type")?.get()).ok()? != "Unigram" {
            return None;
        }
        // Without an `unk_id` the crate fails on an unknown character.
        let unk_id: u32 = serde_json::from_str(model.get("unk_id")?.get()).ok()?;
        let byte_fallback = match model.get("byte_fallback") {
            Some(raw) => serde_json::from_str(raw.get()).ok()?,
            None => false,
        };
        let pieces: Vec<(String, f64)> = serde_json::from_str(model.get("vocab")?.get()).ok()?;
        let added_tokens: Vec<AddedTokenJson> = match json.get("added_tokens") {
            Some(raw) => serde_json::from_str::<Option<_>>(raw.get()).ok()?.unwrap_or_default(),
            None => Vec::new(),
        };

        // A piece listed twice resolves to its last ID, as in `from_tokenizer`.
        let mut ids: HashMap<&str, u32> = HashMap::with_capacity(pieces.len());
        for (id, (piece, _)) in pieces.iter().enumerate() {
            ids.insert(piece.as_str(), u32::try_from(id).ok()?);
        }

        // The crate gives an added token that is in the vocabulary its
        // vocabulary ID, and any other token the next ID past the vocabulary.
        // With every added token in the kept pieces, the cut vocabulary
        // resolves them all as the full one does.
        let mut keep = unk_id as usize + 1;
        for token in &added_tokens {
            keep = keep.max(*ids.get(token.content.as_str())? as usize + 1);
        }
        if keep > MAX_KEPT_PIECES || keep > pieces.len() {
            return None;
        }

        let scores: Vec<f64> = pieces.iter().map(|(_, score)| *score).collect();
        let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);
        let trie = DoubleArrayTrie::build(
            pieces
                .iter()
                .map(|(piece, _)| (piece.as_bytes(), ids[piece.as_str()]))
                .collect(),
        );
        let mut byte_ids = [NONE; 256];
        for (byte, slot) in byte_ids.iter_mut().enumerate() {
            if let Some(id) = ids.get(format!("<0x{byte:02X}>").as_str()) {
                *slot = *id;
            }
        }
        let median = median_token_length(ids.keys().copied());
        drop(ids);

        let kept_vocab = RawValue::from_string(serde_json::to_string(&pieces[..keep]).ok()?).ok()?;
        drop(pieces);
        model.insert("vocab", &kept_vocab);
        let kept_model = RawValue::from_string(serde_json::to_string(&model).ok()?).ok()?;
        json.insert("model", &kept_model);
        let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&json).ok()?).ok()?;

        let fast = FastUnigram {
            trie,
            scores,
            unk_id,
            unk_score: min_score - UNK_PENALTY,
            byte_fallback,
            byte_ids,
        };
        Some((tokenizer, fast, median))
    }

    /// ID of the unknown token.
    pub(crate) fn unk_id(&self) -> u32 {
        self.unk_id
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
                let id = self.trie.token(node);
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
    use serde_json::{Value, json};
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

    /// Checks `from_json` against the crate's tokenizer on `texts`, and
    /// returns the tokenizer `from_json` kept.
    fn assert_from_json_matches_crate(bytes: &[u8], texts: &[String]) -> Tokenizer {
        let full = Tokenizer::from_bytes(bytes).expect("load tokenizer");
        let unk_id = match full.get_model() {
            ModelWrapper::Unigram(model) => serde_json::to_value(model).unwrap()["unk_id"].as_u64(),
            _ => panic!("not a Unigram tokenizer"),
        };
        let (kept, fast, median) = FastUnigram::from_json(bytes).expect("from_json applies");
        assert_eq!(Some(fast.unk_id() as u64), unk_id);
        assert_eq!(
            median,
            median_token_length(full.get_vocab(false).keys().map(String::as_str))
        );
        for text in texts {
            let want = full
                .encode_fast(text.as_str(), false)
                .expect("crate encode")
                .get_ids()
                .to_vec();
            let got = fast.encode_ids(&kept, text).expect("fast encode");
            assert_eq!(got, want, "token-id mismatch for {text:?}");
        }
        kept
    }

    /// A small Unigram tokenizer over pieces cut from `texts`, with scores
    /// drawn from a handful of values so that tied paths are common.
    fn small_unigram(texts: &[String], byte_fallback: bool) -> Tokenizer {
        Tokenizer::from_str(&small_unigram_json(texts, byte_fallback).to_string()).expect("build tokenizer")
    }

    /// `small_unigram_json` with only the added tokens that are pieces of the
    /// vocabulary, which is the layout `from_json` applies to.
    fn small_unigram_json_with_added_pieces(texts: &[String], byte_fallback: bool) -> Value {
        let mut json = small_unigram_json(texts, byte_fallback);
        json["added_tokens"].as_array_mut().unwrap().truncate(3);
        json
    }

    fn added_piece(id: usize, content: &str) -> Value {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false,
               "rstrip": false, "normalized": false, "special": true})
    }

    fn small_unigram_json(texts: &[String], byte_fallback: bool) -> Value {
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
        json!({
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
        })
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
    fn missing_unk_id_falls_back() {
        let tok = small_unigram(&corpus(), false);
        assert!(FastUnigram::from_tokenizer(&tok, None).is_none());
    }

    #[test]
    fn from_json_matches_crate() {
        let texts = corpus();
        for byte_fallback in [false, true] {
            let json = small_unigram_json_with_added_pieces(&texts, byte_fallback);
            let kept = assert_from_json_matches_crate(json.to_string().as_bytes(), &texts);
            // Only the pieces that carry the added tokens' IDs.
            assert_eq!(kept.get_vocab_size(false), 3);
        }
    }

    #[test]
    fn from_json_keeps_the_pieces_up_to_the_last_added_token() {
        let texts = corpus();
        let mut json = small_unigram_json_with_added_pieces(&texts, false);
        // An added token that is a piece listed twice takes its last ID.
        let vocab = json["model"]["vocab"].as_array_mut().unwrap();
        let piece = vocab[5][0].as_str().unwrap().to_string();
        vocab.insert(20, json!([piece, -0.5]));
        assert_eq!(vocab.iter().rposition(|listed| listed[0] == piece.as_str()), Some(20));
        json["added_tokens"]
            .as_array_mut()
            .unwrap()
            .push(added_piece(20, &piece));
        let kept = assert_from_json_matches_crate(json.to_string().as_bytes(), &texts);
        assert_eq!(kept.get_vocab_size(false), 21);
    }

    #[test]
    fn from_json_matches_crate_on_unigram_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-vocab-quantized/tokenizer.json"
        );
        let kept = assert_from_json_matches_crate(&std::fs::read(path).unwrap(), &corpus());
        assert_eq!(kept.get_vocab_size(false), 2);
    }

    #[test]
    fn from_json_falls_back_when_an_added_token_is_not_a_piece() {
        // "hello world" and "and" get IDs past the vocabulary.
        let json = small_unigram_json(&corpus(), false);
        assert!(FastUnigram::from_json(json.to_string().as_bytes()).is_none());
    }

    #[test]
    fn from_json_falls_back_when_an_added_token_is_a_late_piece() {
        let mut json = small_unigram_json_with_added_pieces(&corpus(), false);
        let vocab = json["model"]["vocab"].as_array_mut().unwrap();
        while vocab.len() <= MAX_KEPT_PIECES {
            vocab.push(json!([format!("piece{}", vocab.len()), -30.0]));
        }
        let late = format!("piece{MAX_KEPT_PIECES}");
        json["added_tokens"]
            .as_array_mut()
            .unwrap()
            .push(added_piece(MAX_KEPT_PIECES, &late));
        assert!(FastUnigram::from_json(json.to_string().as_bytes()).is_none());
    }

    #[test]
    fn from_json_falls_back_without_unk_id() {
        let mut json = small_unigram_json_with_added_pieces(&corpus(), false);
        json["model"]["unk_id"] = Value::Null;
        assert!(FastUnigram::from_json(json.to_string().as_bytes()).is_none());
    }

    #[test]
    fn from_json_falls_back_for_other_models() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-int8/tokenizer.json"
        );
        assert!(FastUnigram::from_json(&std::fs::read(path).unwrap()).is_none());
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
        let tok = Tokenizer::from_file(&path).expect("load cached tokenizer");
        assert_matches_crate(&tok, &corpus());
        drop(tok);
        assert_from_json_matches_crate(&std::fs::read(path).expect("read cached tokenizer"), &corpus());
    }
}
