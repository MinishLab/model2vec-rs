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
//! builds the trie straight from `tokenizer.json` instead, and the added
//! vocabulary with the crate's own `AddedVocabulary`, so the crate's model is
//! never built. Either way, no `Tokenizer` is kept.

use crate::model::median_token_length;
use crate::trie::{DoubleArrayTrie, NONE};
use serde::de::{DeserializeOwned, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::value::RawValue;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::{
    AddedToken, AddedVocabulary, DecoderWrapper, Model, NormalizerWrapper, OffsetReferential, OffsetType,
    PaddingParams, PostProcessorWrapper, PreTokenizer, PreTokenizerWrapper, Token, Tokenizer, TruncationParams,
    models::{ModelWrapper, unigram::UnigramTrainer},
};

/// `tokenizers`' `K_UNK_PENALTY`: an unknown character scores this much below
/// the lowest-scoring piece.
const UNK_PENALTY: f64 = 10.0;

/// An `added_tokens` entry, in the shape the crate deserializes it.
#[derive(Deserialize)]
struct AddedTokenJson {
    /// Required, as in the crate, which only uses it for a warning: an added
    /// token's ID comes from the vocabulary.
    #[serde(rename = "id")]
    _id: u32,
    #[serde(flatten)]
    token: AddedToken,
}

/// The vocabulary of a `tokenizer.json`, answering as the crate's Unigram
/// model answers for it, so that `AddedVocabulary::add_tokens` resolves the
/// added tokens' IDs without that model being built. `add_tokens` asks only
/// for `token_to_id` and `get_vocab_size`.
struct JsonVocab<'a> {
    pieces: &'a [(String, f64)],
    /// ID of each piece; a piece listed twice has its last ID.
    ids: HashMap<&'a str, u32>,
}

impl<'a> JsonVocab<'a> {
    /// The vocabulary of `[piece, score]` pairs, or `None` when an ID does
    /// not fit in `u32`.
    fn new(pieces: &'a [(String, f64)]) -> Option<JsonVocab<'a>> {
        let mut ids = HashMap::with_capacity(pieces.len());
        for (id, (piece, _)) in pieces.iter().enumerate() {
            ids.insert(piece.as_str(), u32::try_from(id).ok()?);
        }
        Some(JsonVocab { pieces, ids })
    }
}

impl Model for JsonVocab<'_> {
    type Trainer = UnigramTrainer;

    fn tokenize(&self, _sequence: &str) -> tokenizers::Result<Vec<Token>> {
        Err("JsonVocab only resolves token IDs".into())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.pieces.get(id as usize).map(|(piece, _)| piece.clone())
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.ids.iter().map(|(piece, id)| (piece.to_string(), *id)).collect()
    }

    /// Every listed piece, a piece listed twice included, as `Unigram` counts.
    fn get_vocab_size(&self) -> usize {
        self.pieces.len()
    }

    fn save(&self, _folder: &Path, _prefix: Option<&str>) -> tokenizers::Result<Vec<PathBuf>> {
        Err("JsonVocab only resolves token IDs".into())
    }

    fn get_trainer(&self) -> UnigramTrainer {
        UnigramTrainer::default()
    }
}

/// Field `key` of a JSON object deserialized, its default when absent, or
/// `None` when it does not deserialize.
fn field<T: DeserializeOwned + Default>(object: &HashMap<&str, &RawValue>, key: &str) -> Option<T> {
    match object.get(key) {
        Some(raw) => serde_json::from_str(raw.get()).ok(),
        None => Some(T::default()),
    }
}

/// The top-level keys of `tokenizer.json` that the crate reads.
const TOP_LEVEL_KEYS: [&str; 9] = [
    "version",
    "truncation",
    "padding",
    "added_tokens",
    "normalizer",
    "pre_tokenizer",
    "post_processor",
    "decoder",
    "model",
];

/// The top-level fields of `tokenizer.json`, unparsed, or `None` for a key
/// the crate does not know or a key listed twice.
fn top_level(bytes: &[u8]) -> Option<HashMap<&str, &RawValue>> {
    struct Fields;

    impl<'de> Visitor<'de> for Fields {
        type Value = HashMap<&'de str, &'de RawValue>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a JSON object")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut fields = HashMap::new();
            while let Some((key, value)) = map.next_entry()? {
                // The crate fails on a key it does not know, and reads every copy of a repeated one.
                if !TOP_LEVEL_KEYS.contains(&key) || fields.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom("unknown or repeated key"));
                }
            }
            Ok(fields)
        }
    }

    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let fields = deserializer.deserialize_map(Fields).ok()?;
    deserializer.end().ok()?;
    Some(fields)
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

/// IDs-only Unigram tokenizer over a byte trie of the model's vocabulary,
/// behind the crate's own added-token extraction, normalizer and
/// pre-tokenizer.
#[derive(Clone)]
pub(crate) struct FastUnigram {
    added_vocabulary: AddedVocabulary,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: Option<PreTokenizerWrapper>,
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
    /// is not Unigram or the pipeline does something this module skips. The
    /// added vocabulary, normalizer and pre-tokenizer are copied out of `tok`,
    /// so the caller can drop it.
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
            added_vocabulary: tok.get_added_vocabulary().clone(),
            normalizer: tok.get_normalizer().cloned(),
            pre_tokenizer: tok.get_pre_tokenizer().cloned(),
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
    /// tokenizer and the median token length of the vocabulary.
    ///
    /// The trie, scores and byte-fallback IDs are built from the `[piece,
    /// score]` list as `from_tokenizer` builds them from the crate's model.
    /// The normalizer, pre-tokenizer and added tokens are deserialized into
    /// the crate's own types, and the added vocabulary is built as `Tokenizer`
    /// deserialization builds it, with [`JsonVocab`] in place of the model.
    ///
    /// Applies to a Unigram model with an `unk_id`. A file the crate would
    /// refuse returns `None`, so loading it fails with the crate's error.
    pub(crate) fn from_json(bytes: &[u8]) -> Option<(FastUnigram, usize)> {
        // Fields stay unparsed until the model is known to be Unigram, so
        // other tokenizers only pay for scanning the file.
        let json = top_level(bytes)?;
        let model: HashMap<&str, &RawValue> = serde_json::from_str(json.get("model")?.get()).ok()?;
        if serde_json::from_str::<String>(model.get("type")?.get()).ok()? != "Unigram" {
            return None;
        }
        // Without an `unk_id` the crate fails on an unknown character.
        let unk_id: u32 = serde_json::from_str(model.get("unk_id")?.get()).ok()?;
        let byte_fallback: bool = field(&model, "byte_fallback")?;
        let pieces: Vec<(String, f64)> = serde_json::from_str(model.get("vocab")?.get()).ok()?;
        // The crate refuses an `unk_id` outside the vocabulary.
        if unk_id as usize >= pieces.len() {
            return None;
        }

        // A piece listed twice resolves to its last ID, as in `from_tokenizer`.
        let vocab = JsonVocab::new(&pieces)?;

        let scores: Vec<f64> = pieces.iter().map(|(_, score)| *score).collect();
        let unk_score = scores.iter().copied().fold(f64::INFINITY, f64::min) - UNK_PENALTY;
        let trie = DoubleArrayTrie::build(vocab.ids.iter().map(|(piece, &id)| (piece.as_bytes(), id)).collect());
        let mut byte_ids = [NONE; 256];
        for (byte, slot) in byte_ids.iter_mut().enumerate() {
            if let Some(id) = vocab.token_to_id(&format!("<0x{byte:02X}>")) {
                *slot = id;
            }
        }
        let median = median_token_length(vocab.ids.keys().copied());

        // The rest of the file is read only now that the trie is built:
        // allocating what is kept of it after the build's temporaries are
        // freed, rather than before, lowers the peak memory of the load.
        //
        // The other fields are deserialized as the crate deserializes them,
        // the unused ones only so that the files the crate refuses fall back.
        if let Some(raw) = json.get("version")
            && serde_json::from_str::<String>(raw.get()).ok()? != "1.0"
        {
            return None;
        }
        field::<Option<TruncationParams>>(&json, "truncation")?;
        field::<Option<PaddingParams>>(&json, "padding")?;
        field::<Option<PostProcessorWrapper>>(&json, "post_processor")?;
        field::<Option<DecoderWrapper>>(&json, "decoder")?;
        let normalizer: Option<NormalizerWrapper> = field(&json, "normalizer")?;
        let pre_tokenizer: Option<PreTokenizerWrapper> = field(&json, "pre_tokenizer")?;
        let added_tokens: Vec<AddedToken> = field::<Vec<AddedTokenJson>>(&json, "added_tokens")?
            .into_iter()
            .map(|entry| entry.token)
            .collect();

        // As `Tokenizer` deserialization does it: one `add_tokens` call with
        // every added token, in file order, against the model and normalizer.
        let mut added_vocabulary = AddedVocabulary::new();
        added_vocabulary.add_tokens(&added_tokens, &vocab, normalizer.as_ref());

        let fast = FastUnigram {
            added_vocabulary,
            normalizer,
            pre_tokenizer,
            trie,
            scores,
            unk_id,
            unk_score,
            byte_fallback,
            byte_ids,
        };
        Some((fast, median))
    }

    /// ID of the unknown token.
    pub(crate) fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Encode text into token IDs, matching `encode_batch_fast(...).get_ids()`
    /// (unknown pieces emit the unk id, which the caller filters out).
    pub(crate) fn encode_ids(&self, text: &str) -> tokenizers::Result<Vec<u32>> {
        // Added tokens, normalization and pre-tokenization exactly as
        // `encode_single_sequence` does them before the model.
        let mut pretokenized = self
            .added_vocabulary
            .extract_and_normalize(self.normalizer.as_ref(), text);
        if let Some(pre_tokenizer) = &self.pre_tokenizer {
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
            "x[PAD]y [UNK] z[UNK]",
            "a<mask>b a <mask> b a   <mask>   b<mask>",
            "a<late>b a <late> b a   <late>   b<late>",
            "band and bandana, and; (and) and_x hello world!",
            "ﬁve five FIVE ﬁ ve",
            "x<extra>y zzz zzzz <extra>",
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

    /// The `unk_id` of a Unigram tokenizer, which the crate keeps private.
    fn unk_id_of(tok: &Tokenizer) -> Option<usize> {
        match tok.get_model() {
            ModelWrapper::Unigram(model) => serde_json::to_value(model).unwrap()["unk_id"]
                .as_u64()
                .map(|id| id as usize),
            _ => panic!("not a Unigram tokenizer"),
        }
    }

    fn assert_matches_crate(tok: &Tokenizer, texts: &[String]) {
        let fast = FastUnigram::from_tokenizer(tok, unk_id_of(tok)).expect("fast path applies");
        assert_ids_match_crate(&fast, tok, texts);
    }

    /// Checks that `fast` gives the token IDs the crate's `tok` gives.
    fn assert_ids_match_crate(fast: &FastUnigram, tok: &Tokenizer, texts: &[String]) {
        for text in texts {
            let want = tok
                .encode_fast(text.as_str(), false)
                .expect("crate encode")
                .get_ids()
                .to_vec();
            let got = fast.encode_ids(text).expect("fast encode");
            assert_eq!(got, want, "token-id mismatch for {text:?}");
        }
    }

    /// Checks `from_json` against the crate's tokenizer: the added
    /// vocabulary, the unknown ID, the median token length and the token IDs
    /// of `texts`.
    fn assert_from_json_matches_crate(bytes: &[u8], texts: &[String]) {
        let full = Tokenizer::from_bytes(bytes).expect("load tokenizer");
        let (fast, median) = FastUnigram::from_json(bytes).expect("from_json applies");
        let added = full.get_added_vocabulary();
        assert_eq!(fast.added_vocabulary.get_vocab(), added.get_vocab());
        assert_eq!(
            fast.added_vocabulary.get_added_tokens_decoder(),
            added.get_added_tokens_decoder()
        );
        assert_eq!(Some(fast.unk_id() as usize), unk_id_of(&full));
        assert_eq!(
            median,
            median_token_length(full.get_vocab(false).keys().map(String::as_str))
        );
        assert_ids_match_crate(&fast, &full, texts);
    }

    /// A small Unigram tokenizer over pieces cut from `texts`, with scores
    /// drawn from a handful of values so that tied paths are common.
    fn small_unigram(texts: &[String], byte_fallback: bool) -> Tokenizer {
        Tokenizer::from_str(&small_unigram_json(texts, byte_fallback).to_string()).expect("build tokenizer")
    }

    /// `token` as an `added_tokens` entry. The crate gives an added token the
    /// ID of the piece it spells, or else the next ID past the vocabulary,
    /// whatever `id` says: `id` only feeds a warning.
    fn added_json(id: usize, token: AddedToken) -> Value {
        let mut entry = serde_json::to_value(token).unwrap();
        entry["id"] = id.into();
        entry
    }

    /// Pads `vocab` with filler pieces up to `len` pieces.
    fn pad(vocab: &mut Vec<Value>, len: usize) {
        while vocab.len() < len {
            vocab.push(json!([format!("filler{}", vocab.len()), -30.0]));
        }
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
    fn from_tokenizer_keeps_what_the_tokenizer_was_given() {
        let texts = corpus();
        let mut tok = small_unigram(&texts, false);
        tok.add_tokens(&[AddedToken::from("zzz", false)]);
        tok.add_special_tokens(&[AddedToken::from("<extra>", true)]);
        assert_matches_crate(&tok, &texts);
        // Special tokens are then split like any other text.
        tok.set_encode_special_tokens(true);
        assert_matches_crate(&tok, &texts);
    }

    #[test]
    fn from_json_matches_crate() {
        let texts = corpus();
        for byte_fallback in [false, true] {
            let json = small_unigram_json(&texts, byte_fallback);
            // "hello world" is not a piece, so it takes an ID past the vocabulary.
            let vocab = json["model"]["vocab"].as_array().unwrap();
            assert!(vocab.iter().all(|listed| listed[0] != "hello world"));
            assert_from_json_matches_crate(json.to_string().as_bytes(), &texts);
        }
    }

    #[test]
    fn from_json_matches_crate_when_a_piece_is_listed_twice() {
        let texts = corpus();
        let mut json = small_unigram_json(&texts, false);
        let vocab = json["model"]["vocab"].as_array_mut().unwrap();
        let piece = vocab[5][0].as_str().unwrap().to_string();
        vocab.insert(20, json!([piece, -0.5]));
        assert_eq!(vocab.iter().filter(|listed| listed[0] == piece.as_str()).count(), 2);
        // An added token that is the piece takes its last ID, and one that is
        // not a piece ("hello world") the ID past every listed piece.
        json["added_tokens"]
            .as_array_mut()
            .unwrap()
            .push(added_json(20, AddedToken::from(piece, true)));
        assert_from_json_matches_crate(json.to_string().as_bytes(), &texts);
    }

    #[test]
    fn from_json_matches_crate_past_the_first_pieces() {
        let texts = corpus();
        let mut json = small_unigram_json(&texts, false);
        let vocab = json["model"]["vocab"].as_array_mut().unwrap();
        pad(vocab, 5000);
        vocab.push(json!(["<late>", -30.0]));
        vocab.push(json!(["<late unk>", -30.0]));
        pad(vocab, 9000);
        // The original XLM-R layout: `<mask>` is the last piece.
        vocab.push(json!(["<mask>", 0.0]));
        json["model"]["unk_id"] = 5001.into();
        json["added_tokens"].as_array_mut().unwrap().extend([
            added_json(5000, AddedToken::from("<late>", false).rstrip(true)),
            added_json(9000, AddedToken::from("<mask>", true).lstrip(true)),
        ]);
        assert_from_json_matches_crate(json.to_string().as_bytes(), &texts);
    }

    #[test]
    fn from_json_matches_crate_on_added_token_flags() {
        let texts = corpus();
        let mut json = small_unigram_json(&texts, true);
        let n = json["model"]["vocab"].as_array().unwrap().len();
        json["added_tokens"].as_array_mut().unwrap().extend([
            // Matched in the normalized text, where NFKC turns "ﬁ" into "fi".
            added_json(n + 2, AddedToken::from("ﬁve", false)),
            added_json(n + 3, AddedToken::from("<mask>", true).lstrip(true).rstrip(true)),
            added_json(
                n + 4,
                AddedToken::from("<late>", false).normalized(false).single_word(true),
            ),
            added_json(n + 5, AddedToken::from("[PAD]", true).normalized(true)),
        ]);
        assert_from_json_matches_crate(json.to_string().as_bytes(), &texts);
    }

    #[test]
    fn from_json_matches_crate_on_unigram_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-vocab-quantized/tokenizer.json"
        );
        assert_from_json_matches_crate(&std::fs::read(path).unwrap(), &corpus());
    }

    #[test]
    fn from_json_falls_back_without_unk_id() {
        let mut json = small_unigram_json(&corpus(), false);
        json["model"]["unk_id"] = Value::Null;
        assert!(FastUnigram::from_json(json.to_string().as_bytes()).is_none());
    }

    #[test]
    fn from_json_falls_back_when_the_crate_refuses_the_file() {
        let json = small_unigram_json(&corpus(), false);
        let refused = |edit: &dyn Fn(&mut Value)| {
            let mut json = json.clone();
            edit(&mut json);
            let text = json.to_string();
            assert!(Tokenizer::from_str(&text).is_err(), "the crate loads it");
            FastUnigram::from_json(text.as_bytes()).is_none()
        };
        assert!(refused(&|j| j["version"] = "2.0".into()));
        assert!(refused(&|j| {
            let len = j["model"]["vocab"].as_array().unwrap().len();
            j["model"]["unk_id"] = len.into();
        }));
        assert!(refused(&|j| j["model"]["vocab"] = json!([])));
        assert!(refused(&|j| j["model"]["byte_fallback"] = Value::Null));
        assert!(refused(&|j| j["truncation"] = json!(42)));
        assert!(refused(&|j| j["padding"] = json!(42)));
        assert!(refused(&|j| j["normalizer"] = json!(42)));
        assert!(refused(&|j| j["pre_tokenizer"] = json!(42)));
        assert!(refused(&|j| j["post_processor"] = json!(42)));
        assert!(refused(&|j| j["decoder"] = json!(42)));
        assert!(refused(&|j| j["added_tokens"] = Value::Null));
        assert!(refused(&|j| {
            j["added_tokens"][0].as_object_mut().unwrap().remove("id");
        }));
        assert!(refused(&|j| {
            j["added_tokens"][0].as_object_mut().unwrap().remove("lstrip");
        }));
        assert!(refused(&|j| j["extra"] = json!(1)));
    }

    /// Whatever `from_json` accepts, the crate accepts, so a file the crate
    /// refuses falls back and fails with the crate's error.
    #[test]
    fn from_json_accepts_only_what_the_crate_accepts() {
        let json = small_unigram_json(&corpus(), false);
        let text = json.to_string();
        assert!(FastUnigram::from_json(text.as_bytes()).is_some());
        let mut cases = vec![
            ("top level in an array".to_string(), format!("[{text}]")),
            ("unknown key".to_string(), text.replacen('{', r#"{"extra":1,"#, 1)),
            (
                "repeated key".to_string(),
                text.replacen('{', r#"{"normalizer":42,"#, 1),
            ),
            (
                "repeated model key".to_string(),
                text.replacen(r#""model":{"#, r#""model":{"vocab":42,"#, 1),
            ),
        ];
        let mut edited = |case: String, edit: &dyn Fn(&mut Value)| {
            let mut json = json.clone();
            edit(&mut json);
            cases.push((case, json.to_string()));
        };
        edited("version 2.0".to_string(), &|j| j["version"] = "2.0".into());
        edited("added token without id".to_string(), &|j| {
            j["added_tokens"][0].as_object_mut().unwrap().remove("id");
        });
        for key in json.as_object().unwrap().keys() {
            edited(format!("no {key}"), &|j| {
                j.as_object_mut().unwrap().remove(key);
            });
            edited(format!("{key}: null"), &|j| j[key] = Value::Null);
            edited(format!("{key}: 42"), &|j| j[key] = json!(42));
        }
        for key in json["model"].as_object().unwrap().keys() {
            edited(format!("no model.{key}"), &|j| {
                j["model"].as_object_mut().unwrap().remove(key);
            });
            edited(format!("model.{key}: null"), &|j| j["model"][key] = Value::Null);
            edited(format!("model.{key}: 42"), &|j| j["model"][key] = json!(42));
        }
        let diverging: Vec<&str> = cases
            .iter()
            .filter(|(_, text)| FastUnigram::from_json(text.as_bytes()).is_some() && Tokenizer::from_str(text).is_err())
            .map(|(case, _)| case.as_str())
            .collect();
        assert!(diverging.is_empty(), "the crate refuses {diverging:?}");
    }

    #[test]
    fn from_json_falls_back_for_other_models() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-int8/tokenizer.json"
        );
        assert!(FastUnigram::from_json(&std::fs::read(path).unwrap()).is_none());
    }

    /// `add_tokens` asks `JsonVocab` only for `token_to_id` and
    /// `get_vocab_size`, but every lookup answers as the crate's `Unigram`.
    #[test]
    fn json_vocab_answers_as_the_crate_model() {
        let mut json = small_unigram_json(&corpus(), true);
        let vocab = json["model"]["vocab"].as_array_mut().unwrap();
        vocab.insert(20, vocab[5].clone());
        let tok = Tokenizer::from_str(&json.to_string()).expect("build tokenizer");
        let ModelWrapper::Unigram(model) = tok.get_model() else {
            panic!("not a Unigram tokenizer");
        };
        let pieces: Vec<(String, f64)> = serde_json::from_value(json["model"]["vocab"].clone()).unwrap();
        let vocab = JsonVocab::new(&pieces).unwrap();
        assert!(vocab.ids.len() < pieces.len(), "no piece is listed twice");
        assert_eq!(vocab.get_vocab(), model.get_vocab());
        assert_eq!(vocab.get_vocab_size(), model.get_vocab_size());
        for id in (0..=pieces.len() as u32).chain([u32::MAX]) {
            assert_eq!(vocab.id_to_token(id), model.id_to_token(id), "id {id}");
        }
        for piece in vocab.ids.keys().copied().chain(["not a piece"]) {
            assert_eq!(vocab.token_to_id(piece), model.token_to_id(piece), "{piece:?}");
        }
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
