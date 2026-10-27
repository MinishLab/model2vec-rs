//! A purpose-built, IDs-only WordPiece tokenizer.
//!
//! `encode_batch_fast` builds full `Encoding` objects (offsets, alignment,
//! masks, type IDs) of which the static-embedding path uses only `get_ids()`.
//! This module implements just the forward text -> IDs mapping for the
//! BertNormalizer + BertPreTokenizer + WordPiece pipeline shared by every
//! non-multilingual POTION model, producing token IDs identical to the
//! `tokenizers` crate (verified by the parity test below).
//!
//! Literal added-token text (e.g. `[MASK]`, `[CLS]`) is matched and resolved
//! to its id before normalization, matching the crate's `AddedVocabulary`
//! step. Pipelines this module does not reproduce exactly (e.g. the
//! SentencePiece-Unigram multilingual model, or added tokens with
//! `normalized`/`single_word`/`lstrip`/`rstrip` set) return `None` from
//! [`FastWordPiece::from_tokenizer`] so the caller falls back to the crate.

use std::cell::RefCell;
use std::collections::HashMap;

use serde::Deserialize;
use serde_json::value::RawValue;
use tokenizers::{
    Tokenizer, models::ModelWrapper, normalizers::NormalizerWrapper, pre_tokenizers::PreTokenizerWrapper,
};
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

use crate::model::median_token_length;
use crate::trie::{DoubleArrayTrie, NONE};

thread_local! {
    static NORMALIZED: RefCell<String> = const { RefCell::new(String::new()) };
}

/// The parts of `tokenizer.json` that [`FastWordPiece::from_json`] reads.
#[derive(Deserialize)]
struct TokenizerJson<'a> {
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    normalizer: Option<NormalizerJson>,
    pre_tokenizer: Option<PreTokenizerJson>,
    #[serde(borrow)]
    model: ModelJson<'a>,
}

#[derive(Deserialize)]
struct AddedTokenJson {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
}

#[derive(Deserialize)]
struct NormalizerJson {
    #[serde(rename = "type")]
    kind: String,
    clean_text: Option<bool>,
    handle_chinese_chars: Option<bool>,
    strip_accents: Option<bool>,
    lowercase: Option<bool>,
}

#[derive(Deserialize)]
struct PreTokenizerJson {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct ModelJson<'a> {
    #[serde(rename = "type")]
    kind: Option<String>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    max_input_chars_per_word: Option<usize>,
    /// Parsed only once the pipeline is known to be WordPiece.
    #[serde(borrow)]
    vocab: Option<&'a RawValue>,
}

/// IDs-only WordPiece tokenizer for the BERT pipeline.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FastWordPiece {
    vocab: DoubleArrayTrie,
    /// Trie node reached by the continuing-subword prefix, or `NONE`.
    continuation: u32,
    /// Vocab IDs of single-ASCII-byte tokens (-1 when absent), so isolated
    /// punctuation words skip the trie walk.
    single_byte: [i32; 128],
    unk_id: Option<u32>,
    max_input_chars_per_word: usize,
    // BertNormalizer settings
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: bool,
    lowercase: bool,
    /// Added tokens matched as literal substrings of the raw text before
    /// normalization, longest content first so a shorter token cannot shadow
    /// a longer one with the same prefix.
    added_tokens: Vec<(String, u32)>,
    /// First byte of every added token, so the scan skips byte positions
    /// that cannot start a match with one array lookup.
    added_token_first_bytes: [bool; 256],
}

impl FastWordPiece {
    /// Build from a loaded tokenizer, or `None` if it is not the
    /// BertNormalizer + BertPreTokenizer + WordPiece pipeline.
    pub(crate) fn from_tokenizer(tok: &Tokenizer) -> Option<FastWordPiece> {
        // Require the exact pipeline; anything else falls back to the crate.
        let ModelWrapper::WordPiece(model) = tok.get_model() else {
            return None;
        };
        if !matches!(tok.get_pre_tokenizer(), Some(PreTokenizerWrapper::BertPreTokenizer(_))) {
            return None;
        }
        let Some(NormalizerWrapper::BertNormalizer(norm)) = tok.get_normalizer() else {
            return None;
        };

        let vocab = tok.get_vocab(true);
        let unk_id = vocab.get(&model.unk_token).copied();

        // The crate matches added tokens with `normalized: false` against raw
        // text; the other flags change matching rules not implemented here.
        let mut added_tokens = Vec::new();
        for (id, t) in tok.get_added_tokens_decoder() {
            if t.content.is_empty() || t.normalized || t.single_word || t.lstrip || t.rstrip {
                return None;
            }
            added_tokens.push((t.content, id));
        }

        Some(Self::assemble(
            &vocab,
            &model.continuing_subword_prefix,
            unk_id,
            model.max_input_chars_per_word,
            norm.clean_text,
            norm.handle_chinese_chars,
            // `strip_accents: None` follows the lowercase setting, matching HuggingFace.
            norm.strip_accents.unwrap_or(norm.lowercase),
            norm.lowercase,
            added_tokens,
        ))
    }

    /// Build straight from `tokenizer.json` bytes without constructing a
    /// [`Tokenizer`]. Returns the tokenizer and the median token length, or
    /// `None` for another pipeline, a missing field, an unk token outside the
    /// vocab, or added tokens that are duplicated, absent from the vocab, or
    /// use matching flags not implemented here. Only the fields read here are
    /// validated, so JSON the crate would reject elsewhere may still load.
    pub(crate) fn from_json(bytes: &[u8]) -> Option<(FastWordPiece, usize)> {
        let json: TokenizerJson = serde_json::from_slice(bytes).ok()?;
        let model = json.model;
        if model.kind.as_deref() != Some("WordPiece") || json.pre_tokenizer?.kind != "BertPreTokenizer" {
            return None;
        }
        let norm = json.normalizer?;
        if norm.kind != "BertNormalizer" {
            return None;
        }
        let lowercase = norm.lowercase?;

        let vocab: HashMap<String, u32> = serde_json::from_str(model.vocab?.get()).ok()?;
        let unk_id = *vocab.get(&model.unk_token?)?;

        // The crate gives an added token that is already in the vocab its
        // vocab id, so with every added token in the vocab the vocab with
        // added tokens equals the model vocab.
        let mut added_tokens: Vec<(String, u32)> = Vec::with_capacity(json.added_tokens.len());
        for t in json.added_tokens {
            if t.content.is_empty() || t.normalized || t.single_word || t.lstrip || t.rstrip {
                return None;
            }
            if added_tokens.iter().any(|(content, _)| *content == t.content) {
                return None;
            }
            let id = *vocab.get(&t.content)?;
            added_tokens.push((t.content, id));
        }

        let median = median_token_length(vocab.keys().map(String::as_str));
        let ft = Self::assemble(
            &vocab,
            &model.continuing_subword_prefix?,
            Some(unk_id),
            model.max_input_chars_per_word?,
            norm.clean_text?,
            norm.handle_chinese_chars?,
            norm.strip_accents.unwrap_or(lowercase),
            lowercase,
            added_tokens,
        );
        Some((ft, median))
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        vocab: &HashMap<String, u32>,
        continuing_subword_prefix: &str,
        unk_id: Option<u32>,
        max_input_chars_per_word: usize,
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: bool,
        lowercase: bool,
        mut added_tokens: Vec<(String, u32)>,
    ) -> FastWordPiece {
        added_tokens.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        let mut added_token_first_bytes = [false; 256];
        for (content, _) in &added_tokens {
            added_token_first_bytes[content.as_bytes()[0] as usize] = true;
        }

        let mut single_byte = [-1_i32; 128];
        for (b, slot) in single_byte.iter_mut().enumerate() {
            if let Some(&id) = vocab.get(&(b as u8 as char).to_string()) {
                *slot = id as i32;
            }
        }

        let vocab = DoubleArrayTrie::build(vocab.iter().map(|(k, &id)| (k.as_bytes(), id)).collect());
        let continuation = vocab.walk(continuing_subword_prefix.as_bytes());

        FastWordPiece {
            vocab,
            continuation,
            single_byte,
            unk_id,
            max_input_chars_per_word,
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
            added_tokens,
            added_token_first_bytes,
        }
    }

    pub(crate) fn unk_id(&self) -> Option<u32> {
        self.unk_id
    }

    /// Encode text into token IDs, matching `encode_batch_fast(...).get_ids()`
    /// (unknown words emit the unk id, which the caller filters out).
    #[cfg(test)]
    pub(crate) fn encode_ids(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_ids_into(text, &mut out);
        out
    }

    /// Like [`Self::encode_ids`], but appends to a caller-owned buffer.
    pub(crate) fn encode_ids_into(&self, text: &str, out: &mut Vec<u32>) {
        NORMALIZED.with_borrow_mut(|scratch| self.encode_ids_with(text, out, scratch));
    }

    fn encode_ids_with(&self, text: &str, out: &mut Vec<u32>, scratch: &mut String) {
        // Split on added tokens (leftmost-longest), then run the BERT
        // pipeline on each span of ordinary text between them.
        let bytes = text.as_bytes();
        let mut start = 0;
        let mut i = 0;
        while i < text.len() {
            let hit = self.added_token_first_bytes[bytes[i] as usize]
                .then(|| {
                    self.added_tokens
                        .iter()
                        .find(|(content, _)| text[i..].starts_with(content.as_str()))
                })
                .flatten();
            match hit {
                Some((content, id)) => {
                    self.encode_span(&text[start..i], out, scratch);
                    out.push(*id);
                    i += content.len();
                    start = i;
                }
                // Added tokens are valid UTF-8, so a match can only start on
                // a char boundary; step a whole char to keep slicing valid.
                None => i += text[i..].chars().next().map_or(1, char::len_utf8),
            }
        }
        self.encode_span(&text[start..], out, scratch);
    }

    /// Normalize, pre-tokenize and WordPiece a span with no added tokens.
    fn encode_span(&self, span: &str, out: &mut Vec<u32>, normalized: &mut String) {
        if span.is_empty() {
            return;
        }
        normalized.clear();
        self.normalize_into(span, normalized);
        pre_tokenize(normalized, |word| self.word_to_ids(word, out));
    }

    // --- WordPiece ---------------------------------------------------------

    fn push_unk(&self, out: &mut Vec<u32>) {
        if let Some(u) = self.unk_id {
            out.push(u);
        }
    }

    /// Greedy longest-match-first. A word that cannot be fully tokenized (or
    /// exceeds `max_input_chars_per_word`) maps to a single unk token.
    fn word_to_ids(&self, word: &str, out: &mut Vec<u32>) {
        // Rune count never exceeds byte length, so short words skip the scan.
        if word.len() > self.max_input_chars_per_word && word.chars().count() > self.max_input_chars_per_word {
            self.push_unk(out);
            return;
        }

        let bytes = word.as_bytes();
        if bytes.len() == 1 {
            let id = self.single_byte[bytes[0] as usize];
            if id >= 0 {
                out.push(id as u32);
            } else {
                self.push_unk(out);
            }
            return;
        }

        // Vocab entries are valid UTF-8, so every match ends on a char boundary.
        let mark = out.len();
        let mut start = 0;
        while start < bytes.len() {
            let node = if start == 0 { 0 } else { self.continuation };
            let found = if node == NONE {
                None
            } else {
                self.vocab.longest_prefix(node, &bytes[start..])
            };
            match found {
                Some((id, len)) => {
                    out.push(id);
                    start += len;
                }
                None => {
                    out.truncate(mark);
                    self.push_unk(out);
                    return;
                }
            }
        }
    }

    // --- BertNormalizer ----------------------------------------------------

    /// Append the normalized `input` to `out`. Edge whitespace is kept, since
    /// pre-tokenization drops it anyway.
    fn normalize_into(&self, input: &str, out: &mut String) {
        if input.is_ascii() {
            self.normalize_ascii(input, out)
        } else {
            self.normalize_unicode(input, out)
        }
    }

    /// Fused clean + lowercase over ASCII bytes. Accent stripping (NFD is
    /// identity below 0x80) and Chinese handling are no-ops for ASCII.
    fn normalize_ascii(&self, input: &str, out: &mut String) {
        out.reserve(input.len());
        for &orig in input.as_bytes() {
            let mut b = orig;
            if self.clean_text {
                match b {
                    b'\t' | b'\n' | b'\r' => b = b' ',
                    0..=0x1f | 0x7f => continue,
                    _ => {}
                }
            }
            if self.lowercase && b.is_ascii_uppercase() {
                b += b'a' - b'A';
            }
            out.push(b as char);
        }
    }

    fn normalize_unicode(&self, input: &str, out: &mut String) {
        // Fused clean + Chinese-char pass.
        let mut chars: Vec<char> = Vec::with_capacity(input.len());
        for c in input.chars() {
            if self.clean_text {
                if c == '\0' || c == '\u{fffd}' || is_control(c) {
                    continue;
                }
                if is_whitespace(c) {
                    chars.push(' ');
                    continue;
                }
            }
            if self.handle_chinese_chars && is_chinese_char(c) {
                chars.push(' ');
                chars.push(c);
                chars.push(' ');
                continue;
            }
            chars.push(c);
        }
        let staged: String = chars.into_iter().collect();

        let processed: String = if self.strip_accents {
            // NFD, drop nonspacing marks, then (optionally) lowercase; left
            // decomposed, matching HuggingFace.
            let mut out = String::with_capacity(staged.len());
            for c in staged.nfd() {
                if c.is_mark_nonspacing() {
                    continue;
                }
                if self.lowercase {
                    out.extend(c.to_lowercase());
                } else {
                    out.push(c);
                }
            }
            out
        } else if self.lowercase {
            staged.chars().flat_map(|c| c.to_lowercase()).collect()
        } else {
            staged
        };

        out.push_str(&processed);
    }
}

// --- BertPreTokenizer -------------------------------------------------------

/// ASCII byte class: 0 = word, 1 = space, 2 = punctuation. Agrees with the
/// rune-level predicates on every value below 0x80.
fn ascii_class(b: u8) -> u8 {
    match b {
        b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r' => 1,
        b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~' => 2,
        _ => 0,
    }
}

/// Split like HuggingFace's BertPreTokenizer: drop whitespace, isolate each
/// punctuation char as its own word, and call `emit` on each word in order.
fn pre_tokenize<'a>(s: &'a str, mut emit: impl FnMut(&'a str)) {
    let bytes = s.as_bytes();
    let mut start: isize = -1;
    let mut i = 0;
    while i < s.len() {
        let (class, size) = if bytes[i] < 0x80 {
            (ascii_class(bytes[i]), 1)
        } else {
            let c = s[i..].chars().next().unwrap();
            let class = if c.is_whitespace() {
                1
            } else if c.is_ascii_punctuation() || c.is_punctuation() {
                2
            } else {
                0
            };
            (class, c.len_utf8())
        };
        match class {
            1 => {
                if start >= 0 {
                    emit(&s[start as usize..i]);
                    start = -1;
                }
            }
            2 => {
                if start >= 0 {
                    emit(&s[start as usize..i]);
                    start = -1;
                }
                emit(&s[i..i + size]);
            }
            _ => {
                if start < 0 {
                    start = i as isize;
                }
            }
        }
        i += size;
    }
    if start >= 0 {
        emit(&s[start as usize..]);
    }
}

// --- Unicode predicates (match HuggingFace's bert normalizer) ----------------

fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

fn is_control(c: char) -> bool {
    match c {
        '\t' | '\n' | '\r' => false,
        _ => c.is_other(),
    }
}

fn is_chinese_char(c: char) -> bool {
    let r = c as u32;
    (0x4E00..=0x9FFF).contains(&r)
        || (0x3400..=0x4DBF).contains(&r)
        || (0x20000..=0x2A6DF).contains(&r)
        || (0x2A700..=0x2B73F).contains(&r)
        || (0x2B740..=0x2B81F).contains(&r)
        || (0x2B920..=0x2CEAF).contains(&r)
        || (0xF900..=0xFAFF).contains(&r)
        || (0x2F800..=0x2FA1F).contains(&r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn corpus() -> Vec<&'static str> {
        vec![
            "Hello, world!",
            "The QUICK brown fox jumps.",
            "café résumé naïve fiancé",
            "Ünïçödé ACCENTS Straße",
            "你好世界 mixed 测试 text",
            "a  b\tc\n  d",
            "supercalifragilisticexpialidocious antidisestablishmentarianism",
            "email@example.com https://x.co/y?z=1 $5.00 <html> C++ #hashtag",
            "123 456.78 -90 1e10",
            "",
            "   ",
            "MiXeD CaSe WoRdS 42",
            // --- clean_text: control/format chars removed, whitespace→space ---
            "a\u{0007}b\u{0000}c",      // bell + NUL removed
            "a\u{000b}b\u{000c}c",      // vertical tab / form feed (Cc → removed, NOT spaced)
            "a\u{007f}b",               // DEL removed
            "a\u{200b}b\u{feff}c",      // zero-width space + BOM (Cf → removed)
            "a\u{00a0}b\u{2009}c",      // NBSP + thin space (Zs → become space → split)
            "a\u{fffd}b",               // replacement char removed
            "\u{0001}\u{0002}\u{0003}", // all-control → empty after normalize
            "  leading and trailing  ", // edge whitespace trim
            "\t\n\r",                   // whitespace/control only
            // --- casing that changes length or is context-sensitive ---
            "İstanbul",                  // U+0130: NFD→I+dot, strip dot, lower→i
            "STRASSE straße GROSS",      // eszett upper/lower asymmetry
            "ﬁle ﬀ ﬄ",                   // ligatures (fi, ff, ffl)
            "ΟΔΟΣ Σίσυφος",              // Greek final-sigma: verify char-level (no context)
            "ＦＵＬＬＷＩＤＴＨ ａｂｃ", // fullwidth latin casing
            // --- accents / combining marks, both compositions ---
            "e\u{0301} a\u{0300} n\u{0303}", // pre-decomposed base+combining
            "é à ñ ü ç",                     // precomposed (NFD path)
            "Å Ω µ",                         // chars with compatibility/canonical quirks
            "क्षत्रि",                         // Devanagari combining (non-Latin marks)
            // --- CJK / handle_chinese_chars boundaries and non-CJK scripts ---
            "一鿿㐀䶿",                       // BMP CJK range endpoints
            "𠀀 𪛖 豈",                       // ext-B (U+20000) + compatibility (U+F900)
            "你好world世界",                  // CJK adjacent to latin, no spaces
            "，。、！？；：",                 // fullwidth/CJK punctuation (category Po → isolate)
            "こんにちは カタカナ 안녕하세요", // kana + hangul: must NOT be space-wrapped
            // --- punctuation isolation: ASCII vs Unicode P vs Unicode symbol ---
            "a≤b x→y n≠m", // Sm math symbols: NOT punctuation → stay in word
            "«guillemets» \u{201c}smart\u{201d} \u{2018}quotes\u{2019}", // Pi/Pf → isolate
            "em—dash en–dash …ellipsis", // Pd + Po
            "e.g., i.e. U.S.A. 3.14 1,000", // dotted abbreviations / decimals
            "!!!??? ((())) [a]{b}", // punctuation runs / adjacency
            "$100 €50 £20 ¥30 ₹40 ₿1", // currency symbols (Sc)
            // --- emoji / symbols / combining sequences ---
            "hello 👋 world 🌍",
            "family 👨‍👩‍👧 ZWJ",      // ZWJ sequence (ZWJ is Cf → removed)
            "🇺🇸 🇯🇵 flags",        // regional indicators
            "thumbs 👍🏽 skintone", // emoji + skin-tone modifier
            // --- WordPiece internals: subwords, unk, max_input_chars ---
            "internationalization preprocessing tokenization playing", // multi-## splits
            "ᚠᚢᚦ 𐐀𐐁 zzqxjk",                                           // runic/deseret/rare → unk path
            "🎉party mixed🔥emoji",                                    // unk char glued to word
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", // 100 chars: at boundary
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", // 101 chars: exceeds → unk
            "字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字", // long multibyte: byte-len ≫ char-len (exercises the byte shortcut)
            // --- literal special/added tokens ---
            "hello [MASK] world",
            "hello [mask] world",
            "[CLS] hi [SEP]",
            "[PAD][UNK][MASK]",
            "no special tokens here",
            "[MASK] [CLS] [SEP] [PAD] [UNK]",
            "foo[SEP]bar text[MASK]here",
            "[unk] lowercase specials", // normalized-form collision check
        ]
    }

    #[test]
    fn fast_matches_crate() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-int8/tokenizer.json"
        );
        let tok = Tokenizer::from_file(path).expect("load fixture tokenizer");
        let ft = FastWordPiece::from_tokenizer(&tok).expect("fixture is a WordPiece pipeline");

        for text in corpus() {
            let want: Vec<u32> = tok.encode_fast(text, false).expect("crate encode").get_ids().to_vec();
            let got = ft.encode_ids(text);
            assert_eq!(got, want, "token-id mismatch for {text:?}");
        }
    }

    /// Random lowercase words, so long subword splits and unknown words occur.
    #[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
    fn noise(n: usize) -> Vec<String> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        (0..n)
            .map(|_| {
                (0..1 + next(12))
                    .map(|_| {
                        (0..1 + next(14))
                            .map(|_| (b'a' + next(26) as u8) as char)
                            .collect::<String>()
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    fn assert_matches_crate(tok: &Tokenizer, texts: &[String]) {
        let ft = FastWordPiece::from_tokenizer(tok).expect("WordPiece pipeline");
        for text in texts {
            let want: Vec<u32> = tok
                .encode_fast(text.as_str(), false)
                .expect("crate encode")
                .get_ids()
                .to_vec();
            assert_eq!(ft.encode_ids(text), want, "token-id mismatch for {text:?}");
        }
    }

    #[test]
    fn duplicate_vocab_key_matches_crate() {
        let json = r###"{
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                           "strip_accents": null, "lowercase": true},
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                      "max_input_chars_per_word": 100,
                      "vocab": {"[UNK]": 0, "play": 1, "##ing": 2, "play": 3, "##s": 4, "##ing": 5, "p": 6}}
        }"###;
        let tok = Tokenizer::from_str(json).expect("build tokenizer");
        let texts: Vec<String> = ["play", "playing plays", "pplay", "ping"].map(String::from).into();
        assert_matches_crate(&tok, &texts);
        assert_json_matches_tokenizer(json.as_bytes());
    }

    /// WordPiece tokenizers of models already in the local Hugging Face
    /// cache (the test does not download them).
    #[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
    #[test]
    fn fast_matches_crate_on_cached_models() {
        let mut texts: Vec<String> = corpus().into_iter().map(String::from).collect();
        texts.extend(noise(2000));
        let cache = hf_hub::Cache::from_env();
        for (repo, file) in [
            ("minishlab/potion-base-8M", "tokenizer.json"),
            ("minishlab/potion-retrieval-32M", "tokenizer.json"),
            ("hotchpotch/bekko-static-embedding-v1", "tokenizer.json"),
            (
                "sentence-transformers/static-retrieval-mrl-en-v1",
                "0_StaticEmbedding/tokenizer.json",
            ),
        ] {
            let Some(path) = cache.model(repo.to_string()).get(file) else {
                eprintln!("skipped: {repo} is not in the local cache");
                continue;
            };
            let tok = Tokenizer::from_file(&path).expect("load cached tokenizer");
            assert_matches_crate(&tok, &texts);
            assert_json_matches_tokenizer(&std::fs::read(&path).expect("read cached tokenizer"));
        }
    }

    /// `from_json` must build exactly what `from_tokenizer` builds from the
    /// crate's tokenizer, or decline.
    fn assert_json_matches_tokenizer(bytes: &[u8]) {
        let tok = Tokenizer::from_bytes(bytes).expect("load tokenizer");
        let (ft, median) = FastWordPiece::from_json(bytes).expect("WordPiece pipeline");
        assert_eq!(Some(&ft), FastWordPiece::from_tokenizer(&tok).as_ref());
        let vocab = tok.get_vocab(false);
        assert_eq!(median, median_token_length(vocab.keys().map(String::as_str)));
    }

    #[test]
    fn from_json_matches_from_tokenizer() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-int8/tokenizer.json"
        );
        assert_json_matches_tokenizer(&std::fs::read(path).expect("read fixture tokenizer"));
    }

    #[test]
    fn from_json_declines_what_it_cannot_reproduce() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-int8/tokenizer.json"
        );
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let declines = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut json = json.clone();
            edit(&mut json);
            FastWordPiece::from_json(json.to_string().as_bytes()).is_none()
        };
        assert!(!declines(&|_| {}));
        assert!(declines(&|j| j["added_tokens"][0]["normalized"] = true.into()));
        assert!(declines(&|j| j["added_tokens"][0]["content"] = "[NOT IN VOCAB]".into()));
        assert!(declines(&|j| {
            let first = j["added_tokens"][0].clone();
            j["added_tokens"].as_array_mut().unwrap().push(first);
        }));
        assert!(declines(&|j| j["model"]["unk_token"] = "[NOT IN VOCAB]".into()));
        assert!(declines(&|j| j["normalizer"]["type"] = "Lowercase".into()));
        assert!(declines(&|j| j["pre_tokenizer"] = serde_json::Value::Null));
        assert!(declines(&|j| {
            j["model"].as_object_mut().unwrap().remove("max_input_chars_per_word");
        }));
    }

    #[test]
    fn normalized_added_token_falls_back() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-int8/tokenizer.json"
        );
        let mut tok = Tokenizer::from_file(path).expect("load fixture tokenizer");
        tok.add_tokens(&[tokenizers::AddedToken::from("hello world", false)]);
        assert!(FastWordPiece::from_tokenizer(&tok).is_none());
        assert!(FastWordPiece::from_json(tok.to_string(false).unwrap().as_bytes()).is_none());
    }
}
