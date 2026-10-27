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

use std::collections::HashMap;

use tokenizers::{
    Tokenizer, models::ModelWrapper, normalizers::NormalizerWrapper, pre_tokenizers::PreTokenizerWrapper,
};
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

/// Magic and format version of the binary cache; bump the version whenever
/// the layout or the meaning of any field changes.
const CACHE_MAGIC: &[u8; 8] = b"M2VWP\x00\x00\x01";

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, tail) = self.0.split_at_checked(n)?;
        self.0 = tail;
        Some(head)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn str(&mut self) -> Option<&'a str> {
        let n = self.u32()? as usize;
        std::str::from_utf8(self.take(n)?).ok()
    }
}

/// IDs-only WordPiece tokenizer for the BERT pipeline.
#[derive(Debug, Clone)]
pub(crate) struct FastWordPiece {
    vocab: HashMap<String, u32>,
    /// Vocab IDs of single-ASCII-byte tokens (-1 when absent), so isolated
    /// punctuation words skip the map lookup.
    single_byte: [i32; 128],
    unk_id: Option<u32>,
    max_input_chars_per_word: usize,
    continuing_subword_prefix: String,
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
            vocab,
            unk_id,
            model.max_input_chars_per_word,
            model.continuing_subword_prefix.clone(),
            norm.clean_text,
            norm.handle_chinese_chars,
            // `strip_accents: None` follows the lowercase setting, matching HuggingFace.
            norm.strip_accents.unwrap_or(norm.lowercase),
            norm.lowercase,
            added_tokens,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        vocab: HashMap<String, u32>,
        unk_id: Option<u32>,
        max_input_chars_per_word: usize,
        continuing_subword_prefix: String,
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

        FastWordPiece {
            vocab,
            single_byte,
            unk_id,
            max_input_chars_per_word,
            continuing_subword_prefix,
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
            added_tokens,
            added_token_first_bytes,
        }
    }

    /// Serialize to the binary cache format, tagged with the checksum of the
    /// `tokenizer.json` it was built from and the model's median token length.
    pub(crate) fn to_cache_bytes(&self, checksum: u64, median_token_length: usize) -> Vec<u8> {
        let mut vocab: Vec<(&str, u32)> = self.vocab.iter().map(|(k, &v)| (k.as_str(), v)).collect();
        vocab.sort_unstable_by_key(|&(_, id)| id);
        let strings_len: usize = vocab.iter().map(|(k, _)| k.len() + 8).sum();

        let mut out = Vec::with_capacity(64 + strings_len);
        out.extend_from_slice(CACHE_MAGIC);
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&(median_token_length as u64).to_le_bytes());
        out.extend_from_slice(&self.unk_id.unwrap_or(u32::MAX).to_le_bytes());
        out.extend_from_slice(&(self.max_input_chars_per_word as u64).to_le_bytes());
        let flags = self.clean_text as u8
            | (self.handle_chinese_chars as u8) << 1
            | (self.strip_accents as u8) << 2
            | (self.lowercase as u8) << 3;
        out.push(flags);
        put_str(&mut out, &self.continuing_subword_prefix);
        for list in [vocab, self.added_tokens.iter().map(|(k, v)| (k.as_str(), *v)).collect()] {
            out.extend_from_slice(&(list.len() as u32).to_le_bytes());
            for (token, id) in list {
                out.extend_from_slice(&id.to_le_bytes());
                put_str(&mut out, token);
            }
        }
        out
    }

    /// Inverse of [`Self::to_cache_bytes`]. Returns the tokenizer and the
    /// median token length, or `None` if the bytes are malformed, from
    /// another format version, or tagged with a different checksum.
    pub(crate) fn from_cache_bytes(bytes: &[u8], checksum: u64) -> Option<(FastWordPiece, usize)> {
        let mut r = Reader(bytes);
        if r.take(CACHE_MAGIC.len())? != CACHE_MAGIC || r.u64()? != checksum {
            return None;
        }
        let median_token_length = r.u64()? as usize;
        let unk_id = Some(r.u32()?).filter(|&id| id != u32::MAX);
        let max_input_chars_per_word = r.u64()? as usize;
        let flags = r.take(1)?[0];
        let continuing_subword_prefix = r.str()?.to_string();

        let n = r.u32()? as usize;
        let mut vocab = HashMap::with_capacity(n);
        for _ in 0..n {
            let id = r.u32()?;
            vocab.insert(r.str()?.to_string(), id);
        }
        let n = r.u32()? as usize;
        let mut added_tokens = Vec::with_capacity(n);
        for _ in 0..n {
            let id = r.u32()?;
            let content = r.str()?;
            if content.is_empty() {
                return None;
            }
            added_tokens.push((content.to_string(), id));
        }
        if !r.0.is_empty() {
            return None;
        }

        let ft = Self::assemble(
            vocab,
            unk_id,
            max_input_chars_per_word,
            continuing_subword_prefix,
            flags & 1 != 0,
            flags & 2 != 0,
            flags & 4 != 0,
            flags & 8 != 0,
            added_tokens,
        );
        Some((ft, median_token_length))
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
        let mut scratch = String::new();
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
                    self.encode_span(&text[start..i], out, &mut scratch);
                    out.push(*id);
                    i += content.len();
                    start = i;
                }
                // Added tokens are valid UTF-8, so a match can only start on
                // a char boundary; step a whole char to keep slicing valid.
                None => i += text[i..].chars().next().map_or(1, char::len_utf8),
            }
        }
        self.encode_span(&text[start..], out, &mut scratch);
    }

    /// Normalize, pre-tokenize and WordPiece a span with no added tokens.
    fn encode_span(&self, span: &str, out: &mut Vec<u32>, scratch: &mut String) {
        if span.is_empty() {
            return;
        }
        let normalized = self.normalize(span);
        for word in pre_tokenize(&normalized) {
            self.word_to_ids(word, out, scratch);
        }
    }

    // --- WordPiece ---------------------------------------------------------

    fn push_unk(&self, out: &mut Vec<u32>) {
        if let Some(u) = self.unk_id {
            out.push(u);
        }
    }

    /// Greedy longest-match-first. A word that cannot be fully tokenized (or
    /// exceeds `max_input_chars_per_word`) maps to a single unk token.
    fn word_to_ids(&self, word: &str, out: &mut Vec<u32>, scratch: &mut String) {
        // Rune count never exceeds byte length, so short words skip the scan.
        if word.len() > self.max_input_chars_per_word && word.chars().count() > self.max_input_chars_per_word {
            self.push_unk(out);
            return;
        }

        if word.len() == 1 {
            let b = word.as_bytes()[0];
            let id = self.single_byte[b as usize];
            if id >= 0 {
                out.push(id as u32);
            } else {
                self.push_unk(out);
            }
            return;
        }

        let mark = out.len();
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while end > start {
                let sub = &word[start..end];
                let id = if start == 0 {
                    self.vocab.get(sub).copied()
                } else {
                    scratch.clear();
                    scratch.push_str(&self.continuing_subword_prefix);
                    scratch.push_str(sub);
                    self.vocab.get(scratch.as_str()).copied()
                };
                if let Some(id) = id {
                    found = Some(id);
                    break;
                }
                // Step back one char (start/end stay on char boundaries).
                end -= sub.chars().next_back().unwrap().len_utf8();
            }
            match found {
                Some(id) => {
                    out.push(id);
                    start = end;
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

    fn normalize(&self, input: &str) -> String {
        if input.is_empty() {
            return String::new();
        }
        if input.is_ascii() {
            self.normalize_ascii(input)
        } else {
            self.normalize_unicode(input)
        }
    }

    /// Fused clean + lowercase + trim over ASCII bytes. Accent stripping (NFD
    /// is identity below 0x80) and Chinese handling are no-ops for ASCII.
    fn normalize_ascii(&self, input: &str) -> String {
        let mut buf = Vec::with_capacity(input.len());
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
            buf.push(b);
        }
        let mut start = 0;
        let mut end = buf.len();
        while start < end && is_ascii_space(buf[start]) {
            start += 1;
        }
        while end > start && is_ascii_space(buf[end - 1]) {
            end -= 1;
        }
        // Bytes stayed ASCII throughout, so this is valid UTF-8.
        String::from_utf8(buf[start..end].to_vec()).unwrap()
    }

    fn normalize_unicode(&self, input: &str) -> String {
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

        processed.trim_matches(|c: char| c.is_whitespace()).to_string()
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
/// punctuation char as its own word. Returned words are zero-copy substrings.
fn pre_tokenize(s: &str) -> Vec<&str> {
    let bytes = s.as_bytes();
    let mut words = Vec::with_capacity(s.len() / 4 + 1);
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
                    words.push(&s[start as usize..i]);
                    start = -1;
                }
            }
            2 => {
                if start >= 0 {
                    words.push(&s[start as usize..i]);
                    start = -1;
                }
                words.push(&s[i..i + size]);
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
        words.push(&s[start as usize..]);
    }
    words
}

// --- Unicode predicates (match HuggingFace's bert normalizer) ----------------

fn is_ascii_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

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

    #[test]
    fn cache_roundtrip_matches() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-int8/tokenizer.json"
        );
        let tok = Tokenizer::from_file(path).expect("load fixture tokenizer");
        let ft = FastWordPiece::from_tokenizer(&tok).expect("fixture is a WordPiece pipeline");
        let bytes = ft.to_cache_bytes(42, 7);

        let (cached, median) = FastWordPiece::from_cache_bytes(&bytes, 42).expect("valid cache");
        assert_eq!(median, 7);
        assert_eq!(cached.vocab, ft.vocab);
        assert_eq!(cached.added_tokens, ft.added_tokens);
        for text in corpus() {
            assert_eq!(cached.encode_ids(text), ft.encode_ids(text), "mismatch for {text:?}");
        }

        assert!(FastWordPiece::from_cache_bytes(&bytes, 43).is_none());
        assert!(FastWordPiece::from_cache_bytes(&bytes[..bytes.len() - 1], 42).is_none());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(FastWordPiece::from_cache_bytes(&trailing, 42).is_none());
        let mut other_version = bytes;
        other_version[7] ^= 0xff;
        assert!(FastWordPiece::from_cache_bytes(&other_version, 42).is_none());
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
    }
}
