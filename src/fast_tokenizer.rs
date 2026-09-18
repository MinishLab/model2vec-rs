use std::collections::HashMap;
use tokenizers::{
    Model, Normalizer, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer,
    models::ModelWrapper,
    normalizers::{NormalizerWrapper, bert::BertNormalizer},
    pre_tokenizers::{PreTokenizerWrapper, bert::BertPreTokenizer},
};

/// Id-only tokenizer for Bert-style WordPiece tokenizers that skips HF's offset bookkeeping.
///
/// Produces the same ids as `Tokenizer::encode_fast(text, false)`. ASCII chunks are handled
/// directly; chunks with non-ASCII characters go through the HF normalizer and pre-tokenizer.
#[derive(Debug)]
pub(crate) struct FastTokenizer {
    vocab: HashMap<String, u32>,
    unk_id: u32,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
    normalizer: BertNormalizer,
    added_tokens: Vec<String>,
}

impl FastTokenizer {
    /// Build a fast tokenizer, or `None` if the tokenizer's pipeline is not supported.
    pub(crate) fn new(tokenizer: &Tokenizer) -> Option<Self> {
        let ModelWrapper::WordPiece(model) = tokenizer.get_model() else {
            return None;
        };
        let Some(NormalizerWrapper::BertNormalizer(normalizer)) = tokenizer.get_normalizer() else {
            return None;
        };
        let Some(PreTokenizerWrapper::BertPreTokenizer(_)) = tokenizer.get_pre_tokenizer() else {
            return None;
        };
        let added_tokens = tokenizer.get_added_tokens_decoder();
        if tokenizer.get_truncation().is_some()
            || tokenizer.get_padding().is_some()
            || added_tokens.values().any(|t| t.normalized)
        {
            return None;
        }
        let vocab = model.get_vocab();
        Some(Self {
            unk_id: *vocab.get(&model.unk_token)?,
            vocab,
            continuing_subword_prefix: model.continuing_subword_prefix.clone(),
            max_input_chars_per_word: model.max_input_chars_per_word,
            normalizer: *normalizer,
            added_tokens: added_tokens.into_values().map(|t| t.content).collect(),
        })
    }

    /// Tokenize `text` into ids, or `None` if it contains an added token and needs the HF tokenizer.
    pub(crate) fn encode(&self, text: &str) -> Option<Vec<u32>> {
        if self.added_tokens.iter().any(|t| text.contains(t.as_str())) {
            return None;
        }
        let mut ids = Vec::new();
        let mut word = String::new();
        for chunk in text.split(|c| self.is_separator(c)) {
            if chunk.is_ascii() {
                self.encode_ascii_chunk(chunk, &mut word, &mut ids);
            } else {
                self.encode_unicode_chunk(chunk, &mut ids)?;
            }
        }
        Some(ids)
    }

    /// ASCII characters that end up splitting words after normalization.
    fn is_separator(&self, c: char) -> bool {
        // With `clean_text`, \x0B and \x0C are removed as control characters instead of splitting.
        matches!(c, '\t' | '\n' | '\r' | ' ') || (!self.normalizer.clean_text && matches!(c, '\x0B' | '\x0C'))
    }

    fn encode_ascii_chunk(&self, chunk: &str, word: &mut String, ids: &mut Vec<u32>) {
        word.clear();
        for b in chunk.bytes() {
            if b.is_ascii_punctuation() {
                self.encode_word(word, ids);
                word.clear();
                self.encode_word(std::str::from_utf8(&[b]).unwrap(), ids);
            } else if !(self.normalizer.clean_text && b.is_ascii_control()) {
                word.push(if self.normalizer.lowercase {
                    b.to_ascii_lowercase()
                } else {
                    b
                } as char);
            }
        }
        self.encode_word(word, ids);
    }

    fn encode_unicode_chunk(&self, chunk: &str, ids: &mut Vec<u32>) -> Option<()> {
        let mut pretokenized = PreTokenizedString::from(chunk);
        pretokenized.normalize(|s| self.normalizer.normalize(s)).ok()?;
        BertPreTokenizer.pre_tokenize(&mut pretokenized).ok()?;
        for (word, _, _) in pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte) {
            self.encode_word(word, ids);
        }
        Some(())
    }

    /// Greedy longest-match WordPiece, mirroring `tokenizers::models::wordpiece::WordPiece::tokenize`.
    fn encode_word(&self, word: &str, ids: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        if word.chars().count() > self.max_input_chars_per_word {
            ids.push(self.unk_id);
            return;
        }
        let first = ids.len();
        let mut piece = String::new();
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let id = loop {
                if start == end {
                    ids.truncate(first);
                    ids.push(self.unk_id);
                    return;
                }
                let lookup = if start == 0 {
                    &word[..end]
                } else {
                    piece.clear();
                    piece.push_str(&self.continuing_subword_prefix);
                    piece.push_str(&word[start..end]);
                    piece.as_str()
                };
                if let Some(&id) = self.vocab.get(lookup) {
                    break id;
                }
                end -= word[start..end].chars().next_back().map_or(1, char::len_utf8);
            };
            ids.push(id);
            start = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CASES: &[&str] = &[
        "",
        "   \n\t ",
        "Hello, World! It's 12,345.67$ (approx.) -- e.g. U.S.A.",
        "snake_case CamelCase kebab-case path/to/file.rs a+b=c",
        "unbelievably antidisestablishmentarianism qzxqzxqzx",
        "Café naïve résumé ÀÉÎõü e\u{301} STRASSE ß İstanbul ΣΑΣ",
        "中文字符 test カタカナ 한국어",
        "emoji 🧿 hi 😀😀 — “quotes” … ﬁ Ⅳ",
        "a\u{00A0}b a\u{3000}b a\u{200B}b a\u{0085}b x\u{2028}y",
        "a\x0Bb a\x0Cb a\x00b a\x1Fb a\x7Fb a\u{FFFD}b a\r\nb",
        "[CLS] hello [SEP] [MASK]ed [mask] [UNK]",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ];

    #[test]
    fn test_matches_hf_tokenizer() {
        let base = Tokenizer::from_file("tests/fixtures/test-model-float32/tokenizer.json").unwrap();
        for (clean_text, lowercase) in [(true, true), (true, false), (false, true), (false, false)] {
            let mut tokenizer = base.clone();
            tokenizer.with_normalizer(Some(BertNormalizer::new(clean_text, true, None, lowercase)));
            let fast = FastTokenizer::new(&tokenizer).expect("fast path should support this tokenizer");
            for text in CASES {
                let expected = tokenizer.encode_fast(*text, false).unwrap().get_ids().to_vec();
                let actual = fast.encode(text).unwrap_or_else(|| expected.clone());
                assert_eq!(
                    actual, expected,
                    "clean_text={clean_text} lowercase={lowercase} text={text:?}"
                );
            }
        }
    }
}
