//! Open-addressing hash table from short vocabulary pieces to token IDs.
//!
//! A piece of up to [`KEY_BYTES`] bytes is keyed by its bytes packed
//! little-endian into two `u64`s, so a lookup is two multiplies and a probe
//! or two, with no string hashing or comparison. Probing goes through one
//! tag byte per slot and only reads an entry when its tag matches; the tags
//! of a whole vocabulary fit in L1, so failed lookups are settled from them.

/// Longest piece the table can hold.
pub(crate) const KEY_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Entry {
    lo: u64,
    hi: u64,
    id: u32,
    len: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PieceTable {
    /// Power-of-two length, at most a quarter full; 0 marks an empty slot.
    tags: Vec<u8>,
    entries: Vec<Entry>,
    shift: u32,
}

impl PieceTable {
    /// Build from `(piece, id)` pairs; pieces that are empty or longer than
    /// [`KEY_BYTES`] are skipped, and for duplicate pieces the smallest id wins.
    pub(crate) fn build<'a>(pieces: impl IntoIterator<Item = (&'a [u8], u32)>) -> PieceTable {
        let mut pieces: Vec<(u32, &[u8])> = pieces
            .into_iter()
            .filter(|(p, _)| !p.is_empty() && p.len() <= KEY_BYTES)
            .map(|(p, id)| (id, p))
            .collect();
        pieces.sort_unstable();

        let size = (4 * pieces.len()).next_power_of_two().max(16);
        let mut table = PieceTable {
            tags: vec![0; size],
            entries: vec![Entry::default(); size],
            shift: 64 - size.trailing_zeros(),
        };
        for (id, piece) in pieces {
            let (lo, hi) = (load(piece, 0), load(piece, 8));
            let len = piece.len();
            if table.get(lo, hi, len).is_some() {
                continue;
            }
            let (mut i, tag) = table.slot(lo, hi);
            while table.tags[i] != 0 {
                i = (i + 1) & (size - 1);
            }
            table.tags[i] = tag;
            table.entries[i] = Entry {
                lo,
                hi,
                id,
                len: len as u32,
            };
        }
        table
    }

    #[inline(always)]
    fn slot(&self, lo: u64, hi: u64) -> (usize, u8) {
        let h = (lo ^ hi.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((h >> self.shift) as usize, (h >> (self.shift - 7)) as u8 | 0x80)
    }

    /// Token ID of the `len`-byte piece packed into `(lo, hi)`, whose bytes
    /// past `len` must be zero.
    #[inline(always)]
    pub(crate) fn get(&self, lo: u64, hi: u64, len: usize) -> Option<u32> {
        let mask = self.tags.len() - 1;
        let (mut i, tag) = self.slot(lo, hi);
        loop {
            let t = self.tags[i];
            if t == tag {
                let e = &self.entries[i];
                if e.lo == lo && e.hi == hi && e.len as usize == len {
                    return Some(e.id);
                }
            } else if t == 0 {
                return None;
            }
            i = (i + 1) & mask;
        }
    }

    /// Token ID of the first `len` bytes of the key `(lo, hi)`.
    #[inline(always)]
    pub(crate) fn get_prefix(&self, lo: u64, hi: u64, len: usize) -> Option<u32> {
        let (lo, hi) = prefix(lo, hi, len);
        self.get(lo, hi, len)
    }
}

/// Mask keeping the low `n <= 8` bytes of a `u64`.
#[inline(always)]
pub(crate) fn low_bytes(n: usize) -> u64 {
    if n >= 8 { u64::MAX } else { (1u64 << (8 * n)) - 1 }
}

/// The key of the first `len <= KEY_BYTES` bytes of `(lo, hi)`.
#[inline(always)]
pub(crate) fn prefix(lo: u64, hi: u64, len: usize) -> (u64, u64) {
    if len <= 8 {
        (lo & low_bytes(len), 0)
    } else {
        (lo, hi & low_bytes(len - 8))
    }
}

/// Drop the first `n` (1..=KEY_BYTES) bytes of the key `(lo, hi)`.
#[inline(always)]
pub(crate) fn shift_out(lo: u64, hi: u64, n: usize) -> (u64, u64) {
    match n {
        16.. => (0, 0),
        8.. => (hi >> (8 * (n - 8)), 0),
        _ => ((lo >> (8 * n)) | (hi << (64 - 8 * n)), hi >> (8 * n)),
    }
}

/// Up to eight bytes of `s` from `i`, little-endian, zero-padded past the end.
#[inline(always)]
pub(crate) fn load(s: &[u8], i: usize) -> u64 {
    match s.get(i..i + 8) {
        Some(b) => u64::from_le_bytes(b.try_into().unwrap()),
        None => {
            let mut buf = [0u8; 8];
            let tail = s.get(i..).unwrap_or(&[]);
            buf[..tail.len()].copy_from_slice(tail);
            u64::from_le_bytes(buf)
        }
    }
}

/// Count the leading ASCII letters and digits of eight packed bytes, and
/// return the bytes with ASCII uppercase letters lowercased.
///
/// Per byte, with the high bit masked off so sums cannot carry: adding
/// `0x80 - lo` sets the high bit when the byte is `>= lo`, and adding
/// `0x80 - hi - 1` when it is `> hi`. Letters are tested with bit `0x20`
/// forced on, which folds `A-Z` onto `a-z` and moves no other byte there.
#[inline(always)]
pub(crate) fn scan_alnum(x: u64) -> (usize, u64) {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = ONES * 0x80;
    let low7 = x & !HIGH;
    let folded = low7 | (ONES * 0x20);
    let letters =
        folded.wrapping_add(ONES * (0x80 - b'a' as u64)) & !folded.wrapping_add(ONES * (0x80 - b'z' as u64 - 1));
    let digits = low7.wrapping_add(ONES * (0x80 - b'0' as u64)) & !low7.wrapping_add(ONES * (0x80 - b'9' as u64 - 1));
    let other = !((letters | digits) & !x) & HIGH;
    ((other.trailing_zeros() >> 3) as usize, x | ((letters & !x & HIGH) >> 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn matches_a_map() {
        let words = [
            "a",
            "ab",
            "abc",
            "hello",
            "abcdefgh",
            "abcdefghi",
            "abcdefghijklmnop",
            "x\0",
            "x",
        ];
        let map: HashMap<&str, u32> = words.iter().enumerate().map(|(i, w)| (*w, i as u32)).collect();
        let table = PieceTable::build(map.iter().map(|(k, &v)| (k.as_bytes(), v)));
        for probe in words.iter().copied().chain(["", "abcd", "abcdefghij", "b", "x\0\0"]) {
            let b = probe.as_bytes();
            let got = if b.is_empty() {
                None
            } else {
                table.get_prefix(load(b, 0), load(b, 8), b.len())
            };
            assert_eq!(got, map.get(probe).copied(), "{probe:?}");
        }
    }

    #[test]
    fn duplicate_pieces_keep_the_smallest_id() {
        let table = PieceTable::build([(&b"ab"[..], 7), (&b"ab"[..], 3)]);
        assert_eq!(table.get(load(b"ab", 0), 0, 2), Some(3));
    }

    #[test]
    fn scan_alnum_matches_bytewise() {
        for b in 0..=255u8 {
            for &pos in &[0usize, 3, 7] {
                let mut bytes = *b"Ab9zZ0aQ";
                bytes[pos] = b;
                let (n, lowered) = scan_alnum(u64::from_le_bytes(bytes));
                let want = bytes.iter().position(|c| !c.is_ascii_alphanumeric()).unwrap_or(8);
                assert_eq!(n, want, "byte {b:#x} at {pos}");
                let lowered = lowered.to_le_bytes();
                for k in 0..n {
                    assert_eq!(lowered[k], bytes[k].to_ascii_lowercase());
                }
            }
        }
    }

    #[test]
    fn shift_out_drops_leading_bytes() {
        let s = b"abcdefghijklmnop";
        let (lo, hi) = (load(s, 0), load(s, 8));
        for n in 1..=16 {
            let (l, h) = shift_out(lo, hi, n);
            assert_eq!((l, h), (load(&s[n..], 0), load(&s[n..], 8)), "n = {n}");
        }
    }
}
