//! A double-array byte trie for longest-prefix vocabulary matching.
//!
//! Node `n`'s child along byte `b` lives at slot `base(n) + b`, and belongs to
//! `n` only if that slot records `n` as its parent. One step is therefore a
//! single array read and compare, with no hashing or substring copies.

/// Marks an absent node, an absent token, or a free slot's parent.
pub(crate) const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
struct Node {
    base: u32,
    parent: u32,
    token: u32,
}

const FREE: Node = Node {
    base: 0,
    parent: NONE,
    token: NONE,
};

/// Failed placements after which a free slot stops being tried as a base.
const MAX_FAILURES: u8 = 16;

/// Free slots still tried during placement: `find(i)` is the smallest
/// candidate at or after `i`. Every slot past the end is a candidate.
#[derive(Default)]
struct Candidates {
    /// `next[i] == i` for a candidate, else a slot no later than the next one.
    next: Vec<usize>,
}

impl Candidates {
    fn find(&mut self, mut i: usize) -> usize {
        while i < self.next.len() && self.next[i] != i {
            let up = self.next[i];
            if up < self.next.len() {
                self.next[i] = self.next[up];
            }
            i = up;
        }
        i
    }

    fn remove(&mut self, i: usize) {
        if i >= self.next.len() {
            let len = self.next.len();
            self.next.extend(len..=i);
        }
        self.next[i] = i + 1;
    }
}

/// Double-array trie over byte strings. The root is node 0.
#[derive(Clone, Debug)]
pub(crate) struct DoubleArrayTrie {
    nodes: Vec<Node>,
}

impl DoubleArrayTrie {
    /// Build from `(key, id)` pairs. Empty keys are ignored; for duplicate
    /// keys the first pair in sorted order wins.
    pub(crate) fn build(mut pieces: Vec<(&[u8], u32)>) -> DoubleArrayTrie {
        pieces.retain(|(key, _)| !key.is_empty());
        pieces.sort_unstable();
        pieces.dedup_by(|a, b| a.0 == b.0);

        let mut nodes = vec![Node { parent: NONE, ..FREE }];
        let mut used = vec![true];
        let mut candidates = Candidates::default();
        candidates.remove(0);
        let mut failures: Vec<u8> = Vec::new();
        // Each queued node covers the sorted pieces in `lo..hi` that share its
        // `depth`-byte prefix.
        let mut queue = std::collections::VecDeque::from([(0u32, 0usize, 0usize, pieces.len())]);
        let mut children: Vec<(u8, usize, usize)> = Vec::new();
        while let Some((node, depth, mut lo, hi)) = queue.pop_front() {
            if lo < hi && pieces[lo].0.len() == depth {
                nodes[node as usize].token = pieces[lo].1;
                lo += 1;
            }
            children.clear();
            while lo < hi {
                let byte = pieces[lo].0[depth];
                let mut end = lo + 1;
                while end < hi && pieces[end].0[depth] == byte {
                    end += 1;
                }
                children.push((byte, lo, end));
                lo = end;
            }
            if children.is_empty() {
                continue;
            }

            // Try bases that put the first child on a candidate free slot.
            // Slots that keep failing are dropped as candidates, which keeps
            // the search short at the cost of a few unused holes.
            let min_byte = children[0].0 as usize;
            let mut slot = candidates.find(min_byte + 1);
            let base = loop {
                let base = slot - min_byte;
                if children
                    .iter()
                    .all(|&(b, _, _)| !used.get(base + b as usize).copied().unwrap_or(false))
                {
                    break base;
                }
                if slot >= failures.len() {
                    failures.resize(slot + 1, 0);
                }
                failures[slot] += 1;
                if failures[slot] >= MAX_FAILURES {
                    candidates.remove(slot);
                }
                slot = candidates.find(slot + 1);
            };

            let top = base + children.last().unwrap().0 as usize;
            if top >= nodes.len() {
                nodes.resize(top + 1, FREE);
                used.resize(top + 1, false);
            }
            nodes[node as usize].base = base as u32;
            for &(b, lo, hi) in &children {
                let slot = base + b as usize;
                used[slot] = true;
                candidates.remove(slot);
                nodes[slot].parent = node;
                queue.push_back((slot as u32, depth + 1, lo, hi));
            }
        }
        assert!(nodes.len() < NONE as usize, "vocabulary trie exceeds 32-bit offsets");
        // Pad so `base + byte` is always in bounds.
        nodes.resize(nodes.len() + 256, FREE);
        DoubleArrayTrie { nodes }
    }

    /// The child of `node` along `byte`, or `NONE`.
    #[inline(always)]
    pub(crate) fn child(&self, node: u32, byte: u8) -> u32 {
        let slot = self.nodes[node as usize].base as usize + byte as usize;
        if self.nodes[slot].parent == node {
            slot as u32
        } else {
            NONE
        }
    }

    /// The token ID `node` spells, or `NONE`.
    #[inline(always)]
    pub(crate) fn token(&self, node: u32) -> u32 {
        self.nodes[node as usize].token
    }

    /// The node reached by walking `key` from the root, or `NONE`.
    pub(crate) fn walk(&self, key: &[u8]) -> u32 {
        let mut node = 0;
        for &byte in key {
            node = self.child(node, byte);
            if node == NONE {
                return NONE;
            }
        }
        node
    }

    /// Number of slots, including free ones.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// The token ID `key` spells, or `NONE`.
    pub(crate) fn get(&self, key: &[u8]) -> u32 {
        match self.walk(key) {
            NONE => NONE,
            node => self.token(node),
        }
    }

    /// Longest token that is a prefix of `bytes`, walking from `start`:
    /// `(token, length)`, or `None` if no prefix of length >= 1 is a token.
    #[inline(always)]
    pub(crate) fn longest_prefix(&self, start: u32, bytes: &[u8]) -> Option<(u32, usize)> {
        let mut node = start;
        let mut best = None;
        for (i, &byte) in bytes.iter().enumerate() {
            node = self.child(node, byte);
            if node == NONE {
                break;
            }
            let token = self.token(node);
            if token != NONE {
                best = Some((token, i + 1));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_map() {
        let words = ["a", "ab", "abc", "b", "##b", "##bc", "héllo", "\u{ff}", "zz", "z"];
        let pieces: Vec<(&[u8], u32)> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_bytes(), i as u32))
            .collect();
        let trie = DoubleArrayTrie::build(pieces);
        for (i, w) in words.iter().enumerate() {
            assert_eq!(trie.get(w.as_bytes()), i as u32, "{w}");
        }
        for missing in ["", "abcd", "c", "##", "#", "hé", "zzz"] {
            assert_eq!(trie.get(missing.as_bytes()), NONE, "{missing}");
        }
        assert_eq!(trie.longest_prefix(0, b"abd"), Some((1, 2)));
        assert_eq!(trie.longest_prefix(0, b"c"), None);
        let cont = trie.walk(b"##");
        assert_eq!(trie.longest_prefix(cont, b"bcd"), Some((5, 2)));
        assert_eq!(trie.longest_prefix(cont, b""), None);
    }

    #[test]
    fn duplicate_keys_keep_the_smallest_id() {
        let trie = DoubleArrayTrie::build(vec![(b"ab".as_slice(), 7), (b"ab", 3), (b"a", 1), (b"ab", 5)]);
        assert_eq!(trie.get(b"ab"), 3);
        assert_eq!(trie.get(b"a"), 1);
    }
}
