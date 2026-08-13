"""BPE motif discovery over canonicalized agent action sequences.

Iteratively merges the most frequent adjacent pair into a new symbol until
target vocabulary size is reached. Implementation is pure Python: our input
tokens are already string atoms (like `EDIT_SRC_PY`), not characters/bytes,
so we don't need a text-tokenization library.

Algorithm:
    vocab = set of atomic tokens in corpus
    sequences = corpus
    while |vocab| < target_size:
        counts = count all adjacent pairs across sequences
        if no pair repeats: stop
        (a, b) = argmax(counts)
        new = f"{a}+{b}"
        replace every (a,b) adjacent pair in sequences with new
        vocab.add(new)
    return vocab, merges (ordered), re-expressed sequences

Each merge is recorded so we can re-express new corpora with the same BPE
model (cross-corpus transfer).

Usage:
    from analysis.preferences.bpe import train_bpe, apply_bpe
    vocab, merges, expressed = train_bpe(sequences, target_size=200)
    new_expressed = apply_bpe(new_sequences, merges)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BPEModel:
    """A learned BPE model: ordered merges + final vocabulary."""

    # Ordered list of merges: [(a, b, new), ...]. Applied in order.
    merges: list[tuple[str, str, str]]
    # Final vocabulary (atoms + all merged tokens).
    vocab: list[str]

    def summary(self) -> dict:
        length_hist = Counter(len(self._tokens_of(t)) for t in self.vocab)
        return {
            "vocab_size": len(self.vocab),
            "n_merges": len(self.merges),
            "length_distribution": dict(sorted(length_hist.items())),
        }

    def _tokens_of(self, merged: str) -> list[str]:
        """How many atomic tokens are in this merged symbol (length = '+' count + 1)."""
        return merged.split("+")


def _pair_counts(sequences: list[list[str]]) -> Counter:
    """Count all adjacent pairs across sequences."""
    counts: Counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += 1
    return counts


def _merge_pair_in_sequences(
    sequences: list[list[str]],
    a: str,
    b: str,
    new: str,
) -> list[list[str]]:
    """Replace every adjacent (a, b) with the merged token `new`."""
    out = []
    for seq in sequences:
        merged_seq = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                merged_seq.append(new)
                i += 2
            else:
                merged_seq.append(seq[i])
                i += 1
        out.append(merged_seq)
    return out


def train_bpe(
    sequences: list[list[str]],
    target_size: int = 200,
    min_pair_frequency: int = 2,
    verbose: bool = False,
) -> tuple[BPEModel, list[list[str]]]:
    """Train BPE on the given sequences until target vocab size is reached.

    Args:
        sequences: list of token lists
        target_size: stop when vocabulary reaches this size
        min_pair_frequency: stop if no pair appears at least this many times
        verbose: if True, print progress every 20 merges

    Returns:
        (BPEModel, re-expressed sequences)
    """
    # Initial atomic vocabulary
    atomic_vocab = sorted({t for seq in sequences for t in seq})
    vocab = list(atomic_vocab)
    merges: list[tuple[str, str, str]] = []
    working_seqs = [list(seq) for seq in sequences]  # copy

    if verbose:
        print(f"Starting BPE: {len(atomic_vocab)} atomic tokens, target {target_size}")

    while len(vocab) < target_size:
        counts = _pair_counts(working_seqs)
        if not counts:
            if verbose:
                print("  no pairs left; stopping")
            break
        (a, b), count = counts.most_common(1)[0]
        if count < min_pair_frequency:
            if verbose:
                print(f"  most frequent pair ({a}, {b}) has count {count} < {min_pair_frequency}; stopping")
            break
        new = f"{a}+{b}"
        working_seqs = _merge_pair_in_sequences(working_seqs, a, b, new)
        merges.append((a, b, new))
        vocab.append(new)
        if verbose and (len(vocab) - len(atomic_vocab)) % 20 == 0:
            print(f"  merge {len(merges):4d}: vocab={len(vocab)}, last=({a}, {b}) count={count}")

    return BPEModel(merges=merges, vocab=vocab), working_seqs


def apply_bpe(sequences: list[list[str]], model: BPEModel) -> list[list[str]]:
    """Apply learned BPE merges to new sequences (cross-corpus transfer)."""
    working = [list(seq) for seq in sequences]
    for a, b, new in model.merges:
        working = _merge_pair_in_sequences(working, a, b, new)
    return working


def vocabulary_coverage(
    sequences: list[list[str]],
    model: BPEModel,
) -> dict:
    """Report what fraction of a corpus gets merged into multi-token symbols.

    Used for cross-benchmark transfer — how well does a trained BPE model
    compress a new corpus?
    """
    applied = apply_bpe(sequences, model)
    total_tokens = sum(len(s) for s in applied)
    total_multitoken = sum(
        1 for s in applied for t in s if "+" in t
    )
    return {
        "n_sequences": len(sequences),
        "total_tokens_after_bpe": total_tokens,
        "total_atomic_tokens_before": sum(len(s) for s in sequences),
        "compression_ratio": total_tokens / max(sum(len(s) for s in sequences), 1),
        "fraction_multitoken_symbols": total_multitoken / max(total_tokens, 1),
    }
