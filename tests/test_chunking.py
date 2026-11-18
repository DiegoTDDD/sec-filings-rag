"""Sentence-aware chunking: size limit, overlap and no lost or split sentences."""
import chunk

_WORDS = ("revenue supply export demand margin credit risk market filing capital "
          "liquidity competition regulation customers products services").split()


def _sentences(n, words=28):
    """Realistic sentences (~180 chars) with a unique id so each one can be traced."""
    return [
        f"Sentence{i:03d} " + " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(words)) + "."
        for i in range(n)
    ]


def test_empty_input_produces_no_chunks():
    assert chunk.chunk_sentences([]) == []


def test_short_text_stays_in_one_chunk():
    sentences = ["Apple designs smartphones.", "It sells services."]
    assert chunk.chunk_sentences(sentences) == ["Apple designs smartphones. It sells services."]


def test_every_sentence_appears_whole_in_some_chunk():
    sentences = _sentences(40)
    chunks = chunk.chunk_sentences(sentences)
    for sentence in sentences:
        assert any(sentence in c for c in chunks), f"lost or split: {sentence[:20]}"


def test_chunks_respect_target_size_plus_overlap():
    sentences = _sentences(40)
    chunks = chunk.chunk_sentences(sentences)
    longest_sentence = max(len(s) for s in sentences)
    limit = chunk.TARGET_CHARS + chunk.OVERLAP_CHARS + longest_sentence
    assert len(chunks) > 1
    assert all(len(c) <= limit for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk.chunk_sentences(_sentences(40))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_word = current.split()[0]
        assert first_word in previous[-chunk.OVERLAP_CHARS - 20:]


def test_overlap_starts_at_a_word_boundary():
    chunks = chunk.chunk_sentences(_sentences(40))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_token = current.split()[0]
        assert first_token in previous.split(), f"chunk starts mid-word: {first_token!r}"


def test_a_single_oversized_sentence_is_kept_intact():
    big = "y" * (chunk.TARGET_CHARS * 2) + "."
    assert chunk.chunk_sentences([big]) == [big]
