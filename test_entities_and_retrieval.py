from weir_whisper.nlp.entities import EntityLinker, tokenize
from weir_whisper.retrieval.bm25 import BM25
from weir_whisper.retrieval.corpus import chunk_markdown


def linker():
    return EntityLinker(["C101", "C201", "C301", "C401", "C501", "C601"])


def test_tag_variants_resolve_to_one_entity():
    lk = linker()
    for text in ("FV207 hunting", "fv-207 hunting", "FV 207 hunting", "fv207 hunting"):
        ex = lk.extract(text)
        assert "FV207" in ex.entities
        assert ex.columns == ["C201"]


def test_nicknames_and_symptoms():
    ex = linker().extract("Depropaniser steam flow cycling, OP sawtoothing; dP across bottom section dropped off")
    assert "C201" in ex.columns
    assert "cycling" in ex.symptoms
    assert "dp_low" in ex.symptoms


def test_unknown_tags_are_kept_as_unresolved():
    ex = linker().extract("Checked FV999 and C201")
    assert "FV999" not in ex.entities
    assert ex.unresolved


def test_tokenizer_normalises_tags():
    assert "fv207" in tokenize("FV-207 sticking")


def test_bm25_prefers_the_matching_document():
    docs = ["steam valve FV207 hunting and sticking", "pump seal leak on P201A", "lagging damaged on C301"]
    s = BM25(docs).scores("fv-207 sticking")
    assert s.argmax() == 0


def test_markdown_chunking_respects_headings():
    text = "# Title\n\n## A\n\npara one.\n\npara two.\n\n## B\n\npara three."
    chunks = chunk_markdown("KB999", text, 50)
    heads = [h for _, h, _ in chunks]
    assert "A" in heads and "B" in heads
    assert all(cid.startswith("KB999#") for cid, _, _ in chunks)
