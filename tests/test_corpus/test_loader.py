import json
import pytest

from auto_style_capture.corpus.loader import load_corpus
from auto_style_capture.corpus.models import Corpus


def test_load_txt_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello world. This is a test.")
    corpus = load_corpus(str(f))
    assert len(corpus) == 1
    assert "Hello world" in corpus.documents[0].text


def test_load_directory(tmp_path):
    (tmp_path / "a.txt").write_text("First document.")
    (tmp_path / "b.md").write_text("Second document.")
    (tmp_path / "c.py").write_text("not a corpus file")
    corpus = load_corpus(str(tmp_path))
    assert len(corpus) == 2


def test_load_json_list_of_strings(tmp_path):
    f = tmp_path / "corpus.json"
    f.write_text(json.dumps(["First text.", "Second text."]))
    corpus = load_corpus(str(f))
    assert len(corpus) == 2


def test_load_json_list_of_objects(tmp_path):
    f = tmp_path / "corpus.json"
    data = [{"text": "Hello.", "author": "Test"}, {"text": "World."}]
    f.write_text(json.dumps(data))
    corpus = load_corpus(str(f))
    assert len(corpus) == 2
    assert corpus.documents[0].metadata.get("author") == "Test"


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_corpus("/nonexistent/path")


def test_corpus_chunks(sample_corpus):
    chunks = sample_corpus.chunks(target_length=20)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_corpus_sample(sample_corpus):
    samples = sample_corpus.sample(1)
    assert len(samples) == 1
