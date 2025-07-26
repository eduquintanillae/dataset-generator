import pytest
from data_chunker import DataChunker
from data_loader import DataLoader
from unittest.mock import patch, mock_open


@pytest.fixture
def sample_text():
    return "This is a sample text for testing the DataChunker class. It contains multiple sentences and paragraphs."


def test_chunk_by_character(sample_text):
    chunker = DataChunker(sample_text, method="character", chunk_size=50)
    chunks = chunker.chunk_text()
    assert len(chunks) > 0
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_chunk_by_word(sample_text):
    chunker = DataChunker(sample_text, method="word", words_per_chunk=5)
    chunks = chunker.chunk_text()
    assert len(chunks) > 0
    assert all(len(chunk.split()) <= 5 for chunk in chunks)


def test_chunk_by_sentence(sample_text):
    chunker = DataChunker(sample_text, method="sentence", sentences_per_chunk=2)
    chunks = chunker.chunk_text()
    assert len(chunks) > 0
    assert all(len(chunk.split(". ")) <= 2 for chunk in chunks)


def test_chunk_by_paragraph(sample_text):
    chunker = DataChunker(sample_text, method="paragraph")
    chunks = chunker.chunk_text()
    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_by_delimiter(sample_text):
    chunker = DataChunker(sample_text, method="delimiter", delimiter=".")
    chunks = chunker.chunk_text()
    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_by_tokens(sample_text):
    chunker = DataChunker(sample_text, method="tokens", tokens_per_chunk=10)
    chunks = chunker.chunk_text()
    assert len(chunks) > 0
    assert all(len(chunk.split()) <= 10 for chunk in chunks)


def test_chunk_by_semantic(sample_text):
    with patch("data_chunker.SentenceTransformer") as MockModel:
        mock_model = MockModel.return_value
        mock_model.encode.return_value = [0.1] * 768  # Mock embedding

        chunker = DataChunker(sample_text, method="semantic", semantic_clusters=2)
        chunks = chunker.chunk_text()

        assert len(chunks) > 0
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)
