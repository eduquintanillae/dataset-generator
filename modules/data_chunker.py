from data_loader import DataLoader

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import nltk

nltk.download("punkt_tab")
# If using tiktoken for OpenAI models:
# import tiktoken


class DataChunker:
    def __init__(
        self,
        text,
        method="paragraph",
        chunk_size=500,
        words_per_chunk=100,
        sentences_per_chunk=3,
        delimiter="\n",
        tokens_per_chunk=512,
        semantic_clusters=10,
    ):
        self.text = text
        self.method = method
        self.chunk_size = chunk_size
        self.words_per_chunk = words_per_chunk
        self.sentences_per_chunk = sentences_per_chunk
        self.delimiter = delimiter
        self.tokens_per_chunk = tokens_per_chunk
        self.semantic_clusters = semantic_clusters

    def chunk_by_character(self):
        return [
            self.text[i : i + self.chunk_size]
            for i in range(0, len(self.text), self.chunk_size)
        ]

    def chunk_by_word(self):
        words = self.text.split()
        return [
            " ".join(words[i : i + self.words_per_chunk])
            for i in range(0, len(words), self.words_per_chunk)
        ]

    def chunk_by_sentence(self):
        sentences = nltk.sent_tokenize(self.text)
        return [
            " ".join(sentences[i : i + self.sentences_per_chunk])
            for i in range(0, len(sentences), self.sentences_per_chunk)
        ]

    def chunk_by_paragraph(self):
        return [p.strip() for p in self.text.split("\n") if p.strip()]

    def chunk_by_delimiter(self):
        return [
            chunk.strip() for chunk in self.text.split(self.delimiter) if chunk.strip()
        ]

    def chunk_by_token(self):
        # Using nltk for tokenization
        tokens = nltk.word_tokenize(self.text)
        return [
            " ".join(tokens[i : i + self.tokens_per_chunk])
            for i in range(0, len(tokens), self.tokens_per_chunk)
        ]
        # If using tiktoken, replace above with:
        # enc = tiktoken.get_encoding("cl100k_base")
        # tokens = enc.encode(self.text)
        # return [enc.decode(tokens[i:i+tokens_per_chunk]) for i in range(0, len(tokens), tokens_per_chunk)]

    def chunk_by_semantic(self):
        sentences = nltk.sent_tokenize(self.text)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)
        kmeans = KMeans(
            n_clusters=min(self.semantic_clusters, len(sentences)), random_state=0
        )
        labels = kmeans.fit_predict(embeddings)
        clusters = [[] for _ in range(max(labels) + 1)]
        for sentence, label in zip(sentences, labels):
            clusters[label].append(sentence)
        return [" ".join(cluster) for cluster in clusters if cluster]

    def chunk_text(self):
        if self.method == "character":
            return self.chunk_by_character()
        elif self.method == "word":
            return self.chunk_by_word()
        elif self.method == "sentence":
            return self.chunk_by_sentence()
        elif self.method == "paragraph":
            return self.chunk_by_paragraph()
        elif self.method == "delimiter":
            return self.chunk_by_delimiter()
        elif self.method == "token":
            return self.chunk_by_token()
        elif self.method == "semantic":
            return self.chunk_by_semantic()
        else:
            raise ValueError("Unknown chunking method.")


if __name__ == "__main__":
    data_loader = DataLoader(
        file_paths=[
            "../assets/attention_is_all_you_need.pdf",
            "../assets/attention_is_all_you_need.txt",
            "../assets/attention_is_all_you_need.docx",
        ]
    )
    data = data_loader.load_data()
    data = data_loader.flatten_content(data)
    char_chunks = DataChunker(data, method="character", chunk_size=500).chunk_text()
    print(f"Number of character chunks: {len(char_chunks)}")
    word_chunks = DataChunker(data, method="word", words_per_chunk=100).chunk_text()
    print(f"Number of word chunks: {len(word_chunks)}")
    sentence_chunks = DataChunker(
        data, method="sentence", sentences_per_chunk=3
    ).chunk_text()
    print(f"Number of sentence chunks: {len(sentence_chunks)}")
    paragraph_chunks = DataChunker(data, method="paragraph").chunk_text()
    print(f"Number of paragraph chunks: {len(paragraph_chunks)}")
    delimiter_chunks = DataChunker(
        data, method="delimiter", delimiter="\n"
    ).chunk_text()
    print(f"Number of delimiter chunks: {len(delimiter_chunks)}")
    token_chunks = DataChunker(data, method="token", tokens_per_chunk=512).chunk_text()
    print(f"Number of token chunks: {len(token_chunks)}")
    semantic_chunks = DataChunker(
        data, method="semantic", semantic_clusters=10
    ).chunk_text()
    print(f"Number of semantic chunks: {len(semantic_chunks)}")
