from pydantic import BaseModel


class Config(BaseModel):
    """
    Configuration for the dataset generator.
    """

    file_paths: list[str] = []
    method: str = "character"
    chunk_size: int = 500
    words_per_chunk: int = 100
    sentences_per_chunk: int = 3
    delimiter: str = "\n"
    tokens_per_chunk: int = 512
    semantic_clusters: int = 10
    model_name: str = "gpt-4o-mini"
    n_questions_per_chunk: int = 2
