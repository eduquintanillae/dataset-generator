import sys, os
import csv


from data_loader import DataLoader
from data_chunker import DataChunker
from data_labeler import DataLabeler


class PipelineManager:
    def __init__(self, config):
        self.config = config
        self.data_labeler = None
        self.data_chunker = None
        self.data_loader = DataLoader(file_paths=self.config.file_paths)

    def run(self):
        data = self.data_loader.load_data()
        data = self.data_loader.flatten_content(data)
        print(f"Loaded data: {len(data)} chunks")

        self.data_chunker = DataChunker(
            text=data,
            method=self.config.method,
            chunk_size=self.config.chunk_size,
            words_per_chunk=self.config.words_per_chunk,
            sentences_per_chunk=self.config.sentences_per_chunk,
            delimiter=self.config.delimiter,
            tokens_per_chunk=self.config.tokens_per_chunk,
        )
        chunks = self.data_chunker.chunk_text()
        print(f"Generated {len(chunks)} chunks using method '{self.config.method}'")

        self.data_labeler = DataLabeler(
            model_name=self.config.model_name,
            chunks=chunks,
            n_questions_per_chunk=self.config.n_questions_per_chunk,
            progress_callback=progress,
        )
        labeled_data = self.data_labeler.label_dataset()
        return labeled_data


def progress(step, total_steps, time, n_input_tokens, n_output_tokens, total_tokens):
    step += 1
    print(f"\n--- Progress: {step}/{total_steps} ---")
    print(f"Time: {time}")
    print(f"Input tokens: {n_input_tokens}")
    print(f"Output tokens: {n_output_tokens}")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config

    config = Config(
        file_paths=[
            "../assets/attention_is_all_you_need.pdf",
            # "../assets/attention_is_all_you_need.txt",
            # "../assets/attention_is_all_you_need.docx",
        ],
        method="character",
        chunk_size=1500,
        model_name="gpt-4o-mini",
        n_questions_per_chunk=2,
    )
    pipeline_manager = PipelineManager(config)
    labeled_data = pipeline_manager.run()
    print(labeled_data)
    print(len(labeled_data["labeled_data"]))
