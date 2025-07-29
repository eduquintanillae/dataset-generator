import sys, os
import csv

from modules.data_loader import DataLoader
from modules.data_chunker import DataChunker
from modules.data_labeler import DataLabeler


class PipelineManager:
    def __init__(self, run_id, config):
        self.run_id = run_id
        self.config = config
        self.data_labeler = None
        self.data_chunker = None
        self.data_loader = DataLoader(file_paths=self.config.file_paths)

    def save_dataset(self, data, path):
        try:
            with open(path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
                print(f"Dataset saved to {path}")
        except Exception as e:
            print(f"Error: {e}")

    def run(self):
        data = self.data_loader.load_data()
        data = self.data_loader.flatten_content(data)
        print(f"Loaded data: {len(data)} characters")

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
        self.save_dataset(
            labeled_data["labeled_data"], f"tmp/{self.run_id}_dataset.csv"
        )

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
