import os
from docx import Document
from PyPDF2 import PdfReader
from typing import List


class DataLoader:
    def __init__(self, file_paths: List[str] = None):
        self.file_paths = file_paths

    def read_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
            return None

    def read_docx(self, file_path):
        try:
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading DOCX file {file_path}: {e}")
            return None

    def read_txt(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            return text.strip()
        except Exception as e:
            print(f"Error reading TXT file {file_path}: {e}")
            return None

    def read_file(self, file_path: str):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".pdf":
            return self.read_pdf(file_path)
        elif ext == ".docx":
            return self.read_docx(file_path)
        elif ext == ".txt":
            return self.read_txt(file_path)
        else:
            print(f"Unsupported file type: {ext}")
            return None

    def load_data(self):
        if not self.file_paths:
            print("No file paths provided.")
            return []

        data = []
        not_supported_files = []
        for file_path in self.file_paths:
            text = self.read_file(file_path)
            if text:
                data.append({"file_path": file_path, "content": text})
            else:
                not_supported_files.append(file_path)
        return data

    def flatten_content(self, data):
        data = "\n".join([item["content"] for item in data if "content" in item])
        return data.strip()


if __name__ == "__main__":
    data_loader = DataLoader(
        file_paths=[
            "../assets/attention_is_all_you_need.pdf",
            "../assets/attention_is_all_you_need.docx",
            "../assets/attention_is_all_you_need.txt",
        ]
    )
    data = data_loader.load_data()
    print(f"Number of files {len(data)}")
    data = data_loader.flatten_content(data)
    print(data)
