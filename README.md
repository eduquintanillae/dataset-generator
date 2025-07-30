# 🛠️ Dataset Generator: First Step to Fine-Tuning

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.12-blue)

This repository provides a framework for generating datasets that can be used to fine-tune large language models (LLMs). It includes modules for loading data, chunking it into manageable pieces, and labeling the data for training purposes.

Example of the generated dataset:
```json
[
  {
    "chunk": "The capital of France is Paris.",
    "question": "What is the capital of France?",
    "answer": "Paris"
  },
  {
    "chunk": "The largest planet in our solar system is Jupiter.",
    "question": "What is the largest planet in our solar system?",
    "answer": "Jupiter"
  }
]
```

## Table of Contents
- [⚙️ Installation & Usage](#installation)
- [🧩 Modules](#modules)
- [🤝 Contributing](#contributing)
- [📝 License](#license)
- [🧑‍💻 Author](#author)

## ⚙️ Installation & Usage
1. Clone the repository
```bash
git clone https://github.com/eduquintanillae/dataset-generator.git
cd dataset-generator
```
2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required packages
```bash
pip install -r requirements.txt
```
4. Create a `.env` file in the root directory and set the `OPENAI_API_KEY` variable with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```
5. Run uvicorn to start the FastAPI server
```bash
uvicorn app.main:app --reload
```

6. Send a request to the POST /generate_dataset endpoint using requests in Python:

<details> <summary>	 Click to expand Python snippet</summary>

```python
import requests

url = "http://localhost:8000/generate_dataset"

files = {
    "files": open("your_file.txt", "rb")
}

data = {
    "method": "your_method",
    "model_name": "your_model",
    "n_questions_per_chunk": 5,
    "chunk_size": 500,
    "words_per_chunk": 100,
    "sentences_per_chunk": 3,
    "delimiter": "\n",
    "tokens_per_chunk": 512,
    "semantic_clusters": 10
}

response = requests.post(url, files=files, data=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
```

If you're using multiple files, change files like this:

```python
files = [
    ("files", open("file1.txt", "rb")),
    ("files", open("file2.txt", "rb")),
]
```
</details>


## 🧩 Modules

This project is structured into several key modules, each responsible for a specific part of the dataset generation process:

### Data Loader

The Data Loader module is responsible for loading data from various sources, such as text files, PDFs, and Word documents. It preprocesses the data to ensure it's in a suitable format for chunking and labeling.

### Data Chunker

The Data Chunker module takes the preprocessed data and divides it into smaller, manageable chunks. This is essential for training LLMs, as they often have limitations on the maximum input size.

The Chunker has several strategies for chunking, including:
- **character**: Splits the text into chunks of a specified number of characters.
- **word**: Splits the text into chunks of a specified number of words.
- **sentence**: Splits the text into chunks based on sentence boundaries.
- **paragraph**: Splits the text into chunks based on paragraph boundaries.
- **delimiter**: Splits the text into chunks based on a specified delimiter (e.g. '\n').
- **token**: Splits the text into chunks based on a specified number of tokens.
- **semantic**: Splits the text into chunks based on semantic meaning.

### Data Labeler

The Data Labeler module is responsible for labeling the chunks of data for training purposes. 
In this step, an LLM is used to generate question-and-answer labels for each chunk based on its content.

### Pipeline Manager
The Pipeline Manager orchestrates the entire dataset generation process. It coordinates the Data Loader, Data Chunker, and Data Labeler modules to ensure a smooth workflow from raw data to labeled dataset.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request. See `CONTRIBUTING.md` for details.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🧑‍💻 Author
Created by [Eduardo Quintanilla](https://github.com/eduquintanillae) - feel free to reach out for any questions or suggestions.