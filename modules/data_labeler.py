from modules.data_loader import DataLoader
from modules.data_chunker import DataChunker
import dotenv
import os
from openai import OpenAI
import re
import time


dotenv.load_dotenv()

SYSTEM_PROMPT = """
You are an expert on labeling datasets based on chunks of text.
Your task is to generate questions based on the provided text chunks.
"""

USER_PROMPT = """
Please generate {n_questions_per_chunk} question-and-answer pairs based on the following context information.
This question and answer will be used to generate a dataset for finetuning an LLM, so please ensure the questions are clear and the answers are concise.
Use the following format for your response:
**1**
question: {{question}} 
answer: {{answer}}

**2**
question: {{question}} 
answer: {{answer}}

--- CONTEXT INFORMATION ---
{chunk}
"""


class DataLabeler:
    def __init__(
        self, model_name, chunks, n_questions_per_chunk, progress_callback=None
    ):
        self.model_name = model_name
        self.model = None
        self.chunks = chunks
        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = USER_PROMPT
        self.n_questions_per_chunk = n_questions_per_chunk
        self.progress_callback = progress_callback

        self.load_model()

    def load_model(self):
        if "gpt-4o-mini" in self.model_name.lower():
            self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def format_user_prompt(self, chunk):
        formatted_prompts = self.user_prompt.format(
            n_questions_per_chunk=self.n_questions_per_chunk, chunk=chunk
        )
        return formatted_prompts

    def label_dataset(self):
        labeled_data = []
        total_input_tokens = 0
        total_output_tokens = 0
        total = len(self.chunks)
        initial_time = time.time()
        for idx, chunk in enumerate(self.chunks):
            start_time = time.time()
            user_prompt = self.format_user_prompt(chunk)
            model_response = self.model_completion(self.system_prompt, user_prompt)
            final_response = self.postprocess_response(model_response["response"])
            for response in final_response:
                labeled_data.append(
                    {
                        "chunk": chunk,
                        "question": response["question"],
                        "answer": response["answer"],
                    }
                )
            final_time = time.time() - start_time
            total_input_tokens += model_response["prompt_tokens"]
            total_output_tokens += model_response["completion_tokens"]
            if self.progress_callback:
                self.progress_callback(
                    step=idx,
                    total_steps=total,
                    time=final_time,
                    n_input_tokens=model_response["prompt_tokens"],
                    n_output_tokens=model_response["completion_tokens"],
                    total_tokens=model_response["total_tokens"],
                )
        total_time = time.time() - initial_time
        results = {
            "labeled_data": labeled_data,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_time": total_time,
        }
        return results

    def model_completion(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.model.chat.completions.create(
            model=self.model_name, messages=messages
        )
        completion_response = {
            "response": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return completion_response

    def postprocess_response(self, response):
        pattern = r"\*\*\d+\*\*\s*question:\s*(.*?)\s*answer:\s*(.*?)(?=\n\*\*|\Z)"
        matches = re.findall(pattern, response, re.DOTALL)
        questions_answers = [
            {"question": match[0].strip(), "answer": match[1].strip()}
            for match in matches
        ]

        return questions_answers


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
    char_chunks = char_chunks[:2]  # Testing

    labeler = DataLabeler(
        model_name="gpt-4o-mini",
        chunks=char_chunks,
        n_questions_per_chunk=2,
    )
    labeled_data = labeler.label_dataset()

    print(f"Labeled data: {labeled_data['labeled_data']}")
    print(f"Number of samples: {len(labeled_data['labeled_data'])}")
