from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from typing import List

from azure.storage.blob import BlobServiceClient
import boto3
from pydantic import BaseModel
import os
import sys
from dotenv import load_dotenv

from modules.pipeline_manager import PipelineManager
from config import Config

from datetime import datetime, timezone
import uuid


load_dotenv()

app = FastAPI()


@app.get("/health")
async def health_check():
    try:
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_dataset")
async def generate_dataset(
    files: List[UploadFile] = File(...),
    method: str = Form(...),
    model_name: str = Form(...),
    n_questions_per_chunk: int = Form(...),
    chunk_size: int = Form(500),
    words_per_chunk: int = Form(100),
    sentences_per_chunk: int = Form(3),
    delimiter: str = Form("\n"),
    tokens_per_chunk: int = Form(512),
    semantic_clusters: int = Form(10),
):
    try:
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        file_paths = []
        for file in files:
            file_location = f"tmp/{file.filename}"
            with open(file_location, "wb") as f:
                f.write(await file.read())
            file_paths.append(file_location)

        config = Config(
            file_paths=file_paths,
            method=method,
            model_name=model_name,
            n_questions_per_chunk=n_questions_per_chunk,
            chunk_size=chunk_size,
            words_per_chunk=words_per_chunk,
            sentences_per_chunk=sentences_per_chunk,
            delimiter=delimiter,
            tokens_per_chunk=tokens_per_chunk,
            semantic_clusters=semantic_clusters,
        )
        pipeline_manager = PipelineManager(run_id, config)
        labeled_data = pipeline_manager.run()

        for file_path in file_paths:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: could not delete {file_path}: {e}")

        return {
            "status": "success",
            "message": "Dataset generated successfully",
            "data": labeled_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
