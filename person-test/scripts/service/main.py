import os
import base64
import json
import shutil
import asyncio
import requests
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configuration
VLLM_API_URL = "http://localhost:18001/v1/chat/completions"
MODEL_NAME = "/dataset/cep/llm-training/person-test/models/merged"
TRAIN_DATA_DIR = "/dataset/cep/llm-training/person-test/data"
SCRIPTS_DIR = "/dataset/cep/llm-training/person-test/scripts"

app = FastAPI(title="Person Identification API", description="API for Person Identification utilizing Qwen3-VL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_identity(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        base64_image = base64.b64encode(contents).decode('utf-8')

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.1
        }

        response = requests.post(VLLM_API_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        try:
            result_json = json.loads(content)
            return {"result": result_json}
        except json.JSONDecodeError:
            return {"result": content.strip()}

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def register_new_person(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    person_dir = os.path.join(TRAIN_DATA_DIR, "raw_images", name)
    os.makedirs(person_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        file_path = os.path.join(person_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

    background_tasks.add_task(run_training_pipeline, name)
    
    return {"status": "upload_success", "message": f"Saved {len(saved_files)} images for {name}. Training queued."}

def run_training_pipeline(name: str):
    print(f"Starting training for {name}...")
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
