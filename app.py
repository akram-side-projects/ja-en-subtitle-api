from fastapi import FastAPI
from transformers import MarianMTModel, MarianTokenizer
import torch

app = FastAPI()

MODEL_DIR = "model"

tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR)
model = MarianMTModel.from_pretrained(MODEL_DIR)
model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate")
def translate(payload: dict):
    texts = payload["text"]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=5,
            max_length=128
        )

    translations = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    return {"translations": translations}
