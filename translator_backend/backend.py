from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests, zipfile, os

MODEL_URL = "https://huggingface.co/mahigodike/translator_model/resolve/main/nllb_model_zip/nllb_model_zip.zip"
MODEL_ZIP_PATH = "nllb_model.zip"
MODEL_PATH = "nllb_model"

app = FastAPI()

def download_model():
    if not os.path.exists(MODEL_ZIP_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_ZIP_PATH, "wb") as f:
            f.write(response.content)

def extract_model():
    if not os.path.exists(MODEL_PATH):
        print("Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODEL_PATH)

# Load model ONCE
print("Initializing model...")
download_model()
extract_model()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

@app.post("/translate")
def translate(req: TranslateRequest):

    tokenizer.src_lang = req.src_lang
    inputs = tokenizer(req.text, return_tensors="pt")
    bos_id = tokenizer.convert_tokens_to_ids(req.tgt_lang)

    generated = model.generate(
        **inputs,
        forced_bos_token_id=bos_id,
        max_length=400
    )

    output = tokenizer.decode(generated[0], skip_special_tokens=True)

    return {"translation": output}
