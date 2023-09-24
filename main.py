from fastapi import FastAPI
from utils import ASRInference
from fastapi import UploadFile, File
import soundfile as sf

app = FastAPI()

asr = ASRInference()

@app.post('/asr')

def inference(file: UploadFile = File(...)):
    audio, _ = sf.read(file.file)
    text = asr.inference(audio)
    return {"text": text}

