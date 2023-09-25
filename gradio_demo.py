import gradio as gr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

# Load the pretrained wav2vec2 model and processor
model_name = "Fatou/customwolof"

model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

def inference(audio_file):
    with open(audio_file, "rb") as f:
        audio, _ = sf.read(f)

    input_values = processor(audio, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

# Create a Gradio interface
demo = gr.Interface(
   fn=inference,
   inputs=gr.inputs.Audio(type="filepath", label="Upload an audio file"),
   outputs=gr.outputs.Textbox(label="Transcription"),
)

demo.launch()
