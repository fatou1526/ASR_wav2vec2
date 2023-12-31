from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import os
class ASRInference:
    def __init__(self, model_name = 'Fatou/customwolof'):
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def inference(self, audio):
        inputs = self.processor(audio, sampling_rate = 16000, return_tensors = 'pt')
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim = -1)
        text = self.processor.decode(predicted_ids[0]).lower()
        return text
