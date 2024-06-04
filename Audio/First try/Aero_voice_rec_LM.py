###prerequisite :
##download ffmpeg
#add it to paath by cmd admin :
#setx /M PATH "%PATH%;C:\Users\thiba\OneDrive - Ecole de l'air\USAFA IA\Vcode\Aero_voice_rec\ffmpeg\bin"

#download https://visualstudio.microsoft.com/fr/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false
#for the use of LM


import gradio as gr
from datasets import load_dataset, load_metric, Audio
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import torchaudio.functional as F
import torchaudio


# Constants
USE_LM = True
MODEL_ID = "Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-atcosim"

# Load the model and processor
model = AutoModelForCTC.from_pretrained(MODEL_ID)
if USE_LM:
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(MODEL_ID)
else:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)


def transcribe_audio(audio_file_path):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    if sample_rate != 16000:
        waveform = F.resample(waveform, orig_freq=sample_rate, new_freq=16000)

    input_values = processor(waveform.squeeze(0).numpy(), return_tensors="pt").input_values
    logits = model(input_values).logits

    # Utilisez .detach() pour enlever logits du graph de calcul et .numpy() pour convertir en array NumPy
    decoded_results = processor.batch_decode(logits.detach().numpy())

    if USE_LM:
        transcription = decoded_results['text']
        #transcription = decoded_results[0]['text']
    else:
        pred_ids = torch.argmax(logits, dim=-1)
        transcriptiontranscription = processor.decode(pred_ids[0])

    return transcription



iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(label="Upload MP3 File", type="filepath"), 
    outputs=gr.Textbox(),
    title="Audio Transcription Service",
    description="Upload an MP3 file to get its transcription using a fine-tuned Wav2Vec 2.0 model."
)


iface.launch()