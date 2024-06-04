from datasets import load_dataset, load_metric, Audio
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import torchaudio.functional as F

#pip install librosa soundfile

USE_LM = False
DATASET_ID = "Jzuluaga/atcosim_corpus"
MODEL_ID = "Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-atcosim"

# 1. Load the dataset
# we only load the 'test' partition, however, if you want to load the 'train' partition, you can change it accordingly
atcosim_corpus_test = load_dataset(DATASET_ID, "default", split="test")

# 2. Load the model
model = AutoModelForCTC.from_pretrained(MODEL_ID)

# 3. Load the processors, we offer support with LM, which should yield better resutls
if USE_LM:
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(MODEL_ID)
else:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

# 4. Format the test sample
sample = next(iter(atcosim_corpus_test))
file_sampling_rate = sample['audio']['sampling_rate']

# resample if neccessary
if file_sampling_rate != 16000:
    resampled_audio = F.resample(torch.tensor(sample["audio"]["array"]), file_sampling_rate, 16000).numpy()
else:
    resampled_audio = torch.tensor(sample["audio"]["array"]).numpy()

input_values = processor(resampled_audio, return_tensors="pt").input_values

# 5. Run the forward pass in the model
with torch.no_grad():
    logits = model(input_values).logits
    
# get the transcription with processor
if USE_LM:
    transcription = processor.batch_decode(logits.numpy()).text
else:
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)

# print the output
print(transcription)
