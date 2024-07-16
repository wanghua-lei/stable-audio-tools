# from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from datasets import load_dataset
from pedalboard.io import AudioFile
import random

# loading our model weights
model = AutoModel.from_pretrained("./MERT-v1-95M", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("./MERT-v1-95M",trust_remote_code=True)

# load demo audio and set processor
# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["audio"].sampling_rate
sampling_rate = 44100
audio = "/mmu-audio-ssd/frontend/audioSep/wanghualei/code/mtg-jamendo-dataset/path/to/raw_30s/audiofile/37/2837.mp3"
with AudioFile(audio) as f:
    waveform = f.read(f.frames)
if waveform.shape[0] > 1:
            waveform = (waveform[0] + waveform[1]) / 2
else:
    waveform = waveform.squeeze(0)

resample_rate = processor.sampling_rate
# make sure the sample_rate aligned
if resample_rate != sampling_rate:
    print(f'setting rate from {sampling_rate} to {resample_rate}')
    resampler = T.Resample(sampling_rate, resample_rate)
else:
    resampler = None

# audio file is decoded on the fly
if resampler is None:
    input_audio = waveform.squeeze(1)
else:
    input_audio = resampler(torch.from_numpy(waveform))

max_length = 24000*25
if input_audio.shape[-1] > max_length:
    max_start = input_audio.shape[-1] - max_length
    start = random.randint(0, max_start)
    input_audio = input_audio[start: start + max_length]


inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")


with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=False)
print(outputs.last_hidden_state.shape)
# take a look at the output shape, there are 13 layers of representation
# each layer performs differently in different downstream tasks, you should choose empirically
all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]

# for utterance level classification tasks, you can simply reduce the representation in time
time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
print(time_reduced_hidden_states.shape) # [13, 768]

# you can even use a learnable weighted average representation
aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
print(weighted_avg_hidden_states.shape) # [768]