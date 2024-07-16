import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.dataset import create_val_dataloader_from_config
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"]= '0'
device = "cuda" if torch.cuda.is_available() else "cpu"
# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
sample_size = 458752
model = model.to(device)

# Set up text and timing conditioning
conditioning = [
    # {"prompt": "The genre of music piece is rock, the instruments used are electric guitar, bass, drums, and voice, the tempo is fast with a mood of angry.","seconds_start": 0, "seconds_total": 95},
    # {"prompt": "The genre of music piece is pop, the instruments include the guitar, the tempo is fast with a lively and energetic mood.","seconds_start": 0, "seconds_total": 95},
    # {"prompt": "The genre of music piece is punk, the instruments used are the guitar, bass, and drums, the tempo is fast with a mood of angry.","seconds_start": 0, "seconds_total": 95},
    # {"prompt": "The genre of the music piece is classical. It is performed by a piano and a cello. The tempo is medium with a dramatic and emotional mood.","seconds_start": 0, "seconds_total": 95},
    # {"prompt": "The genre of the music piece is electronic, featuring synthesizers, electric guitars, and drums. The tempo is medium with a groovy beat and a mysterious mood.","seconds_start": 0, "seconds_total": 95},
    # {"prompt": "The genre of the music piece is jazz, which is performed by saxophone and bass. The tempo is medium with a lively and playful mood.","seconds_start": 0, "seconds_total": 95},
    # {"prompt": "The genre of the music piece is ambient, new age, instrumental. It is medium tempo with a calm and soothing mood. The main instruments used are piano, synthesizer, and strings.","seconds_start": 0, "seconds_total": 95},
    # {"prompt": "This is a cinematic, ambient, atmospheric, and mysterious track. It is perfect for films, documentaries, and video games. The track features piano, synthesizers, and strings. The mood of the track is suspenseful, mysterious, and eerie.","seconds_start": 0, "seconds_total": 95},
    # {"prompt": "The music is a solo piano piece. The tempo is slow and the mood is sad and nostalgic.","seconds_start": 0, "seconds_total": 95},
    # {"prompt": "The genre of the music piece is soundtrack, the instruments used are strings, percussion, and synthesizer, the tempo is medium with a dramatic and suspenseful mood.","seconds_start": 0, "seconds_total": 95},

]

with torch.no_grad():
        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            steps=250,
            cfg_scale=7,
            conditioning=conditioning,
            batch_size = len(conditioning),
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )
        # Rearrange audio batch to a single sequence
        # output = rearrange(output, "b d n -> d (b n)")
        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        for i in range(len(conditioning)):
            name = conditioning[i]['prompt'] +'.wav'
            filename = f'recon/demo/{name}'
            wavs = output[i,:,:]
            torchaudio.save(filename, wavs, sample_rate)

