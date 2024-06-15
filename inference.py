import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.dataset import create_val_dataloader_from_config
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
# sample_size = 983040 #20s
model = model.to(device)
batch_size = 10
# Set up text and timing conditioning
# conditioning = [
#     {"prompt": "An emergency siren ringing with car horn honking","seconds_start": 0, "seconds_total": 10},
#     {"prompt": "A bus engine running followed by a bus horn honking","seconds_start": 0, "seconds_total": 10}
# ]

dataset_config = {
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "my_audio",
            "path": ["/mmu-audio-ssd/frontend/audioSep/wanghualei/code/stable-audio-tools/val/speech"]
        }
    ],
    "training_type": "val",
    "custom_metadata_module": "stable_audio_tools/configs/dataset_configs/custom_metadata/custom_md_example.py",
    "random_crop": True
}

val_dataloder = create_val_dataloader_from_config(
    dataset_config, 
    batch_size=batch_size,
    num_workers=8,
    sample_rate=model_config["sample_rate"],
    sample_size=sample_size,
    audio_channels=model_config.get("audio_channels", 1),
)

with torch.no_grad():
    for batch in tqdm(val_dataloder):
        # Generate stereo audio
        conditioning = batch[0]
        output = generate_diffusion_cond(
            model,
            steps=100,
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
            name = conditioning[i]['path']+'.wav'

            filename = f'recon/musiccaps/{name}'
            wavs = output[i,:,:]
            # wavs = wavs[:,:sample_rate*20]
            # try:
            torchaudio.save(filename, wavs, sample_rate)
            # except:
            #     filename = f'recon/musiccaps/{i}.wav'
            #     torchaudio.save(filename, wavs, sample_rate)

