import importlib
import numpy as np
import io
import os
import posixpath
import random
import re
import subprocess
import time
import torch
import torchaudio
import webdataset as wds
import json
from aeiou.core import is_silence
from os import path
from pedalboard.io import AudioFile
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py
def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4):

    dataset_type = dataset_config.get("dataset_type", None)
    training_type = dataset_config.get("training_type", None)
    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = "stereo"

    if dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        training_dirs = []

        custom_metadata_fn = None
        custom_metadata_module_path = dataset_config.get("custom_metadata_module", None)

        if custom_metadata_module_path is not None:
            spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)                

            custom_metadata_fn = metadata_module.get_custom_metadata

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"
            training_dirs.append(audio_dir_path)

        train_set = SampleDataset(
            training_dirs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels,
            custom_metadata_fn=custom_metadata_fn,
            relpath= None, #training_dirs[0] #TODO: Make relpath relative to each training dir
            training_type =training_type
        )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)


def create_val_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4):

    dataset_type = dataset_config.get("dataset_type", None)
    training_type = dataset_config.get("training_type", None)
    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = "stereo"

    if dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        val_dirs = []

        custom_metadata_fn = None
        custom_metadata_module_path = dataset_config.get("custom_metadata_module", None)

        if custom_metadata_module_path is not None:
            spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)                

            custom_metadata_fn = metadata_module.get_custom_metadata

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"
            val_dirs.append(audio_dir_path)

        val_set = SampleDataset(
            val_dirs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", False),
            force_channels=force_channels,
            custom_metadata_fn=custom_metadata_fn,
            relpath= None, #training_dirs[0] 
            training_type ="val"

        )

        return torch.utils.data.DataLoader(val_set, batch_size, shuffle=False,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)


def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths[0]:               # get a list of relevant filenames
        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        paths, 
        sample_size=65536, 
        sample_rate=44100, 
        keywords=None, 
        relpath=None, 
        random_crop=True,
        force_channels="stereo",
        custom_metadata_fn = None,
        training_type =None,
    ):
        super().__init__()
        self.filenames = []
        self.relpath = relpath
        self.training_type = training_type
        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )
        ######validaton
        if self.training_type =="val":
            # self.filenames = get_audio_filenames(paths, keywords)
            # speech, music, sound =[],[],[]
            # with open("data/test/speech.txt") as f:
            #     for item in f:
            #         speech.append(item.strip())
            # with open("data/test/music.txt") as f:
            #     for item in f:
            #         music.append(item.strip())
            # with open("data/test/kwai1.txt") as f:
            #     for item in f:
            #         sound.append(item.strip())
            # self.filenames = sound
            json_sound = "data/audiocaps/json_files/test.json"

            with open(json_sound, 'r') as f:
                json_obj = json.load(f)
            self.filenames = [item["location"] for item in json_obj]
            self.prompts = [item["caption"] for item in json_obj]
            print(f'Found validation {len(self.filenames)} files')

        ######train
        elif self.training_type =="encodec":
            speech, music, sound, kwai =[],[],[],[]
            with open("data/encodec/speech.txt") as f:
                for item in f:
                    speech.append(item.strip())
            with open("data/encodec/music.txt") as f:
                for item in f:
                    music.append(item.strip())
            with open("data/encodec/sound.txt") as f:
                for item in f:
                    sound.append(item.strip())
            with open("data/encodec/kwai.txt") as f:
                for item in f:
                    kwai.append(item.strip())
            # speech.extend(music)
            # speech.extend(sound)
            
            self.filenames = kwai
            print(f'Found {len(self.filenames)} files about speech, music, sound')

        #######text-caption paired
        elif self.training_type =="diffusion":

            json_MUmusic = "data/MUCaps/train.json"
            json_mtg = 'data/mtg/mtg_tag_filtered.json'
            json_bgm = 'data/kwai_music/bgm_with_tempo_text_A.json'

            json_sound = "data/audiocaps/json_files/train.json"

            with open(json_mtg, 'r') as f:
                json_1 = json.load(f)
            with open(json_bgm, 'r') as f:
                json_2 = json.load(f)
            with open(json_MUmusic, 'r') as f:
                json_3 = json.load(f)
            json_obj = json_1+json_2
            self.prompts = [item["caption"] for item in json_obj]
            self.filenames = [item["location"] for item in json_obj]
            print(f'Found {len(self.filenames)} files about music-text paired')


        self.sr = sample_rate

        self.custom_metadata_fn = custom_metadata_fn

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)
            #torch.Size([1, 441000])
            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info["path"] = audio_filename

            if self.relpath is not None:
                for base_path in self.relpath:
                    if audio_filename.startswith(base_path):
                        info["relpath"] = path.relpath(audio_filename, base_path)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask

            end_time = time.time()

            info["load_time"] = end_time - start_time
            
            # if self.training_type =="diffusion":
            info["prompt"] = self.prompts[idx]

            if self.training_type =="val":
                info ={}
                info["prompt"] = self.prompts[idx]
                info["seconds_start"] = 0
                info["seconds_total"] = 10
                return (audio, info)
            # if self.custom_metadata_fn is not None:
            #     custom_metadata = self.custom_metadata_fn(info, audio)
            #     info.update(custom_metadata)

            #     if "__reject__" in info and info["__reject__"]:
            #         return self[random.randrange(len(self))]

            return (audio, info)
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self[random.randrange(len(self))]

