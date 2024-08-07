#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc

from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..training.utils import copy_state_dict
from .utils import load_ckpt_state_dict
import torch.nn.functional as F
from torch import nn

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False,
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    
class IntConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
            
            #self.int_embedder.to(device)
    
            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)
    
            int_embeds = self.int_embedder(ints).unsqueeze(1)
    
            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
    
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)
    
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class CLAPTextConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix, device=device)
            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class CLAPAudioConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Any = "cuda") -> tp.Any:

        self.model.to(device)
        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]

class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)
        
        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
            finally:
                logging.disable(previous_level)
            
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    

        embeddings = self.proj_out(embeddings.float())

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask


class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from g2p_en import G2p

        self.max_length = max_length

        self.g2p = G2p()

        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts] # shape [batch_size, length]
        
        phoneme_ignore = [" ", *string.punctuation]

        # Remove ignored phonemes and cut to max length
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]

        # Convert to ids
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]

        #Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]
        
        phoneme_ids = torch.tensor(phoneme_ids).to(device)

        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)
        
        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)
  
class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            tokenizer_name: str, # Name of a tokenizer from the Hugging Face transformers library
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from transformers import AutoTokenizer

         # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            finally:
                logging.disable(previous_level)

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
    
        embeddings = self.token_embedder(input_ids)
            
        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask

class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int):
        super().__init__(pretransform.encoded_channels, output_dim)

        self.pretransform = pretransform

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.cat(audio, dim=0)

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)
        
        latents = self.pretransform.encode(audio)

        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]

class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_inputs.append(x[condition_key][0])
                elif condition_key=="promptaudio" and isinstance(x[condition_key], str):
                    from pedalboard.io import AudioFile
                    import torchaudio.transforms as T
                    import torch.nn.functional as F
                    import random
                    with AudioFile(x[condition_key]) as f:
                        waveform = f.read(f.frames)
                        waveform = torch.from_numpy(waveform)
                        sr = f.samplerate
                    if waveform.shape[0] > 1:
                        waveform = (waveform[0] + waveform[1]) / 2
                    else:
                        waveform = waveform.squeeze(0)
                    
                    if sr != 24000:
                        resampler = T.Resample(sr, 24000)
                        waveform = resampler(waveform)
                    max_length = 24000*10
                    if waveform.shape[-1] > max_length:
                        max_start = waveform.shape[-1] - max_length
                        start = random.randint(0, max_start)
                        waveform = waveform[start: start + max_length]
                    else:
                        waveform=F.pad(waveform, (0, max_length - waveform.shape[-1]), "constant", 0.0)
                    conditioner_inputs.append(waveform)
                else:
                    conditioner_inputs.append(x[condition_key])
            
            output[key] = conditioner(conditioner_inputs, device)

        return output

class MERTConditioner(Conditioner):
    """
    A conditioner that turns music into semantic token and embeds them

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            model_name: str = 'MERT-v1-95M',
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoModel
        
        # loading our model weights
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # loading the corresponding preprocessor config
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        
    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Any = "cuda") -> tp.Any:
        
        self.model.to(device)
        self.proj_out.to(device)
        
        inputs = self.processor(audios, sampling_rate=self.processor.sampling_rate, return_tensors="pt")
        inputs['input_values'] = inputs['input_values'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs = inputs.to(device)
        with torch.cuda.amp.autocast(enabled=False):
            outputs = self.model(**inputs, output_hidden_states=False)

        return [outputs.last_hidden_state, torch.ones(outputs.last_hidden_state.shape[0], outputs.last_hidden_state.shape[1]).to(device)]

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        # Self-attention layer
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention layer
        tgt2 = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward layer
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class QFormerText(nn.Module):
    def __init__(self, output_dim, num_layers, num_query_tokens=8):
        super().__init__()
        self.output_dim = output_dim
        self.num_query_tokens = num_query_tokens

        self.encoder = T5Conditioner(output_dim = output_dim, t5_model_name="t5-base", max_length=128, enable_grad=False, project_out=False)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model=output_dim, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu')
            for _ in range(num_layers)
        ])
        self.learned_queries = nn.Parameter(torch.randn(1, num_query_tokens, output_dim))

    def forward(self, texts: tp.List[str], device: torch.device):
        text_embeddings, attention_mask = self.encoder(texts, device)

        # Repeat learned queries for the batch
        learned_queries = self.learned_queries.expand(text_embeddings.size(0), -1, -1)  # (B, num_query_tokens, output_dim)

        # Apply cross attention layers
        for layer in self.cross_attention_layers:
            learned_queries = layer(learned_queries, text_embeddings, memory_key_padding_mask=attention_mask)

        return [learned_queries, torch.ones(learned_queries.shape[0], self.num_query_tokens).to(device)]

class QFormerAudio(nn.Module):
    def __init__(self, output_dim, num_layers, num_query_tokens=8):
        super().__init__()
        self.output_dim = output_dim
        self.num_query_tokens = num_query_tokens

        self.encoder = MERTConditioner(output_dim=output_dim, model_name="MERT-v1-95M")
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model=output_dim, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu')
            for _ in range(num_layers)
        ])
        self.learned_queries = nn.Parameter(torch.randn(1, num_query_tokens, output_dim))

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: torch.device):
        audios = torch.stack(audios)
        audio_embeddings, attention_mask = self.encoder(audios, device)

        # Repeat learned queries for the batch
        learned_queries = self.learned_queries.expand(audio_embeddings.size(0), -1, -1)  # (B, num_query_tokens, output_dim)

        # Apply cross attention layers
        for layer in self.cross_attention_layers:
            learned_queries = layer(learned_queries, audio_embeddings, memory_key_padding_mask=attention_mask)

        return [learned_queries, torch.ones(learned_queries.shape[0], self.num_query_tokens).to(device)]
class ImageProjection(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 4,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(image_embed_dim, cross_attention_dim) #self.num_image_text_embeds *
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor):
        batch_size = image_embeds.shape[0]

        # image
        image_embeds = self.image_embeds(image_embeds)
        image_embeds = image_embeds.reshape(batch_size, -1, self.cross_attention_dim) #self.num_image_text_embeds
        image_embeds = self.norm(image_embeds)
        return image_embeds

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
class Video(Conditioner):
    def __init__(
            self,
            dim: int=1024,
            output_dim: int=768,
            name: str="h94/IP-Adapter",
            enable_grad: bool = False,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)

        self.dim = dim
        self.output_dim = output_dim
        self.enable_grad = enable_grad

        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder")
        self.image_encoder.requires_grad_(False).to(torch.float16)
        self.encoder_hid_proj_ip = ImageProjection(
                image_embed_dim=dim,
                cross_attention_dim=output_dim
            )

    def forward(self, video_batch, device=None):
        self.image_encoder.to(device)
        self.encoder_hid_proj_ip.to(device)
        batch_size = len(video_batch)
        frames = torch.cat(video_batch)
        if torch.all(frames == 0):
            image_embeddings = torch.zeros((batch_size, 32, self.output_dim)).to(device)
        else:
            images = self.image_processor(images=frames, return_tensors="pt").to(device) #32, 1024 .to(torch.float32)
            with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
                image_embeddings = self.image_encoder(**images).image_embeds #4, 1024

            image_embeddings = image_embeddings.reshape(batch_size, -1, self.dim).to(device)
            # image_embeddings = torch.mean(image_embeddings, dim=1, keepdim=True) #mean-> 1,1024
            image_embeddings = self.encoder_hid_proj_ip(image_embeddings) ##bz, 4, 768

        return [image_embeddings, torch.ones(image_embeddings.shape[0], image_embeddings.shape[1]).to(device)]
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}

        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "qaudio":
            conditioners[id] = QFormerAudio(**conditioner_config)
        elif conditioner_type == "qtext":
            conditioners[id] = QFormerText(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(pretransform, **conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)
