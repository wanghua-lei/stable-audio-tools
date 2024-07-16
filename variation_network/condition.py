import torch
import logging, warnings
import string
import typing as tp
import gc
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
        attention_mask = encoded["attention_mask"].to(torch.bool).to(device)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    

        embeddings = self.proj_out(embeddings.float())

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask

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
    def __init__(self, hidden_size, num_layers, num_query_tokens=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens

        self.encoder = T5Conditioner(output_dim=hidden_size, t5_model_name="t5-base", max_length=128, enable_grad=False, project_out=True)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model=hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu')
            for _ in range(num_layers)
        ])
        self.learned_queries = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))

    def forward(self, texts: tp.List[str], device: torch.device):
        text_embeddings, attention_mask = self.encoder(texts, device)

        # Repeat learned queries for the batch
        learned_queries = self.learned_queries.expand(text_embeddings.size(0), -1, -1)  # (B, num_query_tokens, hidden_size)

        # Apply cross attention layers
        for layer in self.cross_attention_layers:
            learned_queries = layer(learned_queries, text_embeddings, memory_key_padding_mask=attention_mask)

        return learned_queries

class QFormerAudio(nn.Module):
    def __init__(self, hidden_size, num_layers, num_query_tokens=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens

        self.encoder = MERTConditioner(output_dim=hidden_size, model_name="MERT-v1-95M")
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model=hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu')
            for _ in range(num_layers)
        ])
        self.learned_queries = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))

    def forward(self, audios: tp.List[torch.Tensor], device: torch.device):
        audio_embeddings, attention_mask = self.encoder(audios, device)

        # Repeat learned queries for the batch
        learned_queries = self.learned_queries.expand(audio_embeddings.size(0), -1, -1)  # (B, num_query_tokens, hidden_size)

        # Apply cross attention layers
        for layer in self.cross_attention_layers:
            learned_queries = layer(learned_queries, audio_embeddings, memory_key_padding_mask=attention_mask)

        return learned_queries


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = ["A blender in action.", "A car driving on the road."]
    model = QFormerText(hidden_size=768, num_layers=6).to(device)
    output = model(texts, device)
    print(output.shape)  # Should be (batch_size, num_query_tokens, hidden_size)

    audios = [torch.randn(24000)]  # Example audio tensor with 24000 samples
    model = QFormerAudio(hidden_size=768, num_layers=6).to(device)
    output = model(audios, device)
    print(output.shape)  # Should be (batch_size, num_query_tokens, hidden_size)






