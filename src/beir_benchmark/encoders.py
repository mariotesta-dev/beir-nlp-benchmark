import torch
import torch.nn.functional as F

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from beir_benchmark.pooling import POOLING_FUNCTIONS

@dataclass(frozen=True)
class ModelConfig:
    hf_name: str
    pooling: str
    query_prefix: str = ""
    doc_prefix: str = ""
    normalize: bool = True
    max_length: int = 512

MODEL_CONFIGS : dict[str, ModelConfig] = {
    "bge-small": ModelConfig(
        hf_name="BAAI/bge-small-en-v1.5",
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "e5-small": ModelConfig(
        hf_name="intfloat/e5-small-v2",
        pooling="mean",
        query_prefix="query: ",
        doc_prefix="passage: ",
    ),
    "qwen3-0.6b": ModelConfig(
        hf_name="Qwen/Qwen3-Embedding-0.6B",
        pooling="last_token",
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: "
        ),
    ),
}

class DenseEncoder:
    def __init__(self, config: ModelConfig, device: str | None = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.hf_name)
        self.model = AutoModel.from_pretrained(config.hf_name).to(self.device)
        self.pool = POOLING_FUNCTIONS[config.pooling]

    @torch.no_grad()
    def encode(self, texts: list[str], prefix: str = "", batch_size: int = 32) -> torch.Tensor:
        """
        Encode a list of texts into dense embeddings.
        """
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[start:start + batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            emb = self.pool(out.last_hidden_state, enc['attention_mask'])

            if self.config.normalize:
                emb = F.normalize(emb, p=2, dim=1)
            
            chunks.append(emb.cpu())
        return torch.stack(chunks, dim=0)
    

# Factory
def make_encoder(key: str) -> DenseEncoder:
    if key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model key: {key}. Available keys: {list(MODEL_CONFIGS.keys())}")
    return DenseEncoder(MODEL_CONFIGS[key])