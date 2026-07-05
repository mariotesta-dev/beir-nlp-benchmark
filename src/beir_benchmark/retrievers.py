import numpy as np
import torch

from typing import Protocol
from rank_bm25 import BM25Okapi

from beir_benchmark.encoders import DenseEncoder

Results = dict[str, list[tuple[str, float]]]

class Retriever(Protocol):
    def index(self, corpus: dict[str, str]) -> None:
        """
        Index the corpus for retrieval.
        """
        ...
    
    def search(self, queries: dict[str, str], top_k: int = 10) -> Results:
        """
        Search the corpus for the given queries and return the top_k results.
        """
        ...

class DenseRetriever:
    def __init__(self, encoder: DenseEncoder):
        self.encoder = encoder
        self.doc_ids : list[str] = []
        self.doc_embs : torch.Tensor | None = None

    def index(self, corpus: dict[str, str]) -> None:
        """
        Encode the corpus into dense embeddings and store them for retrieval.
        """
        self.doc_ids = list(corpus.keys())
        texts = [corpus[doc_id] for doc_id in self.doc_ids]
        self.doc_embs = self.encoder.encode(texts, prefix=self.encoder.config.doc_prefix)

    def search(self, queries: dict[str, str], top_k: int = 10) -> Results:
        """
        Search the corpus for the given queries and return the top_k results.
        """
        q_ids = list(queries.keys())
        q_emb = self.encoder.encode(
            [queries[q_id] for q_id in q_ids],
            prefix=self.encoder.config.query_prefix,
        )
        scores = q_emb @ self.doc_embs.T
        k = min(top_k, scores.shape[1])
        top_vals, top_idx = torch.topk(scores, k=k, dim=1)

        results : Results = {}
        for row, q_id in enumerate(q_ids):
            results[q_id] = [
                (self.doc_ids[col], top_vals[row, rank].item())
                for rank, col in enumerate(top_idx[row].tolist())
            ]

        return results


class BM25Retriever:
    def __init__(self):
        self.doc_ids : list[str] = []
        self.bm25 : BM25Okapi | None = None

    def index(self, corpus: dict[str, str]) -> None:
        self.doc_ids = list(corpus.keys())
        tokenized = [corpus[d].lower().split() for d in self.doc_ids]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, queries: dict[str, str], top_k: int = 10) -> Results:
        results : Results = {}
        for q_id, q_text in queries.items():
            scores = self.bm25.get_scores(q_text.lower().split())
            top_idx = np.argsort(scores)[::-1][:top_k]
            results[q_id] = [(self.doc_ids[i], float(scores[i])) for i in top_idx]
        return results