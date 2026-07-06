# beir-nlp-benchmark

A small benchmark that compares BM25 and dense retrievers on BEIR datasets.
It loads a dataset from Hugging Face, indexes the corpus with each system,
runs the queries and reports ndcg@10, recall@100 and mrr@10.

The dense models are defined in `src/beir_benchmark/encoders.py`:

- `bge-small` (BAAI/bge-small-en-v1.5, cls pooling)
- `e5-small` (intfloat/e5-small-v2, mean pooling)
- `qwen3-0.6b` (Qwen/Qwen3-Embedding-0.6B, last token pooling)

## Install

```bash
uv sync
```

## Run

Locally (uses the GPU if one is visible, otherwise the CPU):

```bash
uv run python tools/run_benchmark.py
```

Pick a dataset, a subset of models or different cutoffs:

```bash
uv run python tools/run_benchmark.py --dataset scifact --models bge-small e5-small --ks 10 100
```

## Example output

```
nfcorpus: 3633 docs, 323 queries

system             ndcg@10  recall@100      mrr@10  index(s)  search(s)
-----------------------------------------------------------------------
bm25                0.2669      0.2107      0.4667       0.3        0.7
bge-small           0.3435      0.3111      0.5284      52.6        0.8
e5-small            0.3247      0.2971      0.5206      51.2        0.3
qwen3-0.6b          0.3608      0.3416      0.5687     153.3        1.1
```
