import time

from beir_benchmark.data import load_beir_hf
from beir_benchmark.encoders import make_encoder, MODEL_CONFIGS
from beir_benchmark.retrievers import DenseRetriever, BM25Retriever
from beir_benchmark.metrics import evaluate_run

METRICS = ["ndcg@10", "recall@100", "mrr@10"]

def run(dataset: str = "nfcorpus",
        models: tuple[str, ...] = ("bge-small", "e5-small", "qwen3-0.6b"),
        ks: tuple[int, ...] = (10, 100)):
    ds = load_beir_hf(dataset)
    print(f"{dataset}: {len(ds.corpus)} docs, {len(ds.queries)} queries\n")

    systems: dict[str, object] = {"bm25": BM25Retriever()}
    for key in models:
        systems[key] = DenseRetriever(make_encoder(key))

    rows = []
    for name, retriever in systems.items():
        t0 = time.perf_counter()
        retriever.index(ds.corpus)
        t_index = time.perf_counter() - t0

        t0 = time.perf_counter()
        results = retriever.search(ds.queries, top_k=max(ks))
        t_search = time.perf_counter() - t0

        scores = evaluate_run(results, ds.qrels, METRICS)
        rows.append((name, scores, t_index, t_search))

    _print_table(rows, ks)


def _print_table(rows):
    header = f"{'system':<14}" + "".join(f"{m:>12}" for m in METRICS) + f"{'index(s)':>10}{'search(s)':>11}"
    print(header); print("-" * len(header))
    for name, scores, t_index, t_search in rows:
        line = f"{name:<14}" + "".join(f"{scores[m]:>12.4f}" for m in METRICS)
        print(line + f"{t_index:>10.1f}{t_search:>11.1f}")


if __name__ == "__main__":
    run()

