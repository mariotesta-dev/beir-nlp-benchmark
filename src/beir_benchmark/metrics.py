from ranx import Qrels, Run, evaluate

from beir_benchmark.retrievers import Results   # {qid: [(docid, score), ...]}


def _to_run_dict(results: Results) -> dict[str, dict[str, float]]:
    """ranx wants {qid: {docid: score}}, not lists of tuples."""
    return {
        qid: {doc_id: float(score) for doc_id, score in ranked}
        for qid, ranked in results.items()
    }


def evaluate_run(
    results: Results,
    qrels: dict[str, dict[str, int]],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    metrics = metrics or ["ndcg@10", "recall@100", "mrr@10"]
    scores = evaluate(Qrels(qrels), Run(_to_run_dict(results)), metrics)
    # evaluate() returns a bare float when given a single metric string; normalize.
    return scores if isinstance(scores, dict) else {metrics[0]: scores}