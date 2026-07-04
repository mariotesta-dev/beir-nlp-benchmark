from dataclasses import dataclass
from datasets import load_dataset

@dataclass(frozen=True)
class RetrievalDataset:
    corpus: dict[str, str] # doc_id -> document text
    queries: dict[str, str] # query_id -> query text
    qrels: dict[str, dict[str, int]] # query_id -> {doc_id -> relevance score}

def load_beir_hf(name: str, split: str = "test") -> RetrievalDataset:
    """
    Load any BEIR dataset from HF into a RetrievalDataset object.
    """
    dataset_name = f"BeIR/{name}"

    corpus_ds = load_dataset(dataset_name, "corpus")["corpus"]
    queries_ds = load_dataset(dataset_name, "queries")["queries"]
    qrels_ds = load_dataset(f"{dataset_name}-qrels")[split]

    # concatenate title and text fields for corpus
    corpus = {
        doc["_id"]: f"{doc['title']} {doc['text']}".strip() 
        for doc in corpus_ds
    }

    qrels : dict[str, dict[str, int]] = {}
    for qrel in qrels_ds:
        qid, doc_id, score = str(qrel["query-id"]), str(qrel["corpus-id"]), int(qrel["score"])
        qrels.setdefault(qid, {})[doc_id] = score

    # only keep queries that actually have qrels 
    queries = {query["_id"]: query["text"] for query in queries_ds if query["_id"] in qrels}

    return RetrievalDataset(corpus=corpus, queries=queries, qrels=qrels)

if __name__ == "__main__":
    ds = load_beir_hf("nfcorpus")
    print(len(ds.corpus), len(ds.queries), len(ds.qrels))