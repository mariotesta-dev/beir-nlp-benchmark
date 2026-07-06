import argparse

from beir_benchmark.encoders import MODEL_CONFIGS
from beir_benchmark.run import run


def main() -> None:
    parser = argparse.ArgumentParser(description="run the BEIR retrieval benchmark")
    parser.add_argument("--dataset", default="nfcorpus", help="BEIR dataset name")
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_CONFIGS),
        choices=list(MODEL_CONFIGS),
        help="dense models to evaluate against bm25",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[10, 100],
        help="retrieval cutoffs, the largest one sets the search depth",
    )
    args = parser.parse_args()
    run(dataset=args.dataset, models=tuple(args.models), ks=tuple(args.ks))


if __name__ == "__main__":
    main()
