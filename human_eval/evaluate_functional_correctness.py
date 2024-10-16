import fire
import sys

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    model_name: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file
    )
    print(results)

    record = list()
    record.append("model_name,pass@1,pass@10")
    stem = model_name.split("/")[-1]
    pass_at_1 = results["pass@1"]
    pass_at_10 = results["pass@10"]
    record.append(f"{stem},{pass_at_1},{pass_at_10}")
    with open(f"record-{stem}.csv", mode="w", encoding="utf-8") as o:
        o.write("\n".join(record))


def main():
    fire.Fire(entry_point)


sys.exit(main())
