import os
import argparse


def find_performance_csv(out_dir):
    csv_paths = []

    # os.walkを使ってディレクトリを再帰的に探索
    for dirpath, dirnames, filenames in os.walk(out_dir):
        if "performance.csv" in filenames:
            csv_paths.append(os.path.join(dirpath, "performance.csv"))

    return csv_paths


# 引数の受取
parser = argparse.ArgumentParser(description="HumanEval with vllm")
parser.add_argument("--out_dir", default="./out")
args = parser.parse_args()

performance_csv_paths = find_performance_csv(args.out_dir)

table = list()
table.append("model_name,pass@1,pass@10")
for path in performance_csv_paths:

    with open(path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        model_name, pass_at_1, pass_at_10 = lines[-1].split(",")
        table.append(f"{model_name},{pass_at_1},{pass_at_10}")

with open("./out/performance_table.csv", mode="w", encoding="utf-8") as o:
    o.write("\n".join(table))
