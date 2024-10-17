import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 引数の受取
parser = argparse.ArgumentParser(description="HumanEval with vllm")
parser.add_argument("--input_file", default="./out/model_performance.csv")
args = parser.parse_args()

# CSVファイルを読み込む
df = pd.read_csv(args.input_file)
df = df.sort_values("pass@1", ascending=False)

# プロットの設定
plt.figure(figsize=(12, 6))

colors = sns.color_palette("hls", 2)

# バーチャートを作成
sns.barplot(
    x="model_name",
    y="value",
    hue="metric",
    data=pd.melt(
        df, id_vars=["model_name"], value_vars=["pass@1", "pass@10"], var_name="metric"
    ),
    palette=colors,
)

# グラフの装飾
plt.title("Performance of HumanEval-ja")
plt.xlabel("Model Name")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Metric")
plt.ylim(0.0, 1.0)

# グラフの余白を調整
plt.tight_layout()

# グラフを保存
out_filename = f"./out/model_performance_comparison.png"
plt.savefig(out_filename, dpi=300, bbox_inches="tight")

print(f"complete saving graph.\nsee {out_filename}")
