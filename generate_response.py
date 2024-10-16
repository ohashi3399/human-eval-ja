import argparse
from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from typing import List, Dict


def create_llm(model_name) -> LLM:
    """LLMインスタンスを作成する"""
    return LLM(model_name, dtype="bfloat16", trust_remote_code=True)


def create_sampling_params() -> SamplingParams:
    """サンプリングパラメータを設定する"""
    return SamplingParams(
        max_tokens=2048, temperature=0.7, top_p=0.95, repetition_penalty=1.05
    )


def create_prompts(problems: Dict[str, Dict]) -> List[str]:
    """各問題に対するプロンプトを作成する"""
    system_message = (
        "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
    )
    return [
        f"{system_message}\\n\\n### 指示:\\n{problem['prompt']}\\n\\n### 応答:\\n"
        for problem in problems.values()
    ]


def extract_code(completion: str) -> str:
    """生成された完了テキストからPythonコードを抽出する"""
    code = completion.split("```python\n")[-1].split("```")[0]
    return code.strip()


def generate_completions(
    llm: LLM, prompts: List[str], sampling_params: SamplingParams
) -> List[str]:
    """バッチ処理でcompletionテキストを生成する"""
    outputs = llm.generate(prompts, sampling_params)
    return [extract_code(output.outputs[0].text) for output in outputs]


def main():

    # 引数の受取
    parser = argparse.ArgumentParser(description="HumanEval with vllm")
    parser.add_argument("--model_name", default="llm-jp/llm-jp-3-1.8b-instruct")
    parser.add_argument("--num_trial", type=int, default=1)
    args = parser.parse_args()

    # 問題を読み込む
    problems = read_problems()

    # LLMとサンプリングパラメータを作成
    llm = create_llm(args.model_name)
    sampling_params = create_sampling_params()

    # プロンプトを作成
    prompts = create_prompts(problems)

    # サンプルを作成
    total_samples = list()
    for _ in range(args.num_trial):

        # 完了テキストを生成
        completions = generate_completions(llm, prompts, sampling_params)

        samples = [
            {"task_id": task_id, "completion": completion}
            for (task_id, completion) in zip(problems.keys(), completions)
        ]
        total_samples.extend(samples)

    # 結果を書き込む
    write_jsonl(f"./samples_at_{args.num_trial}.jsonl", total_samples)


if __name__ == "__main__":
    main()
