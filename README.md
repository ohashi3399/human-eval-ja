# HumanEval-ja: Hand-Written Evaluation Set for Japanese

- This repository is a fork of [HumanEval: Hand-Written Evaluation Set](https://github.com/openai/human-eval) and an extention of Japanese language

## Features

- **support HumanEval-ja**
  - evaluate coding ability of your LLMs which can handle Japanese
  - support [HumanEval-ja](https://huggingface.co/datasets/HachiML/humaneval-ja-v0.6)

- **support vLLM**
  - reduce inference time via [vLLM](https://github.com/vllm-project/vllm) library

- **support apply_chat_templates function**
  - directly use the latest LLM without modifying any chat templates

## Environment setup

```sh
git clone https://github.com/ohashi3399/human-eval-ja.git && cd human-eval-ja
pip install -e .
```

### Caution referred by base repository
> **This program exists to run untrusted model-generated code. Users are strongly
> encouraged not to do so outside of a robust security sandbox. The [execution
> call](https://github.com/openai/human-eval/blob/master/human_eval/execution.py#L48-L58)
> in `execution.py` is deliberately commented out to ensure users read this
> disclaimer before running code in a potentially unsafe manner. See the comment in
> `execution.py` for more information and instructions.**

## How to use

1. You can change default sampling parameter from [here](https://github.com/ohashi3399/human-eval-ja/blob/master/generate_response.py#L14)
   - `temperature` and `top_p` were referred from the experiment of [WizardCoder](https://arxiv.org/abs/2306.08568) 
   - `max_tokens` and `repetition_penalty` were experimentally decided

```python
def create_sampling_params() -> SamplingParams:
    """サンプリングパラメータを設定する"""
    return SamplingParams(
        max_tokens=2048, temperature=0.2, top_p=0.95, repetition_penalty=1.05
    )
```

1. Edit `human-eval-ja.sh` like below.
- You can change just model_name or add other models

```sh
#!/bin/bash

model_name="llm-jp/llm-jp-3-1.8b-instruct"
stem="${model_name##*/}"
output_file="./out/${stem}/samples_at_10.jsonl"

python generate_response.py --model_name $model_name --num_trial 10
evaluate_functional_correctness $output_file

model_name="nvidia/Nemotron-Mini-4B-Instruct"
stem="${model_name##*/}"
output_file="./out/${stem}/samples_at_10.jsonl"

python generate_response.py --model_name $model_name --num_trial 10
evaluate_functional_correctness $output_file
```

- `generate_response.py` generates responses of HumanEval-ja
  - `model_name` stands for the model name that you want to evaluate
  - `num_trial` stands for how many times your model will generate responses

2. Run `human-eval-ja.sh`

```sh
source human-eval-ja.sh
```

3. You can see the result like this

```csv
model_name,pass@1,pass@10
llm-jp-3-1.8b-instruct,0.016463414634146342,0.07317073170731707
```

4. Aggregate evaluation summary

```sh
source summary.sh
```

- You can see performance graph like below

![chart](./models_performance.png)

## Citation

Please cite using the following bibtex entry:

```
@article{chen2021codex,
  title={Evaluating Large Language Models Trained on Code},
  author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
  year={2021},
  eprint={2107.03374},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
