#!/bin/bash

model_name="llm-jp/llm-jp-3-1.8b-instruct"
stem="${model_name##*/}"
output_file="./out/${stem}/samples_at_10.jsonl"

python generate_response.py --model_name $model_name --num_trial 10
evaluate_functional_correctness $output_file