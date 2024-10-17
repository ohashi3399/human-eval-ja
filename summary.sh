#!/bin/bash

python aggregate_performance.py --out_dir ./out
python make_chart.py --input_file ./out/performance_table.csv