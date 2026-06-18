python benchmark.py --scenario refactor --turns 20 --json > benchmark_refactor_20_6.json 2> benchmark_refactor_20_6.log              [8m45,6s]
python benchmark.py --scenario refactor --turns 30 --json > benchmark_refactor_30_6.json 2> benchmark_refactor_30_6.log             [14m54,8s]
python benchmark.py --scenario debug --turns 30 --json > benchmark_debug_30_6.json 2> benchmark_debug_30_6.log             [15m49,6s]
python benchmark.py --scenario feature --turns 30 --json > benchmark_feature_30_6.json 2> benchmark_feature_30_6.log            [14m50,6s]
python benchmark.py --scenario default --turns 30 --json > benchmark_default_30_6.json 2> benchmark_default_30_6.log
python benchmark.py --scenario all --turns 30 --json > benchmark_all_30_6.json 2> benchmark_all_30_6.log 