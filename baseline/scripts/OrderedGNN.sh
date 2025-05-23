log_path=./logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u ./baseline/run_gnn.py -t OrderedGNN -g 1 > $log_path/baselines_OrderedGNN.log & echo $!
