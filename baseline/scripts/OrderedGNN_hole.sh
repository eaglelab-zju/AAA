log_path=./logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u ./baseline/run_gnn_hole.py -t OrderedGNN_hole -g 0 > $log_path/baselines_OrderedGNN_hole.log & echo $!
