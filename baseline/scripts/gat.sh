log_path=./logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u ./baseline/run_gnn.py -t gat -g 0 > $log_path/baselines_gat.log & echo $!
