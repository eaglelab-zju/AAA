log_path=./logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u ./baseline/run_gnn.py -t graphSAGE -g 0 > $log_path/baselines_graphSAGE.log & echo $!
