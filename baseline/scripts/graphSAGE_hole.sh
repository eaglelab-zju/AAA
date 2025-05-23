log_path=./logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u ./baseline/run_gnn_hole.py -t graphSAGE_hole -g 0 > $log_path/baselines_graphSAGE_hole.log & echo $!
