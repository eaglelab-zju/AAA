log_path=./logs/
mkdir -p $log_path

CUDA_LAUNCH_BLOCKING=1 HF_ENDPOINT=https://hf-mirror.com nohup python -u ./baseline/run_gnn_hole.py -t gcn_hole -g 0 > $log_path/baselines_gcn_hole.log & echo $!
