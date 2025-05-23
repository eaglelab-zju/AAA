log_path=logs
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u ./baseline/run_baselines_cos.py -g 0 -t gat > $log_path/cos/cos_gat.log & echo $!
