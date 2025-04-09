log_path=logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u run_grasp.py -g 0 > $log_path/grasp.log & echo $!
