log_path=logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u run_grasp_cos.py -g 0 > $log_path/cos.log & echo $!
