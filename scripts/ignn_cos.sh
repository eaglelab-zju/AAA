log_path=logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u run_ignn_cos.py -g 0 > $log_path/ignn_cos.log & echo $!
