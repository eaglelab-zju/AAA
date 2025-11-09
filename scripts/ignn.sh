log_path=logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u run_ignn.py -g 5 > $log_path/ignn.log & echo $!
