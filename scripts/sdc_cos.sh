log_path=logs/
mkdir -p $log_path

HF_ENDPOINT=https://hf-mirror.com  nohup python -u run_sdc_cos.py -g 0 > $log_path/sdc_cos.log & echo $!
