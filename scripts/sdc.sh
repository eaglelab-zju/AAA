log_path=logs/
mkdir -p $log_path

OPENBLAS_NUM_THREADS=1 nohup python -u run_sdc.py  -g 1 > $log_path/sdc.log & echo $!
