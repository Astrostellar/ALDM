#!/bin/bash

# 设置日志路径和时间戳
LOG_FILE="run_log.txt"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# 无限循环
while true; do
    echo "[$TIMESTAMP] Starting the Python script..." | tee -a "$LOG_FILE"
    fuser -vk /dev/nvidia*
    # 运行 Python 脚本
    python main.py -b configs/$1.yaml \
                   -t True \
                   --resume logs/$2/checkpoints/last.ckpt \
                   --gpus 0,1,2,3

    # 脚本异常退出后记录时间
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] Script crashed or exited. Restarting after 10 seconds..." | tee -a "$LOG_FILE"
    
    # 延迟重启时间
    sleep 10
done