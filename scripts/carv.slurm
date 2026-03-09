#!/bin/bash

#SBATCH --job-name=carv    # 作业名
#SBATCH --partition=small               # cpu 队列
#SBATCH --nodes=1                         # 使用1个节点
#SBATCH --ntasks=1                        # 总任务数 1（Python主进程）
#SBATCH --cpus-per-task=64                # 每个任务分配63个CPU核心
#SBATCH --output=/dssg/home/acct-nanli/xxxxxxyz/IPO/logs/%x/%x_%j.out
#SBATCH --error=/dssg/home/acct-nanli/xxxxxxyz/IPO/logs/%x/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxxxxyz@sjtu.edu.cn

#

source ~/.bashrc
cd /dssg/home/acct-nanli/xxxxxxyz/IPO/
# python -u   industry-classify-cn/ind_ipo_pair.py
python -u  "ipo-roadshow-spillover-cn/carv/cal_car_cav.py" 