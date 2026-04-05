#!/bin/bash

#SBATCH --job-name=initialize             # 作业名
#SBATCH --partition=small                 # cpu 队列
#SBATCH --nodes=1                         # 单节点
#SBATCH --ntasks=1                        # 1个Python主进程
#SBATCH --cpus-per-task=64               # 视频处理并行度（PARALLEL_PROCESSES 读此值）
#SBATCH --output=/dssg/home/acct-nanli/xxxxxxyz/IPO/logs/%x/%x_%j.out
#SBATCH --error=/dssg/home/acct-nanli/xxxxxxyz/IPO/logs/%x/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxxxxyz@sjtu.edu.cn

source ~/.bashrc

cd /dssg/home/acct-nanli/xxxxxxyz/IPO/ipo-roadshow-spillover-cn

python -u -m analyze.initialize
