#!/bin/bash -l

#SBATCH -J trainYolom
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1                # GPU assigned to each task
#SBATCH --cpus-per-gpu=7
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH --mail-type=all[,begin,end,fail]
sleep 5s

source ~/.bashrc
conda deactivate
conda activate yolov11

echo 'Environment activated'
python --version
pwd
python yolo_stream_m_35.py
