#!/bin/bash
OMP_NUM_THREADS=10
export OMP_NUM_THREADS
#

DIRNAME=`date '+%Y-%m-%d-%H-%M-%S'`
LOGPATH="./log/$DIRNAME"
LOGFILE="$LOGPATH/log.txt"
if [ ! -d "$LOGPATH" ];then
mkdir "$LOGPATH"
fi


CUDA_VISIBLE_DEVICES=0 python -u ./main.py\
                    --epoch 10000\
                    --batchSize 128\
                    --lr 1e-3\
                    --batchModelSave 1000000\
                    --logPath "$LOGPATH"\
                    --checkPoint ""
