#!/bin/bash

MEM_START="4G"    
MEM_SIZE="2048M"   
CPUS="1,2"

# 1. 기존에 모듈이 떠있는지 확인하고 제거
if lsmod | grep -q "nvmev"; then
    sudo rmmod nvmev
    sleep 1

fi
# 2. 모듈 삽입
MODULE_PATH="/home/meen/nvmevirt_gc/nvmev.ko"

CMD="sudo insmod $MODULE_PATH memmap_start=$MEM_START memmap_size=$MEM_SIZE cpus=$CPUS"
echo "   Command: $CMD"
$CMD
