#!/bin/bash
# =============================================================
# fio_read.sh - Read Performance Measurement
#
# Test 1: Random read 4K        → IOPS, p99 latency
# Test 2: Sequential read 128K  → bandwidth
# Test 3: Mixed R/W 70/30 (60s) → GC 간섭 하 read 성능
#
# 사용법: sudo bash fio_read.sh
# =============================================================

set -e

DIR="/home/meen/nvmevirt_gc/mnt"
RESULT_DIR="./results"
mkdir -p "${RESULT_DIR}"

echo "========================================="
echo " FIO Read Workload"
echo " Dir: ${DIR}"
echo "========================================="

# ---------------------------------------------------------
# Test 1: Random Read 4K (30s)
# ---------------------------------------------------------
echo ""
echo "[Test 1] Random Read 4K"

sudo fio --directory="${DIR}" \
    --direct=1 \
    --ioengine=libaio \
    --rw=randread \
    --bs=4k \
    --size=256M \
    --numjobs=4 \
    --iodepth=32 \
    --time_based \
    --runtime=30 \
    --group_reporting \
    --lat_percentiles=1 \
    --percentile_list=50:90:95:99:99.9:99.99 \
    --name=read_rand4k \
    --output="${RESULT_DIR}/read_rand4k.json" \
    --output-format=json+

echo "[Test 1] Done"
sleep 2

# ---------------------------------------------------------
# Test 2: Sequential Read 128K (30s)
# ---------------------------------------------------------
echo ""
echo "[Test 2] Sequential Read 128K"

sudo fio --directory="${DIR}" \
    --direct=1 \
    --ioengine=libaio \
    --rw=read \
    --bs=128k \
    --size=256M \
    --numjobs=1 \
    --iodepth=32 \
    --time_based \
    --runtime=30 \
    --name=read_seq128k \
    --output="${RESULT_DIR}/read_seq128k.json" \
    --output-format=json+

echo "[Test 2] Done"
sleep 2

# ---------------------------------------------------------
# Test 3: Mixed Random R/W 70/30 (60s)
#
# GC 간섭 하 read 성능. RL의 핵심 차별점.
# 30% write가 GC를 유발하면서 70% read 동시 수행.
# ---------------------------------------------------------
echo ""
echo "[Test 3] Mixed R/W 70/30"

sudo fio --directory="${DIR}" \
    --direct=1 \
    --ioengine=libaio \
    --rw=randrw \
    --rwmixread=70 \
    --bs=4k \
    --size=700M \
    --numjobs=4 \
    --iodepth=32 \
    --norandommap=1 \
    --time_based \
    --runtime=60 \
    --group_reporting \
    --lat_percentiles=1 \
    --percentile_list=50:90:95:99:99.9:99.99 \
    --name=read_mixed \
    --output="${RESULT_DIR}/read_mixed.json" \
    --output-format=json+

echo "[Test 3] Done"

sync
echo ""
echo "========================================="
echo " Read workload complete!"
echo "========================================="