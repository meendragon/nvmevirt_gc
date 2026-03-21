#!/bin/bash
# =============================================================
# fio_write.sh - Multi-Phase Write Workload
#
# Phase 1: Uniform Random (60s)   → Greedy 유리
# Phase 2: Hot/Cold zipf (60s)    → CB 유리
# Phase 3: Sequential Burst (30s) → CAT 유리
# Phase 4: Hot Spot Shift (60s)   → RL만 적응
#
# 사용법: sudo bash fio_write.sh
# =============================================================

set -e

DIR="/home/meen/nvmevirt_gc/mnt"
RESULT_DIR="./results"
mkdir -p "${RESULT_DIR}"

echo "========================================="
echo " FIO Write Workload"
echo " Dir: ${DIR}"
echo "========================================="

# ---------------------------------------------------------
# Phase 0: 초기 채움 (GC 유발을 위해 공간 소진)
# 11.2GB 디바이스, ext4 오버헤드 고려 → 8G 채움
# ---------------------------------------------------------
echo ""
echo "[Phase 0] Initial fill..."

sudo fio --directory="${DIR}" \
    --direct=1 \
    --ioengine=libaio \
    --rw=write \
    --bs=128k \
    --size=8G \
    --numjobs=1 \
    --name=phase0_fill \
    --output="${RESULT_DIR}/phase0_fill.json" \
    --output-format=json

echo "[Phase 0] Fill done"
sleep 2

# ---------------------------------------------------------
# Phase 1: Uniform Random (60s) → Greedy 유리
#
# age 차이 없음 → CB/CAT의 age 계산이 무의미
# Greedy가 VPC 최소만 보고 빠르게 선택
# ---------------------------------------------------------
echo ""
echo "[Phase 1] Uniform random (Greedy territory)"

sudo fio --directory="${DIR}" \
    --direct=1 \
    --ioengine=libaio \
    --rw=randwrite \
    --bs=4k \
    --size=700M \
    --io_size=3G \
    --numjobs=1 \
    --iodepth=32 \
    --norandommap=1 \
    --randrepeat=0 \
    --time_based \
    --runtime=60 \
    --name=phase1_random \
    --output="${RESULT_DIR}/phase1_random.json" \
    --output-format=json

echo "[Phase 1] Done"
sleep 2

# ---------------------------------------------------------
# Phase 2: Hot/Cold Locality (60s) → CB 유리
#
# zipf:1.2 → 소수 LPN에 집중 → hot/cold 뚜렷
# CB가 age로 hot 보호, cold 블록 우선 청소
# Greedy는 hot 블록 건드려서 useless migration
# ---------------------------------------------------------
echo ""
echo "[Phase 2] Hot/Cold zipf locality (CB territory)"

sudo fio --directory="${DIR}" \
    --direct=1 \
    --ioengine=libaio \
    --rw=randwrite \
    --bs=4k \
    --size=700M \
    --io_size=3G \
    --numjobs=1 \
    --iodepth=32 \
    --norandommap=1 \
    --randrepeat=0 \
    --random_distribution=zipf:1.2 \
    --time_based \
    --runtime=60 \
    --name=phase2_hotcold \
    --output="${RESULT_DIR}/phase2_hotcold.json" \
    --output-format=json

echo "[Phase 2] Done"
sleep 2

# ---------------------------------------------------------
# Phase 3: Sequential Burst (30s) → CAT 유리
#
# 순차 쓰기로 기존 데이터 덮어씀 → 모든 블록 IPC 비슷해짐
# Greedy/CB는 아무 블록이나 고름 → erase 편중
# CAT는 erase_cnt 적은 블록 우선 → wear-leveling
# ---------------------------------------------------------
echo ""
echo "[Phase 3] Sequential burst (CAT territory)"

sudo fio --directory="${DIR}" \
    --direct=1 \
    --ioengine=libaio \
    --rw=write \
    --bs=128k \
    --size=700M \
    --io_size=2G \
    --numjobs=1 \
    --iodepth=32 \
    --time_based \
    --runtime=30 \
    --name=phase3_seq \
    --output="${RESULT_DIR}/phase3_sequential.json" \
    --output-format=json

echo "[Phase 3] Done"
sleep 2

# ---------------------------------------------------------
# Phase 4: Hot Spot Shift (60s) → RL만 적응
#
# zipf:1.2를 다른 파일에 적용 → hot 영역이 이동
# 정적 정책은 이전 hot/cold 분포로 잘못 판단
# RL은 state 변화 감지 → 파라미터 재조정
# ---------------------------------------------------------
echo ""
echo "[Phase 4] Hot spot shift (RL territory)"

# 다른 파일에 쓰기 → 이전 hot zone이 cold로 전환
sudo fio --directory="${DIR}" \
    --direct=1 \
    --ioengine=libaio \
    --rw=randwrite \
    --bs=4k \
    --size=700M \
    --io_size=3G \
    --numjobs=1 \
    --iodepth=32 \
    --norandommap=1 \
    --randrepeat=0 \
    --random_distribution=zipf:1.2 \
    --time_based \
    --runtime=60 \
    --name=phase4_shift \
    --output="${RESULT_DIR}/phase4_shift.json" \
    --output-format=json

echo "[Phase 4] Done"
sleep 1

sync
echo ""
echo "========================================="
echo " Write workload complete!"
echo "========================================="