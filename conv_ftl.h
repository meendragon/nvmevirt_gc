// SPDX-License-Identifier: GPL-2.0-only

#ifndef _NVMEVIRT_CONV_FTL_H
#define _NVMEVIRT_CONV_FTL_H

#include <linux/types.h>
#include "pqueue/pqueue.h"
#include "ssd_config.h"
#include "ssd.h"

struct conv_ftl;
typedef struct line *(*victim_select_fn)(struct conv_ftl *, bool);

/* ============================================================
 * GC 모드 정의 (insmod gc_mode=0/1/2/3)
 *
 *  0: Greedy          — VPC 최소 선택
 *  1: Cost-Benefit    — Age×IPC / VPC
 *  2: CAT             — Age×IPC / (VPC × EraseCount)
 *  3: RL              — 통합 공식 + Q-table 가중치 자동 튜닝
 * ============================================================ */
#define GC_MODE_GREEDY  0
#define GC_MODE_CB      1
#define GC_MODE_CAT     2
#define GC_MODE_RL      3

/* GC hot/cold write pointer 내부 라우팅 (nand_cmd.type과 별개) */
#define GC_IO_HOT   10
#define GC_IO_COLD  11

/* ============================================================
 * RL State/Action 공간 정의
 *
 * State (54 = 3×3×3×2):
 *   S1: free_line_ratio      3단계 (여유/보통/긴급)
 *   S2: avg_victim_vpc_ratio  3단계 (더러움/보통/깨끗)
 *   S3: hot_cold_ratio        3단계 (cold위주/혼합/hot위주)
 *   S4: wear_variance         2단계 (균등/불균등)
 *
 * Action (36 = 3×4×3):
 *   A1: alpha_level  3단계 — age_weight 지수 (0.5/1.0/2.0)
 *   A2: delta_level  4단계 — erase_cnt 지수 (0/0.5/1.0/1.5)
 *   A3: hot_thresh   3단계 — hot/cold 분리 기준 (avg×0.5/1.0/1.5)
 * ============================================================ */
#define RL_NUM_S1       3
#define RL_NUM_S2       3
#define RL_NUM_S3       3
#define RL_NUM_S4       2
#define RL_NUM_STATES   (RL_NUM_S1 * RL_NUM_S2 * RL_NUM_S3 * RL_NUM_S4) /* 54 */

#define RL_NUM_A1       3
#define RL_NUM_A2       4
#define RL_NUM_A3       3
#define RL_NUM_ACTIONS  (RL_NUM_A1 * RL_NUM_A2 * RL_NUM_A3) /* 36 */

/* Q-table 고정소수점 스케일 (×1000) */
#define RL_Q_SCALE      1000

/* RL 하이퍼파라미터 (×1000 고정소수점) */
#define RL_ALPHA        100   /* learning rate = 0.1 */
#define RL_GAMMA        900   /* discount factor = 0.9 */
#define RL_EPSILON_INIT 300   /* 초기 exploration rate = 0.3 */
#define RL_EPSILON_MIN  50    /* 최소 exploration rate = 0.05 */
#define RL_EPSILON_DECAY 999  /* epsilon *= 0.999 per episode */

struct rl_config {
	/* Q-table: q[state][action], 고정소수점 ×RL_Q_SCALE */
	int64_t q_table[RL_NUM_STATES][RL_NUM_ACTIONS];

	/* 현재 에피소드 상태 */
	uint32_t cur_state;
	uint32_t cur_action;

	/* 현재 action이 의미하는 파라미터 레벨 */
	uint32_t alpha_level;   /* 0,1,2 */
	uint32_t delta_level;   /* 0,1,2,3 */
	uint32_t hot_thresh_level; /* 0,1,2 */

	/* Exploration rate (×1000 고정소수점) */
	uint32_t epsilon;

	/* Reward 계산용: GC 직전 스냅샷 */
	uint64_t prev_copied_pages;
	uint32_t prev_erase_max;
	uint32_t prev_erase_min;

	/* 통계 */
	uint64_t total_episodes;
	uint64_t total_reward; /* 누적 reward ×1000 */
};

/* ============================================================
 * per-LPN 메타데이터 (Hot/Cold 판별용)
 * ============================================================ */
struct page_meta {
	uint32_t update_cnt;      /* 덮어쓰기 누적 횟수 */
	uint64_t last_write_time; /* 마지막 쓰기 시각 (ns) */
};

struct convparams {
	uint32_t gc_thres_lines;
	uint32_t gc_thres_lines_high;
	bool enable_gc_delay;
	double op_area_pcent;
	int pba_pcent;
};

struct line {
	int id;
	int ipc;
	int vpc;
	struct list_head entry;
	size_t pos;
	uint64_t last_modified_time;
	uint32_t erase_cnt;
};

struct write_pointer {
	struct line *curline;
	uint32_t ch;
	uint32_t lun;
	uint32_t pg;
	uint32_t blk;
	uint32_t pl;
};

struct line_mgmt {
	struct line *lines;
	struct list_head free_line_list;
	pqueue_t *victim_line_pq;
	struct list_head full_line_list;

	uint32_t tt_lines;
	uint32_t free_line_cnt;
	uint32_t victim_line_cnt;
	uint32_t full_line_cnt;
};

struct write_flow_control {
	uint32_t write_credits;
	uint32_t credits_to_refill;
};

struct conv_ftl {
	struct ssd *ssd;
	struct convparams cp;
	struct ppa *maptbl;
	uint64_t *rmap;

	/* Write Pointers: user / gc_hot / gc_cold (3 open blocks) */
	struct write_pointer wp;
	struct write_pointer gc_wp_hot;
	struct write_pointer gc_wp_cold;

	struct line_mgmt lm;
	struct write_flow_control wfc;

	/* Hot/Cold 판별 */
	struct page_meta *page_meta;
	uint64_t avg_hot_degree; /* EMA ×16 고정소수점 */

	/* RL 에이전트 (gc_mode == GC_MODE_RL 일 때만 활성) */
	struct rl_config rl;

	/* 통계 */
	uint64_t gc_count;
	uint64_t gc_copied_pages;
};

void conv_init_namespace(struct nvmev_ns *ns, uint32_t id, uint64_t size,
			 void *mapped_addr, uint32_t cpu_nr_dispatcher);
void conv_remove_namespace(struct nvmev_ns *ns);
bool conv_proc_nvme_io_cmd(struct nvmev_ns *ns, struct nvmev_request *req,
			   struct nvmev_result *ret);

#endif