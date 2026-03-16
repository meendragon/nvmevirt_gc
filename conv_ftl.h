// SPDX-License-Identifier: GPL-2.0-only

#ifndef _NVMEVIRT_CONV_FTL_H
#define _NVMEVIRT_CONV_FTL_H

#include <linux/types.h>
#include "pqueue/pqueue.h"
#include "ssd_config.h"
#include "ssd.h"

struct conv_ftl;

/* GC victim 선택 함수 포인터 타입 */
typedef struct line *(*victim_select_fn)(struct conv_ftl *, bool);

/* IO 타입 (write pointer 라우팅용) */
#define USER_IO    0
#define GC_HOT_IO  1
#define GC_COLD_IO 2

struct convparams {
	uint32_t gc_thres_lines;
	uint32_t gc_thres_lines_high;
	bool enable_gc_delay;
	double op_area_pcent;
	int pba_pcent;
};

struct line {
	int id;
	int ipc;  /* invalid page count */
	int vpc;  /* valid page count */
	struct list_head entry;
	size_t pos;  /* position in pqueue */
	uint64_t last_modified_time;  /* 마지막 invalidation 시각 (ns) */
	uint32_t erase_cnt;           /* 이 라인의 erase 누적 횟수 (CAT wear-leveling용) */
};

/* per-LPN 메타데이터 (Fine-Grained Hot/Cold 분류용) */
struct page_meta {
	uint32_t update_cnt;      /* 해당 LPN이 갱신된 횟수 */
	uint64_t last_write_time; /* 마지막 기록 시각 (ns) */
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

	/* per-LPN 메타데이터 (hot_degree 계산용) */
	struct page_meta *page_meta;

	/* 쓰기 포인터 3개: user / gc_hot / gc_cold */
	struct write_pointer wp;
	struct write_pointer gc_wp_hot;
	struct write_pointer gc_wp_cold;

	struct line_mgmt lm;
	struct write_flow_control wfc;

	uint64_t gc_count;
	uint64_t gc_copied_pages;
	uint64_t gc_hot_copied;   /* GC에서 hot으로 분류되어 복사된 페이지 수 */
	uint64_t gc_cold_copied;  /* GC에서 cold로 분류되어 복사된 페이지 수 */
	uint64_t cur_avg_hot_degree; /* 현재 GC victim의 평균 hot_degree */
};

void conv_init_namespace(struct nvmev_ns *ns, uint32_t id, uint64_t size,
			 void *mapped_addr, uint32_t cpu_nr_dispatcher);
void conv_remove_namespace(struct nvmev_ns *ns);
bool conv_proc_nvme_io_cmd(struct nvmev_ns *ns, struct nvmev_request *req,
			   struct nvmev_result *ret);

#endif