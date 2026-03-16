// SPDX-License-Identifier: GPL-2.0-only
/*
 * conv_ftl.c - Conventional FTL with CAT + Fine-Grained Hot/Cold Redistribution
 *
 * 논문: "Cleaning policies in mobile computers using flash memory"
 *       (M.-L. Chiang, R.-C. Chang, 1999)
 *
 * 구현 요소:
 *   1) CAT (Cost-Age-Times) victim 선택: Age×IPC / (VPC × EraseCount)
 *   2) M6 Fine-Grained Data Redistribution:
 *      - per-LPN page_meta로 update_cnt, last_write_time 추적
 *      - GC 시 valid page별 hot_degree 계산 → 평균 대비 hot/cold 분류
 *      - hot page → gc_wp_hot, cold page → gc_wp_cold로 분리 기록
 *      - 유저 새 쓰기는 항상 hot 취급 (논문: "Data newly written are treated as hot")
 *
 * Write Pointer 구성 (총 3개의 open block):
 *   wp          - 유저 쓰기 전용
 *   gc_wp_hot   - GC hot page 복사 전용
 *   gc_wp_cold  - GC cold page 복사 전용
 */

#include <linux/vmalloc.h>
#include <linux/ktime.h>
#include <linux/sched/clock.h>

#include "nvmev.h"
#include "conv_ftl.h"

/* ============================================================
 * Age 가중치 함수 (논문 Fig.7 기반 계단 함수)
 * ============================================================ */
#define MS_TO_NS(x)   ((uint64_t)(x) * 1000000ULL)
#define SEC_TO_NS(x)  ((uint64_t)(x) * 1000000000ULL)

#define THRESHOLD_VERY_HOT  MS_TO_NS(100)
#define THRESHOLD_HOT       SEC_TO_NS(5)
#define THRESHOLD_WARM      SEC_TO_NS(60)

static uint64_t get_age_weight(uint64_t age_ns)
{
	if (age_ns < THRESHOLD_VERY_HOT)
		return 1;
	else if (age_ns < THRESHOLD_HOT)
		return 5;
	else if (age_ns < THRESHOLD_WARM)
		return 20;
	else
		return 100;
}

/* ============================================================
 * Hot Degree 계산 (논문 Section 3.2)
 *
 * "The hot degree of a block is defined as the number of times
 *  the block has been updated and decreases as the block's age grows."
 *
 * hot_degree = update_cnt × SCALE / age_weight
 *
 * - update_cnt 높을수록 → hot_degree ↑ (자주 수정됨)
 * - age_weight 높을수록 → hot_degree ↓ (오래됨 → cold화)
 * - SCALE(1000)은 정수 나눗셈 정밀도를 위한 스케일링
 * ============================================================ */
#define HOT_DEGREE_SCALE  1000

static uint64_t calc_hot_degree(struct page_meta *meta, uint64_t now)
{
	uint64_t age, age_w;

	if (meta->update_cnt == 0)
		return 0;

	age = (now > meta->last_write_time) ?
	      (now - meta->last_write_time) : 0;
	age_w = get_age_weight(age);

	return ((uint64_t)meta->update_cnt * HOT_DEGREE_SCALE) / age_w;
}

/* ============================================================
 * 기본 유틸리티
 * ============================================================ */
static inline bool last_pg_in_wordline(struct conv_ftl *conv_ftl,
				       struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	return (ppa->g.pg % spp->pgs_per_oneshotpg) == (spp->pgs_per_oneshotpg - 1);
}

static bool should_gc(struct conv_ftl *conv_ftl)
{
	return (conv_ftl->lm.free_line_cnt <= conv_ftl->cp.gc_thres_lines);
}

static inline bool should_gc_high(struct conv_ftl *conv_ftl)
{
	return conv_ftl->lm.free_line_cnt <= conv_ftl->cp.gc_thres_lines_high;
}

static inline struct ppa get_maptbl_ent(struct conv_ftl *conv_ftl,
					uint64_t lpn)
{
	return conv_ftl->maptbl[lpn];
}

static inline void set_maptbl_ent(struct conv_ftl *conv_ftl,
				  uint64_t lpn, struct ppa *ppa)
{
	NVMEV_ASSERT(lpn < conv_ftl->ssd->sp.tt_pgs);
	conv_ftl->maptbl[lpn] = *ppa;
}

static uint64_t ppa2pgidx(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint64_t pgidx;

	pgidx = ppa->g.ch * spp->pgs_per_ch + ppa->g.lun * spp->pgs_per_lun +
		ppa->g.pl * spp->pgs_per_pl + ppa->g.blk * spp->pgs_per_blk +
		ppa->g.pg;

	NVMEV_ASSERT(pgidx < spp->tt_pgs);
	return pgidx;
}

static inline uint64_t get_rmap_ent(struct conv_ftl *conv_ftl,
				    struct ppa *ppa)
{
	return conv_ftl->rmap[ppa2pgidx(conv_ftl, ppa)];
}

static inline void set_rmap_ent(struct conv_ftl *conv_ftl,
				uint64_t lpn, struct ppa *ppa)
{
	conv_ftl->rmap[ppa2pgidx(conv_ftl, ppa)] = lpn;
}

/* ============================================================
 * PQ 콜백 (Greedy Min-Heap)
 * ============================================================ */
static inline int victim_line_cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
	return (next > curr);
}

static inline pqueue_pri_t victim_line_get_pri(void *a)
{
	return ((struct line *)a)->vpc;
}

static inline void victim_line_set_pri(void *a, pqueue_pri_t pri)
{
	((struct line *)a)->vpc = pri;
}

static inline size_t victim_line_get_pos(void *a)
{
	return ((struct line *)a)->pos;
}

static inline void victim_line_set_pos(void *a, size_t pos)
{
	((struct line *)a)->pos = pos;
}

/* ============================================================
 * 쓰기 크레딧
 * ============================================================ */
static inline void consume_write_credit(struct conv_ftl *conv_ftl)
{
	conv_ftl->wfc.write_credits--;
}

static void foreground_gc(struct conv_ftl *conv_ftl);

static inline void check_and_refill_write_credit(struct conv_ftl *conv_ftl)
{
	struct write_flow_control *wfc = &(conv_ftl->wfc);
	if (wfc->write_credits <= 0) {
		foreground_gc(conv_ftl);
		wfc->write_credits += wfc->credits_to_refill;
	}
}

/* ============================================================
 * Victim 선택 ① : Greedy
 * ============================================================ */
static struct line *select_victim_greedy(struct conv_ftl *conv_ftl,
					 bool force)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *victim_line = pqueue_peek(lm->victim_line_pq);

	if (!victim_line)
		return NULL;
	if (!force && (victim_line->vpc > (spp->pgs_per_line / 8)))
		return NULL;

	pqueue_pop(lm->victim_line_pq);
	victim_line->pos = 0;
	lm->victim_line_cnt--;
	return victim_line;
}

/* ============================================================
 * Victim 선택 ② : Cost-Benefit
 * ============================================================ */
static struct line *select_victim_cb(struct conv_ftl *conv_ftl, bool force)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	pqueue_t *q = lm->victim_line_pq;
	struct line *best = NULL;
	uint64_t max_score = 0;
	uint64_t now = ktime_get_ns();
	size_t i;

	if (q->size == 0)
		return NULL;

	for (i = 1; i <= q->size; i++) {
		struct line *c = (struct line *)q->d[i];
		uint64_t age, score;

		if (!c)
			continue;
		age = (now > c->last_modified_time) ?
		      (now - c->last_modified_time) : 0;
		score = (get_age_weight(age) * (uint64_t)c->ipc) /
			((uint64_t)(c->vpc + 1));
		if (score > max_score) {
			max_score = score;
			best = c;
		}
	}

	if (best) {
		pqueue_remove(q, best);
		best->pos = 0;
		lm->victim_line_cnt--;
	}
	return best;
}

/* ============================================================
 * Victim 선택 ③ : CAT (Cost-Age-Times)
 *
 * Score = AgeWeight × IPC / ((VPC+1) × (EraseCount+1))
 * ============================================================ */
static struct line *select_victim_cat(struct conv_ftl *conv_ftl, bool force)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	pqueue_t *q = lm->victim_line_pq;
	struct line *best = NULL;
	uint64_t max_score = 0;
	uint64_t now = ktime_get_ns();
	size_t i;

	if (q->size == 0)
		return NULL;

	for (i = 1; i <= q->size; i++) {
		struct line *c = (struct line *)q->d[i];
		uint64_t age, score;

		if (!c)
			continue;
		age = (now > c->last_modified_time) ?
		      (now - c->last_modified_time) : 0;

		score = (get_age_weight(age) * (uint64_t)c->ipc) /
			((uint64_t)(c->vpc + 1) * ((uint64_t)c->erase_cnt + 1));

		if (score > max_score) {
			max_score = score;
			best = c;
		}
	}

	if (best) {
		pqueue_remove(q, best);
		best->pos = 0;
		lm->victim_line_cnt--;
	}
	return best;
}

/* ============================================================
 * 라인 초기화
 * ============================================================ */
static void init_lines(struct conv_ftl *conv_ftl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	int i;

	lm->tt_lines = spp->tt_lines;
	lm->lines = vmalloc(sizeof(struct line) * lm->tt_lines);

	lm->victim_line_pq = pqueue_init(lm->tt_lines,
					 victim_line_cmp_pri,
					 victim_line_get_pri,
					 victim_line_set_pri,
					 victim_line_get_pos,
					 victim_line_set_pos);

	INIT_LIST_HEAD(&lm->free_line_list);
	INIT_LIST_HEAD(&lm->full_line_list);
	lm->free_line_cnt = 0;

	for (i = 0; i < (int)lm->tt_lines; i++) {
		lm->lines[i] = (struct line){
			.id = i,
			.ipc = 0,
			.vpc = 0,
			.pos = 0,
			.last_modified_time = 0,
			.erase_cnt = 0,
			.entry = LIST_HEAD_INIT(lm->lines[i].entry),
		};
		list_add_tail(&lm->lines[i].entry, &lm->free_line_list);
		lm->free_line_cnt++;
	}

	NVMEV_ASSERT(lm->free_line_cnt == spp->tt_lines);
	lm->victim_line_cnt = 0;
	lm->full_line_cnt = 0;
}

static void remove_lines(struct conv_ftl *conv_ftl)
{
	pqueue_free(conv_ftl->lm.victim_line_pq);
	vfree(conv_ftl->lm.lines);
}

/* ============================================================
 * per-LPN 메타데이터 초기화/해제
 * ============================================================ */
static void init_page_meta(struct conv_ftl *conv_ftl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int i;

	conv_ftl->page_meta = vmalloc(sizeof(struct page_meta) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++) {
		conv_ftl->page_meta[i].update_cnt = 0;
		conv_ftl->page_meta[i].last_write_time = 0;
	}
}

static void remove_page_meta(struct conv_ftl *conv_ftl)
{
	vfree(conv_ftl->page_meta);
}

/* ============================================================
 * 쓰기 유량 제어
 * ============================================================ */
static void init_write_flow_control(struct conv_ftl *conv_ftl)
{
	struct write_flow_control *wfc = &(conv_ftl->wfc);
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	wfc->write_credits = spp->pgs_per_line;
	wfc->credits_to_refill = spp->pgs_per_line;
}

/* ============================================================
 * 주소 및 라인 관련 유틸리티
 * ============================================================ */
static inline void check_addr(int a, int max)
{
	NVMEV_ASSERT(a >= 0 && a < max);
}

static struct line *get_next_free_line(struct conv_ftl *conv_ftl)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *curline = list_first_entry_or_null(&lm->free_line_list,
						       struct line, entry);
	if (!curline) {
		NVMEV_ERROR("No free line left in VIRT !!!!\n");
		return NULL;
	}
	list_del_init(&curline->entry);
	lm->free_line_cnt--;
	return curline;
}

/* ============================================================
 * 쓰기 포인터 관리
 *
 * io_type 라우팅:
 *   USER_IO    → wp          (유저 쓰기)
 *   GC_HOT_IO  → gc_wp_hot   (GC hot page)
 *   GC_COLD_IO → gc_wp_cold  (GC cold page)
 * ============================================================ */
static struct write_pointer *__get_wp(struct conv_ftl *ftl, uint32_t io_type)
{
	switch (io_type) {
	case USER_IO:
		return &ftl->wp;
	case GC_HOT_IO:
		return &ftl->gc_wp_hot;
	case GC_COLD_IO:
		return &ftl->gc_wp_cold;
	default:
		NVMEV_ASSERT(0);
		return NULL;
	}
}

static void prepare_write_pointer(struct conv_ftl *conv_ftl,
				  uint32_t io_type)
{
	struct write_pointer *wp = __get_wp(conv_ftl, io_type);
	struct line *curline = get_next_free_line(conv_ftl);

	NVMEV_ASSERT(wp);
	NVMEV_ASSERT(curline);

	*wp = (struct write_pointer){
		.curline = curline,
		.ch = 0,
		.lun = 0,
		.pg = 0,
		.blk = curline->id,
		.pl = 0,
	};
}

static void advance_write_pointer(struct conv_ftl *conv_ftl,
				  uint32_t io_type)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct write_pointer *wpp = __get_wp(conv_ftl, io_type);

	check_addr(wpp->pg, spp->pgs_per_blk);
	wpp->pg++;
	if ((wpp->pg % spp->pgs_per_oneshotpg) != 0)
		goto out;

	wpp->pg -= spp->pgs_per_oneshotpg;
	check_addr(wpp->ch, spp->nchs);
	wpp->ch++;
	if (wpp->ch != spp->nchs)
		goto out;

	wpp->ch = 0;
	check_addr(wpp->lun, spp->luns_per_ch);
	wpp->lun++;
	if (wpp->lun != spp->luns_per_ch)
		goto out;

	wpp->lun = 0;
	wpp->pg += spp->pgs_per_oneshotpg;
	if (wpp->pg != spp->pgs_per_blk)
		goto out;

	/* 라인 가득 참 → full 또는 victim 리스트 이동 */
	wpp->pg = 0;
	if (wpp->curline->vpc == spp->pgs_per_line) {
		NVMEV_ASSERT(wpp->curline->ipc == 0);
		list_add_tail(&wpp->curline->entry, &lm->full_line_list);
		lm->full_line_cnt++;
	} else {
		NVMEV_ASSERT(wpp->curline->ipc > 0);
		pqueue_insert(lm->victim_line_pq, wpp->curline);
		lm->victim_line_cnt++;
	}

	check_addr(wpp->blk, spp->blks_per_pl);
	wpp->curline = get_next_free_line(conv_ftl);
	wpp->blk = wpp->curline->id;
	check_addr(wpp->blk, spp->blks_per_pl);

	NVMEV_ASSERT(wpp->pg == 0);
	NVMEV_ASSERT(wpp->lun == 0);
	NVMEV_ASSERT(wpp->ch == 0);
	NVMEV_ASSERT(wpp->pl == 0);
out:
	return;
}

static struct ppa get_new_page(struct conv_ftl *conv_ftl, uint32_t io_type)
{
	struct ppa ppa;
	struct write_pointer *wp = __get_wp(conv_ftl, io_type);

	ppa.ppa = 0;
	ppa.g.ch = wp->ch;
	ppa.g.lun = wp->lun;
	ppa.g.pg = wp->pg;
	ppa.g.blk = wp->blk;
	ppa.g.pl = wp->pl;

	NVMEV_ASSERT(ppa.g.pl == 0);
	return ppa;
}

/* ============================================================
 * 매핑/역매핑 테이블
 * ============================================================ */
static void init_maptbl(struct conv_ftl *conv_ftl)
{
	int i;
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	conv_ftl->maptbl = vmalloc(sizeof(struct ppa) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++)
		conv_ftl->maptbl[i].ppa = UNMAPPED_PPA;
}

static void remove_maptbl(struct conv_ftl *conv_ftl)
{
	vfree(conv_ftl->maptbl);
}

static void init_rmap(struct conv_ftl *conv_ftl)
{
	int i;
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	conv_ftl->rmap = vmalloc(sizeof(uint64_t) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++)
		conv_ftl->rmap[i] = INVALID_LPN;
}

static void remove_rmap(struct conv_ftl *conv_ftl)
{
	vfree(conv_ftl->rmap);
}

/* ============================================================
 * FTL 인스턴스 초기화/제거
 * ============================================================ */
static void conv_init_ftl(struct conv_ftl *conv_ftl,
			  struct convparams *cpp, struct ssd *ssd)
{
	conv_ftl->cp = *cpp;
	conv_ftl->ssd = ssd;
	conv_ftl->gc_count = 0;
	conv_ftl->gc_copied_pages = 0;
	conv_ftl->gc_hot_copied = 0;
	conv_ftl->gc_cold_copied = 0;
	conv_ftl->cur_avg_hot_degree = 0;

	init_maptbl(conv_ftl);
	init_rmap(conv_ftl);
	init_page_meta(conv_ftl);
	init_lines(conv_ftl);

	/*
	 * 쓰기 포인터 3개 준비: open block 3개 소비
	 *   USER_IO    → 유저 쓰기
	 *   GC_HOT_IO  → GC hot 복사
	 *   GC_COLD_IO → GC cold 복사
	 */
	prepare_write_pointer(conv_ftl, USER_IO);
	prepare_write_pointer(conv_ftl, GC_HOT_IO);
	prepare_write_pointer(conv_ftl, GC_COLD_IO);

	init_write_flow_control(conv_ftl);

	NVMEV_INFO("Init FTL: %d channels, %ld pages "
		   "(3 open blocks: user/gc_hot/gc_cold)\n",
		   ssd->sp.nchs, ssd->sp.tt_pgs);
}

static void conv_remove_ftl(struct conv_ftl *conv_ftl)
{
	remove_lines(conv_ftl);
	remove_rmap(conv_ftl);
	remove_maptbl(conv_ftl);
	remove_page_meta(conv_ftl);
}

static void conv_init_params(struct convparams *cpp)
{
	cpp->op_area_pcent = OP_AREA_PERCENT;
	/*
	 * open block 3개(user, gc_hot, gc_cold) 사용하므로
	 * 최소 free line이 3개 남아야 안전.
	 */
	cpp->gc_thres_lines = 3;
	cpp->gc_thres_lines_high = 3;
	cpp->enable_gc_delay = 1;
	cpp->pba_pcent = (int)((1 + cpp->op_area_pcent) * 100);
}

/* ============================================================
 * 네임스페이스 초기화/제거
 * ============================================================ */
void conv_init_namespace(struct nvmev_ns *ns, uint32_t id,
			 uint64_t size, void *mapped_addr,
			 uint32_t cpu_nr_dispatcher)
{
	struct ssdparams spp;
	struct convparams cpp;
	struct conv_ftl *conv_ftls;
	struct ssd *ssd;
	uint32_t i;
	const uint32_t nr_parts = SSD_PARTITIONS;

	ssd_init_params(&spp, size, nr_parts);
	conv_init_params(&cpp);

	conv_ftls = kmalloc(sizeof(struct conv_ftl) * nr_parts, GFP_KERNEL);

	for (i = 0; i < nr_parts; i++) {
		ssd = kmalloc(sizeof(struct ssd), GFP_KERNEL);
		ssd_init(ssd, &spp, cpu_nr_dispatcher);
		conv_init_ftl(&conv_ftls[i], &cpp, ssd);
	}

	for (i = 1; i < nr_parts; i++) {
		kfree(conv_ftls[i].ssd->pcie->perf_model);
		kfree(conv_ftls[i].ssd->pcie);
		kfree(conv_ftls[i].ssd->write_buffer);
		conv_ftls[i].ssd->pcie = conv_ftls[0].ssd->pcie;
		conv_ftls[i].ssd->write_buffer = conv_ftls[0].ssd->write_buffer;
	}

	ns->id = id;
	ns->csi = NVME_CSI_NVM;
	ns->nr_parts = nr_parts;
	ns->ftls = (void *)conv_ftls;
	ns->size = (uint64_t)((size * 100) / cpp.pba_pcent);
	ns->mapped = mapped_addr;
	ns->proc_io_cmd = conv_proc_nvme_io_cmd;

	NVMEV_INFO("FTL physical: %lld, logical: %lld (ratio=%d)\n",
		   size, ns->size, cpp.pba_pcent);
}

void conv_remove_namespace(struct nvmev_ns *ns)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	const uint32_t nr_parts = SSD_PARTITIONS;
	uint32_t i;

	for (i = 1; i < nr_parts; i++) {
		conv_ftls[i].ssd->pcie = NULL;
		conv_ftls[i].ssd->write_buffer = NULL;
	}
	for (i = 0; i < nr_parts; i++) {
		conv_remove_ftl(&conv_ftls[i]);
		ssd_remove(conv_ftls[i].ssd);
		kfree(conv_ftls[i].ssd);
	}
	kfree(conv_ftls);
	ns->ftls = NULL;
}

/* ============================================================
 * 유효성 검사
 * ============================================================ */
static inline bool valid_ppa(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	if (ppa->g.ch < 0 || ppa->g.ch >= spp->nchs)       return false;
	if (ppa->g.lun < 0 || ppa->g.lun >= spp->luns_per_ch) return false;
	if (ppa->g.pl < 0 || ppa->g.pl >= spp->pls_per_lun)   return false;
	if (ppa->g.blk < 0 || ppa->g.blk >= spp->blks_per_pl) return false;
	if (ppa->g.pg < 0 || ppa->g.pg >= spp->pgs_per_blk)   return false;
	return true;
}

static inline bool valid_lpn(struct conv_ftl *conv_ftl, uint64_t lpn)
{
	return (lpn < conv_ftl->ssd->sp.tt_pgs);
}

static inline bool mapped_ppa(struct ppa *ppa)
{
	return !(ppa->ppa == UNMAPPED_PPA);
}

static inline struct line *get_line(struct conv_ftl *conv_ftl,
				    struct ppa *ppa)
{
	return &(conv_ftl->lm.lines[ppa->g.blk]);
}

/* ============================================================
 * 페이지 상태 관리
 * ============================================================ */
static void mark_page_invalid(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct nand_block *blk = NULL;
	struct nand_page *pg = NULL;
	bool was_full_line = false;
	struct line *line;

	pg = get_pg(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(pg->status == PG_VALID);
	pg->status = PG_INVALID;

	blk = get_blk(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(blk->ipc >= 0 && blk->ipc < spp->pgs_per_blk);
	blk->ipc++;
	NVMEV_ASSERT(blk->vpc > 0 && blk->vpc <= spp->pgs_per_blk);
	blk->vpc--;

	line = get_line(conv_ftl, ppa);
	NVMEV_ASSERT(line->ipc >= 0 && line->ipc < spp->pgs_per_line);
	if (line->vpc == spp->pgs_per_line) {
		NVMEV_ASSERT(line->ipc == 0);
		was_full_line = true;
	}
	line->ipc++;
	NVMEV_ASSERT(line->vpc > 0 && line->vpc <= spp->pgs_per_line);

	if (line->pos)
		pqueue_change_priority(lm->victim_line_pq, line->vpc - 1, line);
	else
		line->vpc--;

	if (was_full_line) {
		list_del_init(&line->entry);
		lm->full_line_cnt--;
		pqueue_insert(lm->victim_line_pq, line);
		lm->victim_line_cnt++;
	}

	line->last_modified_time = ktime_get_ns();
}

static void mark_page_valid(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct nand_block *blk = NULL;
	struct nand_page *pg = NULL;
	struct line *line;

	pg = get_pg(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(pg->status == PG_FREE);
	pg->status = PG_VALID;

	blk = get_blk(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(blk->vpc >= 0 && blk->vpc < spp->pgs_per_blk);
	blk->vpc++;

	line = get_line(conv_ftl, ppa);
	NVMEV_ASSERT(line->vpc >= 0 && line->vpc < spp->pgs_per_line);
	line->vpc++;
}

static void mark_block_free(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct nand_block *blk = get_blk(conv_ftl->ssd, ppa);
	struct nand_page *pg = NULL;
	int i;

	for (i = 0; i < spp->pgs_per_blk; i++) {
		pg = &blk->pg[i];
		NVMEV_ASSERT(pg->nsecs == spp->secs_per_pg);
		pg->status = PG_FREE;
	}

	NVMEV_ASSERT(blk->npgs == spp->pgs_per_blk);
	blk->ipc = 0;
	blk->vpc = 0;
	blk->erase_cnt++;
}

static void mark_line_free(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *line = get_line(conv_ftl, ppa);

	line->ipc = 0;
	line->vpc = 0;
	line->erase_cnt++;

	list_add_tail(&line->entry, &lm->free_line_list);
	lm->free_line_cnt++;
}

/* ============================================================
 * GC 읽기
 * ============================================================ */
static void gc_read_page(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;

	if (cpp->enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO,
			.cmd = NAND_READ,
			.stime = 0,
			.xfer_size = spp->pgsz,
			.interleave_pci_dma = false,
			.ppa = ppa,
		};
		ssd_advance_nand(conv_ftl->ssd, &gcr);
	}
}

/* ============================================================
 * GC 쓰기 - Hot/Cold 분기 핵심
 *
 * 1) rmap에서 LPN 확인
 * 2) page_meta에서 hot_degree 계산
 * 3) cur_avg_hot_degree와 비교 → hot이면 gc_wp_hot, cold면 gc_wp_cold
 * 4) 해당 wp에서 새 페이지 할당 → 기록
 *
 * NAND 명령에는 항상 GC_IO 사용 (타이밍 모델용).
 * wp 라우팅에만 GC_HOT_IO / GC_COLD_IO 사용.
 * ============================================================ */
static uint64_t gc_write_page(struct conv_ftl *conv_ftl,
			      struct ppa *old_ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	struct ppa new_ppa;
	uint64_t lpn = get_rmap_ent(conv_ftl, old_ppa);
	uint64_t now = ktime_get_ns();
	uint64_t hot_degree;
	uint32_t wp_type;

	NVMEV_ASSERT(valid_lpn(conv_ftl, lpn));

	/* ★ Hot/Cold 판별 */
	hot_degree = calc_hot_degree(&conv_ftl->page_meta[lpn], now);

	if (hot_degree >= conv_ftl->cur_avg_hot_degree) {
		wp_type = GC_HOT_IO;
		conv_ftl->gc_hot_copied++;
	} else {
		wp_type = GC_COLD_IO;
		conv_ftl->gc_cold_copied++;
	}

	/* 해당 wp에서 새 페이지 할당 */
	new_ppa = get_new_page(conv_ftl, wp_type);

	set_maptbl_ent(conv_ftl, lpn, &new_ppa);
	set_rmap_ent(conv_ftl, lpn, &new_ppa);
	mark_page_valid(conv_ftl, &new_ppa);
	conv_ftl->gc_copied_pages++;

	advance_write_pointer(conv_ftl, wp_type);

	/* NAND 쓰기 시뮬레이션 (타이밍은 GC_IO 사용) */
	if (cpp->enable_gc_delay) {
		struct nand_cmd gcw = {
			.type = GC_IO,
			.cmd = NAND_NOP,
			.stime = 0,
			.interleave_pci_dma = false,
			.ppa = &new_ppa,
		};
		if (last_pg_in_wordline(conv_ftl, &new_ppa)) {
			gcw.cmd = NAND_WRITE;
			gcw.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg;
		}
		ssd_advance_nand(conv_ftl->ssd, &gcw);
	}

	return 0;
}

/* ============================================================
 * 블록/플래시페이지 청소
 * ============================================================ */
static void clean_one_flashpg(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	struct nand_page *pg_iter = NULL;
	int cnt = 0, i;
	struct ppa ppa_copy = *ppa;

	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg_iter = get_pg(conv_ftl->ssd, &ppa_copy);
		NVMEV_ASSERT(pg_iter->status != PG_FREE);
		if (pg_iter->status == PG_VALID)
			cnt++;
		ppa_copy.g.pg++;
	}

	ppa_copy = *ppa;
	if (cnt <= 0)
		return;

	if (cpp->enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO,
			.cmd = NAND_READ,
			.stime = 0,
			.xfer_size = spp->pgsz * cnt,
			.interleave_pci_dma = false,
			.ppa = &ppa_copy,
		};
		ssd_advance_nand(conv_ftl->ssd, &gcr);
	}

	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg_iter = get_pg(conv_ftl->ssd, &ppa_copy);
		if (pg_iter->status == PG_VALID)
			gc_write_page(conv_ftl, &ppa_copy);
		ppa_copy.g.pg++;
	}
}

/* ============================================================
 * GC Pre-Scan: victim 블록의 평균 hot_degree 산출
 *
 * do_gc 본체에서 실제 청소 전에 호출.
 * victim 블록 내 모든 valid page의 hot_degree를 구해 평균을 낸다.
 *
 * 논문: "A block is defined as hot if its hot degree exceeds
 *        the average hot degree."
 * ============================================================ */
static void gc_prescan_hot_degree(struct conv_ftl *conv_ftl,
				  struct line *victim_line)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct ppa ppa;
	uint64_t now = ktime_get_ns();
	uint64_t sum = 0;
	uint32_t count = 0;
	int ch, lun, pg;

	ppa.g.blk = victim_line->id;
	ppa.g.pl = 0;

	for (pg = 0; pg < spp->pgs_per_blk; pg++) {
		ppa.g.pg = pg;
		for (ch = 0; ch < spp->nchs; ch++) {
			ppa.g.ch = ch;
			for (lun = 0; lun < spp->luns_per_ch; lun++) {
				struct nand_page *page;
				uint64_t lpn;

				ppa.g.lun = lun;
				page = get_pg(conv_ftl->ssd, &ppa);

				if (page->status != PG_VALID)
					continue;

				lpn = get_rmap_ent(conv_ftl, &ppa);
				if (lpn == INVALID_LPN)
					continue;

				sum += calc_hot_degree(
					&conv_ftl->page_meta[lpn], now);
				count++;
			}
		}
	}

	conv_ftl->cur_avg_hot_degree = (count > 0) ? (sum / count) : 0;

	NVMEV_DEBUG("GC prescan line %d: %u valid pages, "
		    "avg_hot_degree=%llu\n",
		    victim_line->id, count,
		    conv_ftl->cur_avg_hot_degree);
}

/* ============================================================
 * do_gc - victim_select_fn을 파라미터로 받음
 *
 * 흐름:
 *   1) select_fn으로 victim 선택
 *   2) gc_prescan_hot_degree로 평균 hot_degree 산출
 *   3) clean_one_flashpg → gc_write_page에서 hot/cold 분기
 * ============================================================ */
static int do_gc(struct conv_ftl *conv_ftl, bool force,
		 victim_select_fn select_fn)
{
	struct line *victim_line = NULL;
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct ppa ppa;
	int flashpg;

	victim_line = select_fn(conv_ftl, force);
	if (!victim_line)
		return -1;

	conv_ftl->gc_count++;
	ppa.g.blk = victim_line->id;

	NVMEV_DEBUG_VERBOSE("GC line:%d ipc=%d vpc=%d erase_cnt=%u\n",
			    ppa.g.blk, victim_line->ipc, victim_line->vpc,
			    victim_line->erase_cnt);

	conv_ftl->wfc.credits_to_refill = victim_line->ipc;

	/* ★ Phase 1: Pre-scan으로 평균 hot_degree 계산 */
	gc_prescan_hot_degree(conv_ftl, victim_line);

	/* ★ Phase 2: 실제 청소 (gc_write_page에서 hot/cold 분기) */
	for (flashpg = 0; flashpg < spp->flashpgs_per_blk; flashpg++) {
		int ch, lun;

		ppa.g.pg = flashpg * spp->pgs_per_flashpg;
		for (ch = 0; ch < spp->nchs; ch++) {
			for (lun = 0; lun < spp->luns_per_ch; lun++) {
				struct nand_lun *lunp;

				ppa.g.ch = ch;
				ppa.g.lun = lun;
				ppa.g.pl = 0;
				lunp = get_lun(conv_ftl->ssd, &ppa);

				clean_one_flashpg(conv_ftl, &ppa);

				if (flashpg == (spp->flashpgs_per_blk - 1)) {
					struct convparams *cpp = &conv_ftl->cp;

					mark_block_free(conv_ftl, &ppa);

					if (cpp->enable_gc_delay) {
						struct nand_cmd gce = {
							.type = GC_IO,
							.cmd = NAND_ERASE,
							.stime = 0,
							.interleave_pci_dma = false,
							.ppa = &ppa,
						};
						ssd_advance_nand(conv_ftl->ssd,
								 &gce);
					}
					lunp->gc_endtime =
						lunp->next_lun_avail_time;
				}
			}
		}
	}

	mark_line_free(conv_ftl, &ppa);
	return 0;
}

/* ============================================================
 * foreground_gc
 *
 * 전략 변경은 이 한 줄만 수정:
 *   do_gc(conv_ftl, true, select_victim_cat);
 *   do_gc(conv_ftl, true, select_victim_cb);
 *   do_gc(conv_ftl, true, select_victim_greedy);
 * ============================================================ */
static void foreground_gc(struct conv_ftl *conv_ftl)
{
	if (should_gc_high(conv_ftl)) {
		do_gc(conv_ftl, true, select_victim_cat);
	}
}

/* ============================================================
 * NVMe Read
 * ============================================================ */
static bool is_same_flash_page(struct conv_ftl *conv_ftl,
			       struct ppa ppa1, struct ppa ppa2)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint32_t p1 = ppa1.g.pg / spp->pgs_per_flashpg;
	uint32_t p2 = ppa2.g.pg / spp->pgs_per_flashpg;

	return (ppa1.h.blk_in_ssd == ppa2.h.blk_in_ssd) && (p1 == p2);
}

static bool conv_read(struct nvmev_ns *ns, struct nvmev_request *req,
		      struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *conv_ftl = &conv_ftls[0];
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba;
	uint64_t nr_lba = (cmd->rw.length + 1);
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;
	uint64_t lpn;
	uint64_t nsecs_start = req->nsecs_start;
	uint64_t nsecs_completed, nsecs_latest = nsecs_start;
	uint32_t xfer_size, i;
	uint32_t nr_parts = ns->nr_parts;

	struct ppa prev_ppa;
	struct nand_cmd srd = {
		.type = USER_IO,
		.cmd = NAND_READ,
		.stime = nsecs_start,
		.interleave_pci_dma = true,
	};

	NVMEV_ASSERT(conv_ftls);
	if ((end_lpn / nr_parts) >= spp->tt_pgs) {
		NVMEV_ERROR("%s: lpn out of range\n", __func__);
		return false;
	}

	if (LBA_TO_BYTE(nr_lba) <= (KB(4) * nr_parts))
		srd.stime += spp->fw_4kb_rd_lat;
	else
		srd.stime += spp->fw_rd_lat;

	for (i = 0; (i < nr_parts) && (start_lpn <= end_lpn);
	     i++, start_lpn++) {
		conv_ftl = &conv_ftls[start_lpn % nr_parts];
		xfer_size = 0;
		prev_ppa = get_maptbl_ent(conv_ftl, start_lpn / nr_parts);

		for (lpn = start_lpn; lpn <= end_lpn; lpn += nr_parts) {
			uint64_t local_lpn = lpn / nr_parts;
			struct ppa cur_ppa = get_maptbl_ent(conv_ftl,
							    local_lpn);

			if (!mapped_ppa(&cur_ppa) ||
			    !valid_ppa(conv_ftl, &cur_ppa))
				continue;

			if (mapped_ppa(&prev_ppa) &&
			    is_same_flash_page(conv_ftl, cur_ppa, prev_ppa)) {
				xfer_size += spp->pgsz;
				continue;
			}

			if (xfer_size > 0) {
				srd.xfer_size = xfer_size;
				srd.ppa = &prev_ppa;
				nsecs_completed = ssd_advance_nand(
					conv_ftl->ssd, &srd);
				nsecs_latest = max(nsecs_completed,
						   nsecs_latest);
			}

			xfer_size = spp->pgsz;
			prev_ppa = cur_ppa;
		}

		if (xfer_size > 0) {
			srd.xfer_size = xfer_size;
			srd.ppa = &prev_ppa;
			nsecs_completed = ssd_advance_nand(conv_ftl->ssd,
							   &srd);
			nsecs_latest = max(nsecs_completed, nsecs_latest);
		}
	}

	ret->nsecs_target = nsecs_latest;
	ret->status = NVME_SC_SUCCESS;
	return true;
}

/* ============================================================
 * NVMe Write
 *
 * ★ page_meta 업데이트 포인트:
 *   - update_cnt 증가
 *   - last_write_time 갱신
 *   → 이후 GC에서 hot_degree 계산의 기초 데이터가 됨
 *
 * 논문: "Data newly written are treated as hot"
 *   → 유저 쓰기는 모두 wp (hot 계열)에 기록
 * ============================================================ */
static bool conv_write(struct nvmev_ns *ns, struct nvmev_request *req,
		       struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *conv_ftl = &conv_ftls[0];
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct buffer *wbuf = conv_ftl->ssd->write_buffer;

	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba;
	uint64_t nr_lba = (cmd->rw.length + 1);
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;
	uint64_t lpn;
	uint32_t nr_parts = ns->nr_parts;
	uint64_t nsecs_latest;
	uint64_t nsecs_xfer_completed;
	uint32_t allocated_buf_size;

	struct nand_cmd swr = {
		.type = USER_IO,
		.cmd = NAND_WRITE,
		.interleave_pci_dma = false,
		.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg,
	};

	if ((end_lpn / nr_parts) >= spp->tt_pgs) {
		NVMEV_ERROR("%s: lpn out of range\n", __func__);
		return false;
	}

	allocated_buf_size = buffer_allocate(wbuf, LBA_TO_BYTE(nr_lba));
	if (allocated_buf_size < LBA_TO_BYTE(nr_lba))
		return false;

	nsecs_latest = ssd_advance_write_buffer(conv_ftl->ssd,
						req->nsecs_start,
						LBA_TO_BYTE(nr_lba));
	nsecs_xfer_completed = nsecs_latest;
	swr.stime = nsecs_latest;

	for (lpn = start_lpn; lpn <= end_lpn; lpn++) {
		uint64_t local_lpn;
		uint64_t nsecs_completed = 0;
		struct ppa ppa;

		conv_ftl = &conv_ftls[lpn % nr_parts];
		local_lpn = lpn / nr_parts;

		ppa = get_maptbl_ent(conv_ftl, local_lpn);
		if (mapped_ppa(&ppa)) {
			mark_page_invalid(conv_ftl, &ppa);
			set_rmap_ent(conv_ftl, INVALID_LPN, &ppa);
		}

		/* 새 페이지 할당 (유저 쓰기 = hot 취급) */
		ppa = get_new_page(conv_ftl, USER_IO);
		set_maptbl_ent(conv_ftl, local_lpn, &ppa);
		set_rmap_ent(conv_ftl, local_lpn, &ppa);
		mark_page_valid(conv_ftl, &ppa);

		/* ★ page_meta 업데이트 */
		conv_ftl->page_meta[local_lpn].update_cnt++;
		conv_ftl->page_meta[local_lpn].last_write_time =
			ktime_get_ns();

		advance_write_pointer(conv_ftl, USER_IO);

		if (last_pg_in_wordline(conv_ftl, &ppa)) {
			swr.ppa = &ppa;
			nsecs_completed = ssd_advance_nand(conv_ftl->ssd,
							   &swr);
			nsecs_latest = max(nsecs_completed, nsecs_latest);
			schedule_internal_operation(
				req->sq_id, nsecs_completed, wbuf,
				spp->pgs_per_oneshotpg * spp->pgsz);
		}

		consume_write_credit(conv_ftl);
		check_and_refill_write_credit(conv_ftl);
	}

	if ((cmd->rw.control & NVME_RW_FUA) ||
	    (spp->write_early_completion == 0))
		ret->nsecs_target = nsecs_latest;
	else
		ret->nsecs_target = nsecs_xfer_completed;

	ret->status = NVME_SC_SUCCESS;
	return true;
}

/* ============================================================
 * NVMe Flush (GC 통계 출력 포함)
 * ============================================================ */
static void conv_flush(struct nvmev_ns *ns, struct nvmev_request *req,
		       struct nvmev_result *ret)
{
	uint64_t start, latest;
	uint32_t i;
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	uint64_t total_gc = 0, total_copied = 0;
	uint64_t total_hot = 0, total_cold = 0;

	start = local_clock();
	latest = start;
	for (i = 0; i < ns->nr_parts; i++)
		latest = max(latest, ssd_next_idle_time(conv_ftls[i].ssd));

	for (i = 0; i < ns->nr_parts; i++) {
		total_gc += conv_ftls[i].gc_count;
		total_copied += conv_ftls[i].gc_copied_pages;
		total_hot += conv_ftls[i].gc_hot_copied;
		total_cold += conv_ftls[i].gc_cold_copied;
	}

	printk(KERN_INFO "NVMeVirt: [FLUSH - GC Stats]\n");
	printk(KERN_INFO "NVMeVirt:  Total GC: %llu\n", total_gc);
	printk(KERN_INFO "NVMeVirt:  Total Copied: %llu "
			 "(hot=%llu, cold=%llu)\n",
	       total_copied, total_hot, total_cold);
	printk(KERN_INFO "NVMeVirt:  Avg Pages/GC: %llu\n",
	       total_gc > 0 ? total_copied / total_gc : 0);

	if (total_copied > 0) {
		printk(KERN_INFO "NVMeVirt:  Hot ratio: %llu%%\n",
		       (total_hot * 100) / total_copied);
	}

	/* Wear-leveling 통계 */
	if (ns->nr_parts > 0) {
		struct line_mgmt *lm = &conv_ftls[0].lm;
		uint64_t sum = 0;
		uint32_t max_ec = 0, min_ec = UINT_MAX;
		uint32_t j;

		for (j = 0; j < lm->tt_lines; j++) {
			uint32_t ec = lm->lines[j].erase_cnt;
			sum += ec;
			if (ec > max_ec) max_ec = ec;
			if (ec < min_ec) min_ec = ec;
		}
		printk(KERN_INFO "NVMeVirt: [Wear-Leveling] "
		       "avg_ec=%llu min=%u max=%u\n",
		       lm->tt_lines > 0 ? sum / lm->tt_lines : 0,
		       min_ec == UINT_MAX ? 0 : min_ec, max_ec);
	}

	ret->status = NVME_SC_SUCCESS;
	ret->nsecs_target = latest;
}

/* ============================================================
 * IO 명령 디스패처
 * ============================================================ */
bool conv_proc_nvme_io_cmd(struct nvmev_ns *ns, struct nvmev_request *req,
			   struct nvmev_result *ret)
{
	struct nvme_command *cmd = req->cmd;

	NVMEV_ASSERT(ns->csi == NVME_CSI_NVM);

	switch (cmd->common.opcode) {
	case nvme_cmd_write:
		if (!conv_write(ns, req, ret))
			return false;
		break;
	case nvme_cmd_read:
		if (!conv_read(ns, req, ret))
			return false;
		break;
	case nvme_cmd_flush:
		conv_flush(ns, req, ret);
		break;
	default:
		NVMEV_ERROR("%s: unimplemented: %s (0x%x)\n", __func__,
			    nvme_opcode_string(cmd->common.opcode),
			    cmd->common.opcode);
		break;
	}

	return true;
}