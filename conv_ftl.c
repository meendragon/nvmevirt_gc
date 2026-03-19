// SPDX-License-Identifier: GPL-2.0-only
/*
 * conv_ftl.c - Conventional FTL with RL-based GC Policy Tuning
 *
 * GC Modes (insmod gc_mode=N):
 *   0: Greedy          — VPC 최소, 단순 빠름
 *   1: Cost-Benefit    — Age×IPC/VPC, locality 반영
 *   2: CAT             — Age×IPC/(VPC×EraseCount), wear-leveling 포함
 *   3: RL              — 통합 scoring 공식의 exponent를 Q-table로 자동 튜닝
 *
 * 통합 Scoring 공식 (RL mode):
 *   Score = AgeWeight^α × IPC / ((VPC+1) × (EraseCount+1)^δ)
 *
 *   α=0 → Greedy와 유사,  δ=0 → CB와 동일,  δ>0 → CAT 계열
 *   Q-table이 (α, δ, hot_threshold) 조합을 state에 따라 선택.
 *
 * 모든 모드에서 M6 Fine-Grained Hot/Cold Redistribution 활성:
 *   gc_wp_hot / gc_wp_cold 2개의 GC write pointer로 분리 기록.
 */

#include <linux/vmalloc.h>
#include <linux/ktime.h>
#include <linux/sched/clock.h>
#include <linux/moduleparam.h>
#include <linux/random.h>

#include "nvmev.h"
#include "conv_ftl.h"

/* ============================================================
 * 모듈 파라미터
 * ============================================================ */
static int gc_mode = GC_MODE_CAT;
module_param(gc_mode, int, 0644);
MODULE_PARM_DESC(gc_mode, "0=Greedy 1=CB 2=CAT 3=RL");

/* ============================================================
 * 정수 제곱근 (커널에서 float 사용 불가)
 * ============================================================ */
static uint32_t isqrt_u64(uint64_t n)
{
	uint64_t x, y;

	if (n <= 1)
		return (uint32_t)n;
	x = n;
	y = (x + 1) / 2;
	while (y < x) {
		x = y;
		y = (x + n / x) / 2;
	}
	return (uint32_t)x;
}

/* ============================================================
 * Age 가중치 (논문 Fig.7 계단 함수)
 * ============================================================ */
#define MS_TO_NS(x)  ((uint64_t)(x) * 1000000ULL)
#define SEC_TO_NS(x) ((uint64_t)(x) * 1000000000ULL)

#define TH_VERY_HOT  MS_TO_NS(100)
#define TH_HOT       SEC_TO_NS(5)
#define TH_WARM      SEC_TO_NS(60)

static uint64_t get_age_weight(uint64_t age_ns)
{
	if (age_ns < TH_VERY_HOT)  return 1;
	if (age_ns < TH_HOT)       return 5;
	if (age_ns < TH_WARM)      return 20;
	return 100;
}

/* ============================================================
 * Hot Degree (논문 Section 3.2)
 * hot_degree = update_cnt × 1000 / age_weight
 * ============================================================ */
#define HD_SCALE 1000

static uint64_t calc_hot_degree(struct page_meta *pm, uint64_t now)
{
	uint64_t age, aw;

	if (pm->update_cnt == 0)
		return 0;
	age = (now > pm->last_write_time) ? (now - pm->last_write_time) : 0;
	aw = get_age_weight(age);
	return ((uint64_t)pm->update_cnt * HD_SCALE) / aw;
}

static bool is_hot_page(struct conv_ftl *conv_ftl, uint64_t lpn, uint64_t now)
{
	uint64_t deg = calc_hot_degree(&conv_ftl->page_meta[lpn], now);
	/* avg_hot_degree는 ×16 고정소수점이므로 비교 시 deg도 ×16 */
	return (deg * 16) >= conv_ftl->avg_hot_degree;
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

static inline struct ppa get_maptbl_ent(struct conv_ftl *ftl, uint64_t lpn)
{
	return ftl->maptbl[lpn];
}

static inline void set_maptbl_ent(struct conv_ftl *ftl, uint64_t lpn,
				  struct ppa *ppa)
{
	NVMEV_ASSERT(lpn < ftl->ssd->sp.tt_pgs);
	ftl->maptbl[lpn] = *ppa;
}

static uint64_t ppa2pgidx(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	uint64_t idx = ppa->g.ch * spp->pgs_per_ch +
		       ppa->g.lun * spp->pgs_per_lun +
		       ppa->g.pl * spp->pgs_per_pl +
		       ppa->g.blk * spp->pgs_per_blk + ppa->g.pg;
	NVMEV_ASSERT(idx < spp->tt_pgs);
	return idx;
}

static inline uint64_t get_rmap_ent(struct conv_ftl *ftl, struct ppa *ppa)
{
	return ftl->rmap[ppa2pgidx(ftl, ppa)];
}

static inline void set_rmap_ent(struct conv_ftl *ftl, uint64_t lpn,
				struct ppa *ppa)
{
	ftl->rmap[ppa2pgidx(ftl, ppa)] = lpn;
}

/* ============================================================
 * PQ 콜백 (Greedy Min-Heap)
 * CB/CAT/RL은 Linear Scan하므로 PQ 정렬에 의존 안 함
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
static inline void consume_write_credit(struct conv_ftl *ftl)
{
	ftl->wfc.write_credits--;
}

static void foreground_gc(struct conv_ftl *conv_ftl);

static inline void check_and_refill_write_credit(struct conv_ftl *ftl)
{
	struct write_flow_control *wfc = &ftl->wfc;
	if (wfc->write_credits <= 0) {
		foreground_gc(ftl);
		wfc->write_credits += wfc->credits_to_refill;
	}
}

/* ============================================================
 * Victim 선택 ① Greedy — PQ Root O(1)
 * ============================================================ */
static struct line *select_victim_greedy(struct conv_ftl *ftl, bool force)
{
	struct line_mgmt *lm = &ftl->lm;
	struct line *v = pqueue_peek(lm->victim_line_pq);

	if (!v) return NULL;
	if (!force && v->vpc > (ftl->ssd->sp.pgs_per_line / 8)) return NULL;

	pqueue_pop(lm->victim_line_pq);
	v->pos = 0;
	lm->victim_line_cnt--;
	return v;
}

/* ============================================================
 * Victim 선택 ② Cost-Benefit — Linear Scan O(N)
 * Score = AgeWeight × IPC / (VPC + 1)
 * ============================================================ */
static struct line *select_victim_cb(struct conv_ftl *ftl, bool force)
{
	struct line_mgmt *lm = &ftl->lm;
	pqueue_t *q = lm->victim_line_pq;
	struct line *best = NULL;
	uint64_t max_s = 0, now = ktime_get_ns();
	size_t i;

	if (q->size == 0) return NULL;
	for (i = 1; i <= q->size; i++) {
		struct line *c = (struct line *)q->d[i];
		uint64_t age, s;
		if (!c) continue;
		age = (now > c->last_modified_time) ? (now - c->last_modified_time) : 0;
		s = get_age_weight(age) * (uint64_t)c->ipc / ((uint64_t)(c->vpc + 1));
		if (s > max_s) { max_s = s; best = c; }
	}
	if (best) { pqueue_remove(q, best); best->pos = 0; lm->victim_line_cnt--; }
	return best;
}

/* ============================================================
 * Victim 선택 ③ CAT — Linear Scan O(N)
 * Score = AgeWeight × IPC / ((VPC+1) × (EraseCount+1))
 * ============================================================ */
static struct line *select_victim_cat(struct conv_ftl *ftl, bool force)
{
	struct line_mgmt *lm = &ftl->lm;
	pqueue_t *q = lm->victim_line_pq;
	struct line *best = NULL;
	uint64_t max_s = 0, now = ktime_get_ns();
	size_t i;

	if (q->size == 0) return NULL;
	for (i = 1; i <= q->size; i++) {
		struct line *c = (struct line *)q->d[i];
		uint64_t age, s;
		if (!c) continue;
		age = (now > c->last_modified_time) ? (now - c->last_modified_time) : 0;
		s = get_age_weight(age) * (uint64_t)c->ipc /
		    ((uint64_t)(c->vpc + 1) * ((uint64_t)c->erase_cnt + 1));
		if (s > max_s) { max_s = s; best = c; }
	}
	if (best) { pqueue_remove(q, best); best->pos = 0; lm->victim_line_cnt--; }
	return best;
}

/* ============================================================
 * Victim 선택 ④ Unified (RL mode) — Linear Scan O(N)
 *
 * Score = apply_alpha(AgeWeight) × IPC / ((VPC+1) × apply_delta(EraseCount+1))
 *
 * alpha_level: 0→sqrt(aw)  1→aw  2→aw²
 * delta_level: 0→1(무시)  1→sqrt(ec+1)  2→(ec+1)  3→(ec+1)×sqrt(ec+1)
 * ============================================================ */
static uint64_t apply_alpha(uint64_t aw, uint32_t level)
{
	switch (level) {
	case 0:  return isqrt_u64(aw);         /* aw^0.5 */
	case 1:  return aw;                    /* aw^1.0 */
	default: return aw * aw;               /* aw^2.0 */
	}
}

static uint64_t apply_delta(uint64_t ec1, uint32_t level)
{
	switch (level) {
	case 0:  return 1;                              /* ec^0.0 = 무시 */
	case 1:  return isqrt_u64(ec1);                 /* ec^0.5 */
	case 2:  return ec1;                            /* ec^1.0 */
	default: return ec1 * isqrt_u64(ec1);           /* ec^1.5 */
	}
}

static struct line *select_victim_unified(struct conv_ftl *ftl, bool force)
{
	struct line_mgmt *lm = &ftl->lm;
	struct rl_config *rl = &ftl->rl;
	pqueue_t *q = lm->victim_line_pq;
	struct line *best = NULL;
	uint64_t max_s = 0, now = ktime_get_ns();
	size_t i;

	if (q->size == 0) return NULL;

	for (i = 1; i <= q->size; i++) {
		struct line *c = (struct line *)q->d[i];
		uint64_t age, aw, numerator, denominator, s;

		if (!c) continue;

		age = (now > c->last_modified_time) ? (now - c->last_modified_time) : 0;
		aw = get_age_weight(age);

		numerator = apply_alpha(aw, rl->alpha_level) * (uint64_t)c->ipc;
		denominator = (uint64_t)(c->vpc + 1) *
			      apply_delta((uint64_t)c->erase_cnt + 1, rl->delta_level);
		if (denominator == 0) denominator = 1;

		s = numerator / denominator;
		if (s > max_s) { max_s = s; best = c; }
	}

	if (best) { pqueue_remove(q, best); best->pos = 0; lm->victim_line_cnt--; }
	return best;
}

/* ============================================================
 * RL 인프라: State 이산화
 * ============================================================ */
static uint32_t rl_get_state(struct conv_ftl *ftl)
{
	struct line_mgmt *lm = &ftl->lm;
	struct ssdparams *spp = &ftl->ssd->sp;
	pqueue_t *q = lm->victim_line_pq;
	uint32_t s1, s2, s3, s4;
	uint32_t free_pct, i;
	uint64_t vpc_sum = 0;
	uint32_t vpc_avg_pct;
	uint32_t erase_max = 0, erase_min = UINT_MAX;

	/* S1: free line ratio */
	free_pct = (lm->free_line_cnt * 100) / lm->tt_lines;
	if (free_pct > 30)      s1 = 0; /* 여유 */
	else if (free_pct > 10) s1 = 1; /* 보통 */
	else                    s1 = 2; /* 긴급 */

	/* S2: victim 평균 VPC ratio */
	for (i = 1; i <= q->size; i++) {
		struct line *c = (struct line *)q->d[i];
		if (c) vpc_sum += c->vpc;
	}
	vpc_avg_pct = (q->size > 0) ?
		      (uint32_t)((vpc_sum * 100) / (q->size * spp->pgs_per_line)) : 50;
	if (vpc_avg_pct < 25)      s2 = 0; /* 더러움 (ipc 많음) */
	else if (vpc_avg_pct < 60) s2 = 1; /* 보통 */
	else                       s2 = 2; /* 깨끗 (vpc 높음) */

	/* S3: hot/cold ratio — avg_hot_degree 기반 간이 판별 */
	/* avg_hot_degree가 높으면 hot 위주 워크로드 */
	if (ftl->avg_hot_degree < 8)       s3 = 0; /* cold 위주 (×16에서 0.5 미만) */
	else if (ftl->avg_hot_degree < 48) s3 = 1; /* 혼합 */
	else                               s3 = 2; /* hot 위주 */

	/* S4: wear variance */
	for (i = 0; i < lm->tt_lines; i++) {
		uint32_t ec = lm->lines[i].erase_cnt;
		if (ec > erase_max) erase_max = ec;
		if (ec < erase_min) erase_min = ec;
	}
	s4 = ((erase_max - erase_min) > 20) ? 1 : 0; /* 불균등/균등 */

	return s1 * (RL_NUM_S2 * RL_NUM_S3 * RL_NUM_S4) +
	       s2 * (RL_NUM_S3 * RL_NUM_S4) +
	       s3 * RL_NUM_S4 + s4;
}

/* ============================================================
 * RL 인프라: Action 디코드
 * action_idx = a1 * (RL_NUM_A2 * RL_NUM_A3) + a2 * RL_NUM_A3 + a3
 * ============================================================ */
static void rl_decode_action(struct rl_config *rl, uint32_t action)
{
	rl->cur_action = action;
	rl->alpha_level     = action / (RL_NUM_A2 * RL_NUM_A3);
	rl->delta_level     = (action / RL_NUM_A3) % RL_NUM_A2;
	rl->hot_thresh_level = action % RL_NUM_A3;
}

/* ============================================================
 * RL 인프라: Epsilon-Greedy Action 선택
 * ============================================================ */
static uint32_t rl_select_action(struct rl_config *rl, uint32_t state)
{
	uint32_t rand_val, a, best_a = 0;
	int64_t best_q;

	/* epsilon-greedy: epsilon 확률로 random, 아니면 argmax Q */
	rand_val = get_random_u32() % 1000;
	if (rand_val < rl->epsilon) {
		/* Explore */
		return get_random_u32() % RL_NUM_ACTIONS;
	}

	/* Exploit: argmax Q(state, a) */
	best_q = rl->q_table[state][0];
	for (a = 1; a < RL_NUM_ACTIONS; a++) {
		if (rl->q_table[state][a] > best_q) {
			best_q = rl->q_table[state][a];
			best_a = a;
		}
	}
	return best_a;
}

/* ============================================================
 * RL 인프라: Reward 계산 + Q-table 갱신
 *
 * Reward = -copy_ratio × 1000 - wear_penalty × 10 + reclaim_bonus
 *
 * copy_ratio: copied_pages / (copied + reclaimed)  → 낮을수록 좋음
 * wear_penalty: (erase_max - erase_min) 변화량      → 클수록 나쁨
 * reclaim_bonus: 회수된 페이지 수 / pgs_per_line    → 클수록 좋음
 * ============================================================ */
static void rl_update(struct conv_ftl *ftl, uint32_t new_state)
{
	struct rl_config *rl = &ftl->rl;
	struct line_mgmt *lm = &ftl->lm;
	int64_t reward;
	int64_t max_q_next, old_q;
	uint64_t copied, reclaimed;
	uint32_t erase_max = 0, erase_min = UINT_MAX;
	uint32_t wear_delta;
	uint32_t i, best_a;

	/* 복사/회수 페이지 수 계산 */
	copied = ftl->gc_copied_pages - rl->prev_copied_pages;
	reclaimed = ftl->wfc.credits_to_refill; /* = victim의 ipc */

	/* wear 변화량 */
	for (i = 0; i < lm->tt_lines; i++) {
		uint32_t ec = lm->lines[i].erase_cnt;
		if (ec > erase_max) erase_max = ec;
		if (ec < erase_min) erase_min = ec;
	}
	wear_delta = (erase_max - erase_min) -
		     (rl->prev_erase_max - rl->prev_erase_min);

	/* Reward 계산 (×1000 고정소수점) */
	if (copied + reclaimed > 0) {
		reward = -((int64_t)copied * 1000) / (int64_t)(copied + reclaimed);
	} else {
		reward = 0;
	}
	reward -= (int64_t)wear_delta * 10;
	if (ftl->ssd->sp.pgs_per_line > 0)
		reward += ((int64_t)reclaimed * 200) / ftl->ssd->sp.pgs_per_line;

	/* max Q(new_state, a') */
	max_q_next = rl->q_table[new_state][0];
	for (best_a = 1; best_a < RL_NUM_ACTIONS; best_a++) {
		if (rl->q_table[new_state][best_a] > max_q_next)
			max_q_next = rl->q_table[new_state][best_a];
	}

	/* Q-learning update:
	 * Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]
	 * 모든 값 ×1000 고정소수점 */
	old_q = rl->q_table[rl->cur_state][rl->cur_action];
	rl->q_table[rl->cur_state][rl->cur_action] = old_q +
		RL_ALPHA * (reward + RL_GAMMA * max_q_next / RL_Q_SCALE - old_q) / RL_Q_SCALE;

	/* Epsilon decay */
	rl->epsilon = (rl->epsilon * RL_EPSILON_DECAY) / 1000;
	if (rl->epsilon < RL_EPSILON_MIN)
		rl->epsilon = RL_EPSILON_MIN;

	/* 통계 */
	rl->total_episodes++;
	rl->total_reward += reward;

	/* 스냅샷 갱신 */
	rl->prev_copied_pages = ftl->gc_copied_pages;
	rl->prev_erase_max = erase_max;
	rl->prev_erase_min = erase_min;
}

/* ============================================================
 * RL 인프라: Hot threshold 적용
 * hot_thresh_level: 0→avg×0.5, 1→avg×1.0, 2→avg×1.5
 * ============================================================ */
static bool rl_is_hot_page(struct conv_ftl *ftl, uint64_t lpn, uint64_t now)
{
	uint64_t deg = calc_hot_degree(&ftl->page_meta[lpn], now);
	uint64_t threshold;

	if (gc_mode != GC_MODE_RL)
		return is_hot_page(ftl, lpn, now); /* 비-RL 모드는 기존 기준 */

	/* RL 모드: hot_thresh_level에 따라 기준 조절 */
	switch (ftl->rl.hot_thresh_level) {
	case 0:  threshold = ftl->avg_hot_degree / 2; break; /* avg×0.5 (관대) */
	case 2:  threshold = ftl->avg_hot_degree * 3 / 2; break; /* avg×1.5 (엄격) */
	default: threshold = ftl->avg_hot_degree; break; /* avg×1.0 (기본) */
	}

	return (deg * 16) >= threshold;
}

/* ============================================================
 * 라인 초기화
 * ============================================================ */
static void init_lines(struct conv_ftl *ftl)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct line_mgmt *lm = &ftl->lm;
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
			.id = i, .ipc = 0, .vpc = 0, .pos = 0,
			.last_modified_time = 0, .erase_cnt = 0,
			.entry = LIST_HEAD_INIT(lm->lines[i].entry),
		};
		list_add_tail(&lm->lines[i].entry, &lm->free_line_list);
		lm->free_line_cnt++;
	}
	NVMEV_ASSERT(lm->free_line_cnt == spp->tt_lines);
	lm->victim_line_cnt = 0;
	lm->full_line_cnt = 0;
}

static void remove_lines(struct conv_ftl *ftl)
{
	pqueue_free(ftl->lm.victim_line_pq);
	vfree(ftl->lm.lines);
}

static void init_page_meta(struct conv_ftl *ftl)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	int i;

	ftl->page_meta = vmalloc(sizeof(struct page_meta) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++) {
		ftl->page_meta[i].update_cnt = 0;
		ftl->page_meta[i].last_write_time = 0;
	}
	ftl->avg_hot_degree = 16; /* 1.0 in ×16 fixed-point */
}

static void remove_page_meta(struct conv_ftl *ftl) { vfree(ftl->page_meta); }

static void init_rl(struct conv_ftl *ftl)
{
	struct rl_config *rl = &ftl->rl;

	memset(rl->q_table, 0, sizeof(rl->q_table));
	rl->cur_state = 0;
	rl->cur_action = 0;
	rl->alpha_level = 1;      /* 기본: aw^1.0 */
	rl->delta_level = 2;      /* 기본: ec^1.0 (= CAT) */
	rl->hot_thresh_level = 1; /* 기본: avg×1.0 */
	rl->epsilon = RL_EPSILON_INIT;
	rl->prev_copied_pages = 0;
	rl->prev_erase_max = 0;
	rl->prev_erase_min = 0;
	rl->total_episodes = 0;
	rl->total_reward = 0;
}

static void init_write_flow_control(struct conv_ftl *ftl)
{
	ftl->wfc.write_credits = ftl->ssd->sp.pgs_per_line;
	ftl->wfc.credits_to_refill = ftl->ssd->sp.pgs_per_line;
}

/* ============================================================
 * Write Pointer 관리
 * ============================================================ */
static inline void check_addr(int a, int max) { NVMEV_ASSERT(a >= 0 && a < max); }

static struct line *get_next_free_line(struct conv_ftl *ftl)
{
	struct line_mgmt *lm = &ftl->lm;
	struct line *cur = list_first_entry_or_null(&lm->free_line_list,
						   struct line, entry);
	if (!cur) { NVMEV_ERROR("No free line!\n"); return NULL; }
	list_del_init(&cur->entry);
	lm->free_line_cnt--;
	return cur;
}

static struct write_pointer *__get_wp(struct conv_ftl *ftl, uint32_t io_type)
{
	switch (io_type) {
	case USER_IO:    return &ftl->wp;
	case GC_IO_HOT:  return &ftl->gc_wp_hot;
	case GC_IO_COLD: return &ftl->gc_wp_cold;
	default: NVMEV_ASSERT(0); return NULL;
	}
}

static void prepare_write_pointer(struct conv_ftl *ftl, uint32_t io_type)
{
	struct write_pointer *wp = __get_wp(ftl, io_type);
	struct line *cur = get_next_free_line(ftl);
	NVMEV_ASSERT(wp && cur);
	*wp = (struct write_pointer){
		.curline = cur, .ch = 0, .lun = 0,
		.pg = 0, .blk = cur->id, .pl = 0,
	};
}

static void advance_write_pointer(struct conv_ftl *ftl, uint32_t io_type)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct line_mgmt *lm = &ftl->lm;
	struct write_pointer *wpp = __get_wp(ftl, io_type);

	check_addr(wpp->pg, spp->pgs_per_blk);
	wpp->pg++;
	if ((wpp->pg % spp->pgs_per_oneshotpg) != 0) return;

	wpp->pg -= spp->pgs_per_oneshotpg;
	check_addr(wpp->ch, spp->nchs);
	wpp->ch++;
	if (wpp->ch != spp->nchs) return;

	wpp->ch = 0;
	check_addr(wpp->lun, spp->luns_per_ch);
	wpp->lun++;
	if (wpp->lun != spp->luns_per_ch) return;

	wpp->lun = 0;
	wpp->pg += spp->pgs_per_oneshotpg;
	if (wpp->pg != spp->pgs_per_blk) return;

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
	wpp->curline = get_next_free_line(ftl);
	wpp->blk = wpp->curline->id;
	check_addr(wpp->blk, spp->blks_per_pl);
	NVMEV_ASSERT(wpp->pg == 0 && wpp->lun == 0 && wpp->ch == 0 && wpp->pl == 0);
}

static struct ppa get_new_page(struct conv_ftl *ftl, uint32_t io_type)
{
	struct write_pointer *wp = __get_wp(ftl, io_type);
	struct ppa ppa = { .ppa = 0 };
	ppa.g.ch = wp->ch; ppa.g.lun = wp->lun;
	ppa.g.pg = wp->pg; ppa.g.blk = wp->blk; ppa.g.pl = wp->pl;
	NVMEV_ASSERT(ppa.g.pl == 0);
	return ppa;
}

/* ============================================================
 * 매핑/역매핑 테이블
 * ============================================================ */
static void init_maptbl(struct conv_ftl *ftl)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	int i;
	ftl->maptbl = vmalloc(sizeof(struct ppa) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++) ftl->maptbl[i].ppa = UNMAPPED_PPA;
}
static void remove_maptbl(struct conv_ftl *ftl) { vfree(ftl->maptbl); }

static void init_rmap(struct conv_ftl *ftl)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	int i;
	ftl->rmap = vmalloc(sizeof(uint64_t) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++) ftl->rmap[i] = INVALID_LPN;
}
static void remove_rmap(struct conv_ftl *ftl) { vfree(ftl->rmap); }

/* ============================================================
 * FTL 인스턴스 초기화/제거
 * ============================================================ */
static void conv_init_ftl(struct conv_ftl *ftl, struct convparams *cpp,
			  struct ssd *ssd)
{
	ftl->cp = *cpp;
	ftl->ssd = ssd;
	ftl->gc_count = 0;
	ftl->gc_copied_pages = 0;

	init_maptbl(ftl);
	init_rmap(ftl);
	init_page_meta(ftl);
	init_lines(ftl);
	init_rl(ftl);

	prepare_write_pointer(ftl, USER_IO);
	prepare_write_pointer(ftl, GC_IO_HOT);
	prepare_write_pointer(ftl, GC_IO_COLD);
	init_write_flow_control(ftl);

	NVMEV_INFO("Init FTL: %d ch, %ld pgs, gc_mode=%d\n",
		   ssd->sp.nchs, ssd->sp.tt_pgs, gc_mode);
}

static void conv_remove_ftl(struct conv_ftl *ftl)
{
	remove_lines(ftl);
	remove_rmap(ftl);
	remove_maptbl(ftl);
	remove_page_meta(ftl);
}

static void conv_init_params(struct convparams *cpp)
{
	cpp->op_area_pcent = OP_AREA_PERCENT;
	cpp->gc_thres_lines = 3; /* 3 open blocks: user, gc_hot, gc_cold */
	cpp->gc_thres_lines_high = 3;
	cpp->enable_gc_delay = 1;
	cpp->pba_pcent = (int)((1 + cpp->op_area_pcent) * 100);
}

/* ============================================================
 * 네임스페이스 초기화/제거
 * ============================================================ */
void conv_init_namespace(struct nvmev_ns *ns, uint32_t id, uint64_t size,
			 void *mapped_addr, uint32_t cpu_nr_dispatcher)
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

	NVMEV_INFO("FTL physical=%lld logical=%lld (ratio=%d) gc_mode=%d\n",
		   size, ns->size, cpp.pba_pcent, gc_mode);
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
 * PPA/LPN 유효성 검사
 * ============================================================ */
static inline bool valid_ppa(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	if (ppa->g.ch < 0 || ppa->g.ch >= spp->nchs) return false;
	if (ppa->g.lun < 0 || ppa->g.lun >= spp->luns_per_ch) return false;
	if (ppa->g.pl < 0 || ppa->g.pl >= spp->pls_per_lun) return false;
	if (ppa->g.blk < 0 || ppa->g.blk >= spp->blks_per_pl) return false;
	if (ppa->g.pg < 0 || ppa->g.pg >= spp->pgs_per_blk) return false;
	return true;
}
static inline bool valid_lpn(struct conv_ftl *ftl, uint64_t lpn)
{
	return lpn < ftl->ssd->sp.tt_pgs;
}
static inline bool mapped_ppa(struct ppa *ppa) { return ppa->ppa != UNMAPPED_PPA; }
static inline struct line *get_line(struct conv_ftl *ftl, struct ppa *ppa)
{
	return &ftl->lm.lines[ppa->g.blk];
}

/* ============================================================
 * 페이지 상태 관리
 * ============================================================ */
static void mark_page_invalid(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct line_mgmt *lm = &ftl->lm;
	struct nand_block *blk;
	struct nand_page *pg;
	struct line *line;
	bool was_full = false;

	pg = get_pg(ftl->ssd, ppa);
	NVMEV_ASSERT(pg->status == PG_VALID);
	pg->status = PG_INVALID;

	blk = get_blk(ftl->ssd, ppa);
	blk->ipc++; blk->vpc--;

	line = get_line(ftl, ppa);
	if (line->vpc == spp->pgs_per_line) { was_full = true; }
	line->ipc++;
	if (line->pos)
		pqueue_change_priority(lm->victim_line_pq, line->vpc - 1, line);
	else
		line->vpc--;

	if (was_full) {
		list_del_init(&line->entry);
		lm->full_line_cnt--;
		pqueue_insert(lm->victim_line_pq, line);
		lm->victim_line_cnt++;
	}
	line->last_modified_time = ktime_get_ns();
}

static void mark_page_valid(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct nand_block *blk;
	struct nand_page *pg;
	struct line *line;

	pg = get_pg(ftl->ssd, ppa);
	NVMEV_ASSERT(pg->status == PG_FREE);
	pg->status = PG_VALID;
	blk = get_blk(ftl->ssd, ppa);
	blk->vpc++;
	line = get_line(ftl, ppa);
	line->vpc++;
}

static void mark_block_free(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct nand_block *blk = get_blk(ftl->ssd, ppa);
	int i;
	for (i = 0; i < spp->pgs_per_blk; i++) blk->pg[i].status = PG_FREE;
	blk->ipc = 0; blk->vpc = 0; blk->erase_cnt++;
}

static void mark_line_free(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct line *line = get_line(ftl, ppa);
	line->ipc = 0; line->vpc = 0; line->erase_cnt++;
	list_add_tail(&line->entry, &ftl->lm.free_line_list);
	ftl->lm.free_line_cnt++;
}

/* ============================================================
 * GC 읽기/쓰기 (Hot/Cold 분기 포함)
 * ============================================================ */
static void gc_read_page(struct conv_ftl *ftl, struct ppa *ppa)
{
	if (ftl->cp.enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO, .cmd = NAND_READ, .stime = 0,
			.xfer_size = ftl->ssd->sp.pgsz,
			.interleave_pci_dma = false, .ppa = ppa,
		};
		ssd_advance_nand(ftl->ssd, &gcr);
	}
}

static uint64_t gc_write_page(struct conv_ftl *ftl, struct ppa *old_ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	uint64_t lpn = get_rmap_ent(ftl, old_ppa);
	uint64_t now = ktime_get_ns();
	uint32_t wp_type;
	struct ppa new_ppa;

	NVMEV_ASSERT(valid_lpn(ftl, lpn));

	/* Hot/Cold 분기 (RL 모드면 rl_is_hot_page, 아니면 is_hot_page) */
	wp_type = rl_is_hot_page(ftl, lpn, now) ? GC_IO_HOT : GC_IO_COLD;

	new_ppa = get_new_page(ftl, wp_type);
	set_maptbl_ent(ftl, lpn, &new_ppa);
	set_rmap_ent(ftl, lpn, &new_ppa);
	mark_page_valid(ftl, &new_ppa);
	ftl->gc_copied_pages++;
	advance_write_pointer(ftl, wp_type);

	if (ftl->cp.enable_gc_delay) {
		struct nand_cmd gcw = {
			.type = GC_IO, .cmd = NAND_NOP, .stime = 0,
			.interleave_pci_dma = false, .ppa = &new_ppa,
		};
		if (last_pg_in_wordline(ftl, &new_ppa)) {
			gcw.cmd = NAND_WRITE;
			gcw.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg;
		}
		ssd_advance_nand(ftl->ssd, &gcw);
	}
	return 0;
}

static void clean_one_flashpg(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct nand_page *pg;
	int cnt = 0, i;
	struct ppa copy = *ppa;

	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg = get_pg(ftl->ssd, &copy);
		NVMEV_ASSERT(pg->status != PG_FREE);
		if (pg->status == PG_VALID) cnt++;
		copy.g.pg++;
	}
	copy = *ppa;
	if (cnt <= 0) return;

	if (ftl->cp.enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO, .cmd = NAND_READ, .stime = 0,
			.xfer_size = spp->pgsz * cnt,
			.interleave_pci_dma = false, .ppa = &copy,
		};
		ssd_advance_nand(ftl->ssd, &gcr);
	}
	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg = get_pg(ftl->ssd, &copy);
		if (pg->status == PG_VALID) gc_write_page(ftl, &copy);
		copy.g.pg++;
	}
}

/* ============================================================
 * do_gc — 전략 선택 + RL 통합
 *
 * 비-RL 모드: victim_select_fn을 직접 받아 사용
 * RL 모드:   select_fn 무시, RL이 state→action→unified scoring 수행
 * ============================================================ */
static int do_gc(struct conv_ftl *ftl, bool force, victim_select_fn select_fn)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct line *victim = NULL;
	struct ppa ppa;
	int flashpg;
	uint32_t new_state;

	/* RL 모드: GC 전에 state 관찰 → action 선택 → 파라미터 적용 */
	if (gc_mode == GC_MODE_RL) {
		struct rl_config *rl = &ftl->rl;
		uint32_t state = rl_get_state(ftl);
		uint32_t action = rl_select_action(rl, state);

		rl->cur_state = state;
		rl_decode_action(rl, action);

		/* unified scoring으로 victim 선택 */
		victim = select_victim_unified(ftl, force);
	} else {
		victim = select_fn(ftl, force);
	}

	if (!victim) return -1;

	ftl->gc_count++;
	ppa.g.blk = victim->id;
	ftl->wfc.credits_to_refill = victim->ipc;

	/* 블록 청소 */
	for (flashpg = 0; flashpg < spp->flashpgs_per_blk; flashpg++) {
		int ch, lun;
		ppa.g.pg = flashpg * spp->pgs_per_flashpg;
		for (ch = 0; ch < spp->nchs; ch++) {
			for (lun = 0; lun < spp->luns_per_ch; lun++) {
				struct nand_lun *lunp;
				ppa.g.ch = ch; ppa.g.lun = lun; ppa.g.pl = 0;
				lunp = get_lun(ftl->ssd, &ppa);
				clean_one_flashpg(ftl, &ppa);

				if (flashpg == (spp->flashpgs_per_blk - 1)) {
					mark_block_free(ftl, &ppa);
					if (ftl->cp.enable_gc_delay) {
						struct nand_cmd gce = {
							.type = GC_IO, .cmd = NAND_ERASE,
							.stime = 0, .interleave_pci_dma = false,
							.ppa = &ppa,
						};
						ssd_advance_nand(ftl->ssd, &gce);
					}
					lunp->gc_endtime = lunp->next_lun_avail_time;
				}
			}
		}
	}
	mark_line_free(ftl, &ppa);

	/* RL 모드: GC 완료 후 reward 계산 + Q-table 갱신 */
	if (gc_mode == GC_MODE_RL) {
		new_state = rl_get_state(ftl);
		rl_update(ftl, new_state);
	}

	return 0;
}

/* ============================================================
 * foreground_gc — gc_mode에 따라 전략 분기
 * ============================================================ */
static void foreground_gc(struct conv_ftl *ftl)
{
	if (!should_gc_high(ftl))
		return;

	switch (gc_mode) {
	case GC_MODE_GREEDY:
		do_gc(ftl, true, select_victim_greedy);
		break;
	case GC_MODE_CB:
		do_gc(ftl, true, select_victim_cb);
		break;
	case GC_MODE_CAT:
		do_gc(ftl, true, select_victim_cat);
		break;
	case GC_MODE_RL:
		do_gc(ftl, true, NULL); /* RL은 내부에서 unified 사용 */
		break;
	default:
		do_gc(ftl, true, select_victim_greedy);
		break;
	}
}

/* ============================================================
 * NVMe Read
 * ============================================================ */
static bool is_same_flash_page(struct conv_ftl *ftl,
			       struct ppa p1, struct ppa p2)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	return (p1.h.blk_in_ssd == p2.h.blk_in_ssd) &&
	       (p1.g.pg / spp->pgs_per_flashpg == p2.g.pg / spp->pgs_per_flashpg);
}

static bool conv_read(struct nvmev_ns *ns, struct nvmev_request *req,
		      struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *ftl = &conv_ftls[0];
	struct ssdparams *spp = &ftl->ssd->sp;
	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba, nr_lba = cmd->rw.length + 1;
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;
	uint64_t lpn, nsecs_start = req->nsecs_start;
	uint64_t nsecs_completed, nsecs_latest = nsecs_start;
	uint32_t xfer_size, i, nr_parts = ns->nr_parts;
	struct ppa prev_ppa;
	struct nand_cmd srd = { .type = USER_IO, .cmd = NAND_READ,
				.stime = nsecs_start, .interleave_pci_dma = true };

	if ((end_lpn / nr_parts) >= spp->tt_pgs) return false;
	srd.stime += (LBA_TO_BYTE(nr_lba) <= KB(4) * nr_parts) ?
		     spp->fw_4kb_rd_lat : spp->fw_rd_lat;

	for (i = 0; (i < nr_parts) && (start_lpn <= end_lpn); i++, start_lpn++) {
		ftl = &conv_ftls[start_lpn % nr_parts];
		xfer_size = 0;
		prev_ppa = get_maptbl_ent(ftl, start_lpn / nr_parts);

		for (lpn = start_lpn; lpn <= end_lpn; lpn += nr_parts) {
			struct ppa cur = get_maptbl_ent(ftl, lpn / nr_parts);
			if (!mapped_ppa(&cur) || !valid_ppa(ftl, &cur)) continue;
			if (mapped_ppa(&prev_ppa) && is_same_flash_page(ftl, cur, prev_ppa)) {
				xfer_size += spp->pgsz; continue;
			}
			if (xfer_size > 0) {
				srd.xfer_size = xfer_size; srd.ppa = &prev_ppa;
				nsecs_completed = ssd_advance_nand(ftl->ssd, &srd);
				nsecs_latest = max(nsecs_completed, nsecs_latest);
			}
			xfer_size = spp->pgsz; prev_ppa = cur;
		}
		if (xfer_size > 0) {
			srd.xfer_size = xfer_size; srd.ppa = &prev_ppa;
			nsecs_completed = ssd_advance_nand(ftl->ssd, &srd);
			nsecs_latest = max(nsecs_completed, nsecs_latest);
		}
	}
	ret->nsecs_target = nsecs_latest;
	ret->status = NVME_SC_SUCCESS;
	return true;
}

/* ============================================================
 * NVMe Write — page_meta 갱신 + avg_hot_degree EMA
 * ============================================================ */
static bool conv_write(struct nvmev_ns *ns, struct nvmev_request *req,
		       struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *ftl = &conv_ftls[0];
	struct ssdparams *spp = &ftl->ssd->sp;
	struct buffer *wbuf = ftl->ssd->write_buffer;
	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba, nr_lba = cmd->rw.length + 1;
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;
	uint64_t lpn, nsecs_latest, nsecs_xfer;
	uint32_t nr_parts = ns->nr_parts, alloc;
	struct nand_cmd swr = { .type = USER_IO, .cmd = NAND_WRITE,
				.interleave_pci_dma = false,
				.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg };

	if ((end_lpn / nr_parts) >= spp->tt_pgs) return false;
	alloc = buffer_allocate(wbuf, LBA_TO_BYTE(nr_lba));
	if (alloc < LBA_TO_BYTE(nr_lba)) return false;

	nsecs_latest = ssd_advance_write_buffer(ftl->ssd, req->nsecs_start,
						LBA_TO_BYTE(nr_lba));
	nsecs_xfer = nsecs_latest;
	swr.stime = nsecs_latest;

	for (lpn = start_lpn; lpn <= end_lpn; lpn++) {
		uint64_t local_lpn, nsecs_done = 0;
		struct ppa ppa;
		struct page_meta *pm;
		uint64_t degree;

		ftl = &conv_ftls[lpn % nr_parts];
		local_lpn = lpn / nr_parts;

		ppa = get_maptbl_ent(ftl, local_lpn);
		if (mapped_ppa(&ppa)) {
			mark_page_invalid(ftl, &ppa);
			set_rmap_ent(ftl, INVALID_LPN, &ppa);
		}

		ppa = get_new_page(ftl, USER_IO);
		set_maptbl_ent(ftl, local_lpn, &ppa);
		set_rmap_ent(ftl, local_lpn, &ppa);
		mark_page_valid(ftl, &ppa);

		/* page_meta 갱신 */
		pm = &ftl->page_meta[local_lpn];
		pm->update_cnt++;
		pm->last_write_time = ktime_get_ns();

		/* avg_hot_degree EMA 갱신 (×16 고정소수점) */
		degree = calc_hot_degree(pm, pm->last_write_time);
		ftl->avg_hot_degree = (ftl->avg_hot_degree * 15 + degree * 16) / 16;

		advance_write_pointer(ftl, USER_IO);

		if (last_pg_in_wordline(ftl, &ppa)) {
			swr.ppa = &ppa;
			nsecs_done = ssd_advance_nand(ftl->ssd, &swr);
			nsecs_latest = max(nsecs_done, nsecs_latest);
			schedule_internal_operation(req->sq_id, nsecs_done, wbuf,
						   spp->pgs_per_oneshotpg * spp->pgsz);
		}
		consume_write_credit(ftl);
		check_and_refill_write_credit(ftl);
	}

	if ((cmd->rw.control & NVME_RW_FUA) || !spp->write_early_completion)
		ret->nsecs_target = nsecs_latest;
	else
		ret->nsecs_target = nsecs_xfer;
	ret->status = NVME_SC_SUCCESS;
	return true;
}

/* ============================================================
 * NVMe Flush — GC/RL 통계 출력
 * ============================================================ */
static void conv_flush(struct nvmev_ns *ns, struct nvmev_request *req,
		       struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	uint64_t start, latest;
	uint64_t total_gc = 0, total_cp = 0;
	uint32_t i;

	start = local_clock();
	latest = start;
	for (i = 0; i < ns->nr_parts; i++) {
		latest = max(latest, ssd_next_idle_time(conv_ftls[i].ssd));
		total_gc += conv_ftls[i].gc_count;
		total_cp += conv_ftls[i].gc_copied_pages;
	}

	printk(KERN_INFO "NVMeVirt: [FLUSH] gc_mode=%d gc=%llu copied=%llu avg=%llu\n",
	       gc_mode, total_gc, total_cp,
	       total_gc > 0 ? total_cp / total_gc : 0);

	/* RL 통계 */
	if (gc_mode == GC_MODE_RL && ns->nr_parts > 0) {
		struct rl_config *rl = &conv_ftls[0].rl;
		printk(KERN_INFO "NVMeVirt: [RL] episodes=%llu epsilon=%u "
		       "avg_reward=%lld alpha=%u delta=%u hot_thresh=%u\n",
		       rl->total_episodes, rl->epsilon,
		       rl->total_episodes > 0 ?
			(int64_t)(rl->total_reward / (int64_t)rl->total_episodes) : 0,
		       rl->alpha_level, rl->delta_level, rl->hot_thresh_level);
	}

	/* Wear 통계 */
	if (ns->nr_parts > 0) {
		struct line_mgmt *lm = &conv_ftls[0].lm;
		uint64_t sum = 0;
		uint32_t mx = 0, mn = UINT_MAX, j;
		for (j = 0; j < lm->tt_lines; j++) {
			uint32_t ec = lm->lines[j].erase_cnt;
			sum += ec;
			if (ec > mx) mx = ec;
			if (ec < mn) mn = ec;
		}
		printk(KERN_INFO "NVMeVirt: [Wear] avg=%llu min=%u max=%u\n",
		       lm->tt_lines > 0 ? sum / lm->tt_lines : 0,
		       mn == UINT_MAX ? 0 : mn, mx);
	}

	ret->status = NVME_SC_SUCCESS;
	ret->nsecs_target = latest;
}

/* ============================================================
 * IO Dispatcher
 * ============================================================ */
bool conv_proc_nvme_io_cmd(struct nvmev_ns *ns, struct nvmev_request *req,
			   struct nvmev_result *ret)
{
	struct nvme_command *cmd = req->cmd;
	NVMEV_ASSERT(ns->csi == NVME_CSI_NVM);

	switch (cmd->common.opcode) {
	case nvme_cmd_write:
		if (!conv_write(ns, req, ret)) return false;
		break;
	case nvme_cmd_read:
		if (!conv_read(ns, req, ret)) return false;
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