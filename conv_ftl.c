// SPDX-License-Identifier: GPL-2.0-only
/*
 * ================================================================
 * conv_ftl.c - Conventional FTL with RL-based GC Policy Tuning
 * ================================================================
 *
 * [이 파일의 역할]
 *   NVMeVirt 가상 SSD의 FTL(Flash Translation Layer) 로직 전체를 구현.
 *   호스트로부터 NVMe Read/Write/Flush 커맨드를 받아 처리하며,
 *   내부적으로 주소 매핑, GC(Garbage Collection), RL(Reinforcement Learning)
 *   기반 정책 튜닝을 수행한다.
 *
 * [GC 모드] (insmod gc_mode=N)
 *
 *   ┌─────────┬──────────────────────────────────────────────────┐
 *   │ Mode 0  │ Greedy: VPC 최소 선택. PQ Root O(1).             │
 *   │ Mode 1  │ CB: Age×IPC/(VPC+1). Linear Scan O(N).          │
 *   │ Mode 2  │ CAT: Age×IPC/((VPC+1)×(EC+1)). Scan O(N).      │
 *   │ Mode 3  │ RL: AW^α×IPC/((VPC+1)×(EC+1)^δ).               │
 *   │         │ Q-table이 α, δ, hot_threshold를 자동 조절.       │
 *   └─────────┴──────────────────────────────────────────────────┘
 *
 * [핵심 기능]
 *   1) 4가지 GC victim 선택 전략 (insmod 파라미터로 선택)
 *   2) M6 Fine-Grained Hot/Cold Redistribution (모든 모드에서 활성)
 *      - per-LPN page_meta로 hot degree 추적
 *      - GC 시 hot → gc_wp_hot, cold → gc_wp_cold로 분리 기록
 *   3) Q-table RL Agent (gc_mode=3 전용)
 *      - State: 공간압박 × victim상태 × hot편향 × wear균등 (54칸)
 *      - Action: α레벨 × δ레벨 × hot기준 (36칸)
 *      - Reward: -복사비율 - wear불균형 + 회수보너스
 *
 * [참고 논문]
 *   "Cleaning policies in mobile computers using flash memory"
 *   (M.-L. Chiang, R.-C. Chang, 1999)
 *
 * ================================================================
 */

#include <linux/vmalloc.h>       /* vmalloc/vfree: 대용량 배열 할당 */
#include <linux/ktime.h>         /* ktime_get_ns(): 나노초 시각 */
#include <linux/sched/clock.h>   /* local_clock(): flush에서 사용 */
#include <linux/moduleparam.h>   /* module_param(): insmod 파라미터 */
#include <linux/random.h>        /* get_random_u32(): RL epsilon-greedy */

#include "nvmev.h"      /* NVMeVirt 공통 (NVMEV_ASSERT, NVMEV_INFO 등) */
#include "conv_ftl.h"   /* 이 파일의 모든 구조체/상수 정의 */

/* ================================================================
 * 모듈 파라미터: insmod 시 gc_mode=N 으로 GC 전략 선택
 *
 * 기본값: GC_MODE_CAT(2)
 * 런타임 변경: echo 3 > /sys/module/nvmev/parameters/gc_mode
 *             (하지만 GC 도중 변경은 위험할 수 있음)
 *
 * 0644: root만 쓰기, 모두 읽기 가능
 * ================================================================ */
static int gc_mode = GC_MODE_CAT;
module_param(gc_mode, int, 0644);
MODULE_PARM_DESC(gc_mode, "0=Greedy 1=CB 2=CAT 3=RL");

/* ================================================================
 * 정수 제곱근 (Newton's method)
 *
 * [필요한 이유]
 *   커널에서는 float/double을 쓸 수 없다 (FPU context 문제).
 *   RL의 apply_alpha(), apply_delta()에서 x^0.5을 계산해야 하므로
 *   정수 제곱근이 필요하다.
 *
 * [알고리즘]
 *   Newton-Raphson: y = (x + n/x) / 2 를 수렴할 때까지 반복.
 *   O(log n) 반복으로 정확한 floor(sqrt(n))을 구함.
 *
 * [사용처]
 *   apply_alpha(aw, 0): sqrt(aw) → α=0.5에 해당
 *   apply_delta(ec, 1): sqrt(ec+1) → δ=0.5에 해당
 *   apply_delta(ec, 3): (ec+1) × sqrt(ec+1) → δ=1.5에 해당
 *
 * ================================================================ */
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

/* ================================================================
 * Age 가중치 함수 (논문 Fig.7 기반)
 *
 * [논문 원문]
 *   "age is normalized by a heuristic transformation function,
 *    aiming to avoid age being too large to overemphasize age
 *    and affect segment choosing."
 *
 * [설계]
 *   age(나노초)를 4개 구간으로 나누어 고정 가중치를 반환.
 *   실제 age 값(수백억 ns)을 직접 쓰면 score가 overflow하거나
 *   age만 지배적이 되므로, 구간별 정규화가 필요.
 *
 *   ┌──────────────┬──────────┬──────────────────────────┐
 *   │ 구간          │ 가중치   │ 의미                      │
 *   ├──────────────┼──────────┼──────────────────────────┤
 *   │ 0 ~ 100ms    │ 1        │ Very Hot: 방금 기록됨     │
 *   │ 100ms ~ 5s   │ 5        │ Hot: 아직 활성 상태       │
 *   │ 5s ~ 60s     │ 20       │ Warm: 식어가는 중         │
 *   │ 60s ~        │ 100      │ Cold/Frozen: 정적 데이터  │
 *   └──────────────┴──────────┴──────────────────────────┘
 *
 * [사용처]
 *   select_victim_cb():      Score 분자에 직접 곱함
 *   select_victim_cat():     Score 분자에 직접 곱함
 *   select_victim_unified(): apply_alpha()를 거쳐 α 지수 적용 후 곱함
 *   calc_hot_degree():       분모에 사용 (age가 크면 hot degree 감소)
 *
 * [구간 경계값 튜닝]
 *   현재 100ms / 5s / 60s는 경험적 설정.
 *   워크로드에 따라 조절 가능. RL에서는 이것까지 action으로
 *   만들 수 있으나, 현재 action 공간이 이미 36이라 제외함.
 *
 * ================================================================ */
#define MS_TO_NS(x)  ((uint64_t)(x) * 1000000ULL)
#define SEC_TO_NS(x) ((uint64_t)(x) * 1000000000ULL)

#define TH_VERY_HOT  MS_TO_NS(100)   /* 100 밀리초 */
#define TH_HOT       SEC_TO_NS(5)    /* 5 초 */
#define TH_WARM      SEC_TO_NS(60)   /* 60 초 */

static uint64_t get_age_weight(uint64_t age_ns)
{
	if (age_ns < TH_VERY_HOT)  return 1;    /* 0 ~ 100ms */
	if (age_ns < TH_HOT)       return 5;    /* 100ms ~ 5s */
	if (age_ns < TH_WARM)      return 20;   /* 5s ~ 60s */
	return 100;                              /* 60s ~ */
}

/* ================================================================
 * Hot Degree 계산
 *
 * [논문 Section 3.2]
 *   "The hot degree of a block is defined as the number of times
 *    the block has been updated and decreases as the block's age grows."
 *
 * [공식]
 *   hot_degree = update_cnt × HD_SCALE / get_age_weight(age)
 *
 *   - update_cnt: 이 LPN이 덮어쓰기된 횟수 (page_meta.update_cnt)
 *   - age: 마지막 쓰기 이후 경과 시간 (now - page_meta.last_write_time)
 *   - HD_SCALE(1000): 정수 나눗셈 정밀도를 위한 스케일링
 *
 * [예시]
 *   LPN이 10번 쓰이고, 30초 전에 마지막 쓰기:
 *     age_weight = 20 (Warm 구간)
 *     hot_degree = 10 × 1000 / 20 = 500
 *
 *   LPN이 10번 쓰이고, 0.05초 전에 마지막 쓰기:
 *     age_weight = 1 (Very Hot 구간)
 *     hot_degree = 10 × 1000 / 1 = 10000  ← 매우 hot!
 *
 * [사용처]
 *   is_hot_page(): hot_degree × 16 >= avg_hot_degree 이면 hot
 *   rl_is_hot_page(): hot_thresh_level에 따라 기준 조절
 *   conv_write(): EMA 평균 갱신에 사용
 *
 * ================================================================ */
#define HD_SCALE 1000

static uint64_t calc_hot_degree(struct page_meta *pm, uint64_t now)
{
	uint64_t age, aw;

	if (pm->update_cnt == 0)
		return 0;  /* 한 번도 쓰이지 않은 LPN */

	age = (now > pm->last_write_time) ? (now - pm->last_write_time) : 0;
	aw = get_age_weight(age);
	return ((uint64_t)pm->update_cnt * HD_SCALE) / aw;
}

/*
 * is_hot_page() - 비-RL 모드에서의 hot/cold 판별 (고정 기준)
 *
 * avg_hot_degree는 ×16 고정소수점이므로, degree도 ×16하여 비교.
 * 예: avg_hot_degree=800(실제 50), degree=60 → 60×16=960 > 800 → hot
 *
 * [사용처] rl_is_hot_page()에서 gc_mode != RL일 때 위임됨.
 */
static bool is_hot_page(struct conv_ftl *ftl, uint64_t lpn, uint64_t now)
{
	uint64_t deg = calc_hot_degree(&ftl->page_meta[lpn], now);
	return (deg * 16) >= ftl->avg_hot_degree;
}

/* ================================================================
 * 기본 유틸리티 함수
 * ================================================================ */

/*
 * last_pg_in_wordline() - 현재 페이지가 워드라인의 마지막인지 확인
 *
 * [배경]
 *   NAND는 워드라인(oneshot program) 단위로 프로그래밍함.
 *   워드라인의 마지막 페이지에 도달해야 실제 NAND_WRITE 발행.
 *   그 전까지는 NAND_NOP (데이터를 page buffer에 모으는 중).
 *
 * [사용처]
 *   conv_write():    swr.cmd = NAND_WRITE 조건
 *   gc_write_page(): gcw.cmd = NAND_WRITE 조건
 */
static inline bool last_pg_in_wordline(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	return (ppa->g.pg % spp->pgs_per_oneshotpg) == (spp->pgs_per_oneshotpg - 1);
}

/*
 * should_gc() / should_gc_high() - GC 트리거 조건
 *
 * should_gc():      background GC용 (현재 사용하지 않지만 향후 확장 대비)
 * should_gc_high(): foreground GC 트리거. check_and_refill_write_credit()에서 호출.
 *
 * free_line_cnt가 gc_thres_lines(3) 이하이면 true.
 * 3인 이유: open block 3개(user, gc_hot, gc_cold)가 항상 필요하므로.
 */
static bool should_gc(struct conv_ftl *ftl)
{
	return (ftl->lm.free_line_cnt <= ftl->cp.gc_thres_lines);
}

static inline bool should_gc_high(struct conv_ftl *ftl)
{
	return ftl->lm.free_line_cnt <= ftl->cp.gc_thres_lines_high;
}

/* ================================================================
 * 매핑 테이블 접근 함수
 *
 * get_maptbl_ent(lpn): LPN → PPA 조회
 * set_maptbl_ent(lpn, ppa): LPN → PPA 기록
 *
 * [호출 시점]
 *   conv_read():     get → PPA로 NAND 읽기
 *   conv_write():    get(기존 매핑 확인) → set(새 PPA 기록)
 *   gc_write_page(): set(valid page의 새 위치 기록)
 *
 * ================================================================ */
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

/*
 * ppa2pgidx() - PPA 구조체를 선형 페이지 인덱스로 변환
 *
 * 채널/LUN/플레인/블록/페이지 주소를 하나의 정수로 펼침.
 * rmap 배열의 인덱스로 사용됨.
 *
 * 공식: ch × pgs_per_ch + lun × pgs_per_lun + pl × pgs_per_pl
 *       + blk × pgs_per_blk + pg
 */
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

/*
 * get_rmap_ent() / set_rmap_ent() - 역매핑 테이블 접근
 *
 * PPA → LPN 조회/기록.
 * GC 시 "이 물리 페이지에 뭐가 저장돼있지?"를 알기 위해 필요.
 * 실제 SSD에서는 NAND의 OOB(Out-Of-Band) 영역에 저장되는 정보.
 */
static inline uint64_t get_rmap_ent(struct conv_ftl *ftl, struct ppa *ppa)
{
	return ftl->rmap[ppa2pgidx(ftl, ppa)];
}

static inline void set_rmap_ent(struct conv_ftl *ftl, uint64_t lpn,
				struct ppa *ppa)
{
	ftl->rmap[ppa2pgidx(ftl, ppa)] = lpn;
}

/* ================================================================
 * PQ(Priority Queue) 콜백 함수
 *
 * pqueue 라이브러리가 내부적으로 호출하는 콜백들.
 * Greedy는 vpc 기준 Min-Heap으로 동작:
 *   - cmp: next > curr이면 교환 → 작은 vpc가 루트로
 *   - get_pri: line의 vpc를 우선순위로 사용
 *   - set_pri: pqueue_change_priority() 시 vpc 갱신
 *   - get_pos/set_pos: 큐 내부 인덱스 관리
 *
 * CB/CAT/RL은 전체 큐를 Linear Scan하므로 PQ 정렬 순서에 의존하지 않음.
 * 다만 중간 원소 제거(pqueue_remove)와 우선순위 갱신(pqueue_change_priority)은
 * PQ 구조를 통해 이루어지므로 콜백은 여전히 필요.
 *
 * ================================================================ */
static inline int victim_line_cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
	return (next > curr); /* 부모(next)가 자식(curr)보다 크면 교환 → Min-Heap */
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

/* ================================================================
 * 쓰기 크레딧 관리
 *
 * [개념]
 *   호스트가 무한히 쓰면 free line이 고갈된다.
 *   write_credits가 0이 되면 foreground_gc()를 강제 호출하여
 *   공간을 확보한 후 크레딧을 리필한다.
 *
 * [흐름]
 *   conv_write()의 매 LPN마다:
 *     ① consume_write_credit() → credits--
 *     ② check_and_refill_write_credit():
 *        credits ≤ 0 이면:
 *          foreground_gc() 호출 (GC 수행)
 *          credits += credits_to_refill (이전 GC의 회수량)
 *
 * ================================================================ */
static inline void consume_write_credit(struct conv_ftl *ftl)
{
	ftl->wfc.write_credits--;
}

/* forward declaration: foreground_gc()가 뒤에 정의되므로 */
static void foreground_gc(struct conv_ftl *ftl);

static inline void check_and_refill_write_credit(struct conv_ftl *ftl)
{
	struct write_flow_control *wfc = &ftl->wfc;
	if (wfc->write_credits <= 0) {
		foreground_gc(ftl);    /* GC 수행으로 공간 확보 */
		wfc->write_credits += wfc->credits_to_refill; /* 크레딧 리필 */
	}
}

/* ================================================================
 *
 *           ★★★ GC VICTIM 선택 전략 ★★★
 *
 * 이하 4개 함수가 이 FTL의 핵심 알고리즘.
 * do_gc()에서 victim_select_fn 파라미터로 전달받아 호출되거나,
 * RL 모드에서는 select_victim_unified()가 내부 호출됨.
 *
 * ================================================================ */

/* ================================================================
 * Victim 선택 ① Greedy
 *
 * [알고리즘]
 *   PQ(Min-Heap)의 루트를 O(1)로 꺼냄.
 *   vpc가 가장 작은 라인 = 복사할 valid page가 가장 적은 라인.
 *
 * [장점]
 *   - 빠름: O(1) + O(log N) for pop
 *   - uniform random 워크로드에서 최적 (모든 블록의 age가 비슷)
 *
 * [단점]
 *   - age 무시: locality 높은 워크로드에서 hot 블록을 잘못 선택
 *     → 복사 직후 hot 데이터가 다시 무효화 → "useless migration" (논문 Fig.6)
 *   - erase count 무시: 특정 블록만 반복 GC → wear 불균형
 *
 * [force 파라미터]
 *   false: vpc > pgs_per_line/8 이면 NULL 반환 (GC 효율 낮으면 패스)
 *   true:  무조건 하나 선택 (foreground GC에서 공간 급할 때)
 *
 * ================================================================ */
static struct line *select_victim_greedy(struct conv_ftl *ftl, bool force)
{
	struct line_mgmt *lm = &ftl->lm;
	struct line *v = pqueue_peek(lm->victim_line_pq); /* 루트(최소 vpc) 확인 */

	if (!v) return NULL; /* victim 후보가 하나도 없음 */

	if (!force && v->vpc > (ftl->ssd->sp.pgs_per_line / 8))
		return NULL; /* vpc가 전체의 12.5% 이상이면 비효율적이라 패스 */

	pqueue_pop(lm->victim_line_pq); /* 루트를 큐에서 제거 */
	v->pos = 0;                      /* "큐에 없음" 표시 */
	lm->victim_line_cnt--;
	return v;
}

/* ================================================================
 * Victim 선택 ② Cost-Benefit (CB)
 *
 * [논문 원문 - Kawaguchi et al., 1995]
 *   "cost-benefit policy chooses to clean segments that maximize
 *    the formula: age × (1-u) / 2u"
 *
 * [이 구현에서의 공식]
 *   Score = get_age_weight(age) × IPC / (VPC + 1)
 *
 *   - age: 라인의 last_modified_time으로부터의 경과 시간
 *   - get_age_weight(age): 논문 Fig.7 계단 함수로 정규화
 *   - IPC = ipc (무효 페이지 수) ≈ (1-u) × pgs_per_line
 *   - VPC + 1 ≈ u × pgs_per_line (+1은 0-division 방지)
 *
 * [왜 Linear Scan인가?]
 *   age는 시간이 흐르면서 모든 라인에서 동시에 변함.
 *   PQ에 넣을 때의 age와 꺼낼 때의 age가 달라져서
 *   Heap 순서가 무의미해짐 → 매번 전체를 스캔해야 정확한 score 비교 가능.
 *
 * [복잡도] O(N), N = victim 후보 수. 보통 수십~수백.
 *
 * [장점]
 *   - age 반영: hot 데이터(최근 수정) 보호, cold(오래된) 우선 청소
 *   - locality 높은 워크로드에서 Greedy 대비 erase 28~55% 감소 (논문 결과)
 *
 * [단점]
 *   - erase count 무시 → wear 불균형
 *   - uniform random에서는 Greedy보다 약간 비효율 (스캔 오버헤드)
 *
 * ================================================================ */
static struct line *select_victim_cb(struct conv_ftl *ftl, bool force)
{
	struct line_mgmt *lm = &ftl->lm;
	pqueue_t *q = lm->victim_line_pq;
	struct line *best = NULL;
	uint64_t max_s = 0, now = ktime_get_ns();
	size_t i;

	if (q->size == 0) return NULL;

	/*
	 * pqueue 내부 배열 q->d[]를 1번부터 순회.
	 * (0번은 pqueue 라이브러리가 더미로 사용)
	 */
	for (i = 1; i <= q->size; i++) {
		struct line *c = (struct line *)q->d[i];
		uint64_t age, s;

		if (!c) continue; /* 안전 체크 */

		/* age = 현재 시각 - 마지막 무효화 시각 */
		age = (now > c->last_modified_time) ?
		      (now - c->last_modified_time) : 0;

		/* Score = AgeWeight × IPC / (VPC + 1) */
		s = get_age_weight(age) * (uint64_t)c->ipc /
		    ((uint64_t)(c->vpc + 1));

		if (s > max_s) {
			max_s = s;
			best = c;
		}
	}

	if (best) {
		pqueue_remove(q, best); /* 큐 중간에서 안전하게 제거 + 재정렬 */
		best->pos = 0;
		lm->victim_line_cnt--;
	}
	return best;
}

/* ================================================================
 * Victim 선택 ③ CAT (Cost-Age-Times)
 *
 * [논문 원문 - Chiang & Chang, 1999]
 *   "The cleaner chooses to clean segments that minimize:
 *    Cleaning_Cost × (1/Age) × Number_of_Cleaning"
 *
 *   이를 뒤집으면 maximize:
 *    Score = Age × IPC / (VPC × EraseCount)
 *
 * [이 구현에서의 공식]
 *   Score = get_age_weight(age) × IPC / ((VPC+1) × (EraseCount+1))
 *
 *   CB 공식에 (EraseCount+1)이 분모에 추가된 것이 유일한 차이.
 *
 * [EraseCount의 효과]
 *   많이 지워진 블록 → EraseCount 큼 → 분모 큼 → Score 낮음 → GC 후순위
 *   적게 지워진 블록 → EraseCount 작음 → 분모 작음 → Score 높음 → GC 우선
 *   → 결과적으로 모든 블록의 erase count가 균등해짐 (wear-leveling)
 *
 * [장점]
 *   - CB의 장점(age 반영) + wear-leveling
 *   - 논문 실험: Greedy 대비 erase 55% 감소, CB 대비 29% 감소
 *
 * [단점]
 *   - 워크로드가 변해도 공식이 고정됨 (α=1, δ=1 고정)
 *   → 이 한계를 RL 모드가 극복함
 *
 * ================================================================ */
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

		age = (now > c->last_modified_time) ?
		      (now - c->last_modified_time) : 0;

		/* CB 공식의 분모에 (erase_cnt + 1)이 추가됨 */
		s = get_age_weight(age) * (uint64_t)c->ipc /
		    ((uint64_t)(c->vpc + 1) * ((uint64_t)c->erase_cnt + 1));

		if (s > max_s) {
			max_s = s;
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

/* ================================================================
 * Victim 선택 ④ Unified (RL 모드 전용)
 *
 * [통합 공식]
 *   Score = apply_alpha(AgeWeight, α_level) × IPC
 *           / ((VPC+1) × apply_delta(EraseCount+1, δ_level))
 *
 * [apply_alpha - Age 가중치 지수 적용]
 *   α_level=0 → aw^0.5 = sqrt(aw)    : age 영향 약화
 *   α_level=1 → aw^1.0 = aw           : CB/CAT와 동일
 *   α_level=2 → aw^2.0 = aw²          : age 영향 강화 (cold 극우선)
 *
 * [apply_delta - EraseCount 가중치 지수 적용]
 *   δ_level=0 → 1                      : erase 무시 (= CB)
 *   δ_level=1 → sqrt(ec+1)             : 약한 wear-leveling
 *   δ_level=2 → ec+1                   : CAT와 동일
 *   δ_level=3 → (ec+1) × sqrt(ec+1)   : 강한 wear-leveling
 *
 * [이 함수의 호출 시점]
 *   do_gc()에서 gc_mode == GC_MODE_RL일 때만 호출.
 *   호출 전에 rl_decode_action()이 rl->alpha_level, delta_level을 설정해둠.
 *
 * [왜 pow()가 아닌 switch로 구현하는가]
 *   커널에서 float을 못 쓰므로 math.h의 pow() 사용 불가.
 *   isqrt_u64()로 정수 제곱근을 구하고 switch로 지수를 합성.
 *
 * ================================================================ */

/*
 * apply_alpha() - AgeWeight에 α 지수 적용
 *
 * 입력: aw = get_age_weight(age)의 반환값 (1, 5, 20, 100 중 하나)
 * 출력: aw^α에 해당하는 정수
 *
 * 예시:
 *   aw=100, level=0 → sqrt(100) = 10
 *   aw=100, level=1 → 100
 *   aw=100, level=2 → 10000
 */
static uint64_t apply_alpha(uint64_t aw, uint32_t level)
{
	switch (level) {
	case 0:  return isqrt_u64(aw);   /* aw^0.5 */
	case 1:  return aw;              /* aw^1.0 */
	default: return aw * aw;         /* aw^2.0 */
	}
}

/*
 * apply_delta() - (EraseCount+1)에 δ 지수 적용
 *
 * 입력: ec1 = erase_cnt + 1
 * 출력: ec1^δ에 해당하는 정수 (분모에 들어감)
 *
 * 예시 (erase_cnt=99 → ec1=100):
 *   level=0 → 1         (erase 무시)
 *   level=1 → sqrt(100) = 10
 *   level=2 → 100
 *   level=3 → 100 × 10 = 1000
 */
static uint64_t apply_delta(uint64_t ec1, uint32_t level)
{
	switch (level) {
	case 0:  return 1;                    /* ec^0 = 무시 */
	case 1:  return isqrt_u64(ec1);       /* ec^0.5 */
	case 2:  return ec1;                  /* ec^1.0 */
	default: return ec1 * isqrt_u64(ec1); /* ec^1.5 */
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

		age = (now > c->last_modified_time) ?
		      (now - c->last_modified_time) : 0;
		aw = get_age_weight(age);

		/*
		 * 통합 공식:
		 *   numerator   = aw^α × IPC
		 *   denominator = (VPC+1) × (EC+1)^δ
		 *   score       = numerator / denominator
		 *
		 * α, δ는 rl->alpha_level, rl->delta_level에서 결정됨.
		 * 이 값들은 do_gc() 시작 시 rl_decode_action()이 설정.
		 */
		numerator = apply_alpha(aw, rl->alpha_level) * (uint64_t)c->ipc;
		denominator = (uint64_t)(c->vpc + 1) *
			      apply_delta((uint64_t)c->erase_cnt + 1, rl->delta_level);
		if (denominator == 0) denominator = 1; /* 안전장치 */

		s = numerator / denominator;
		if (s > max_s) {
			max_s = s;
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

/* ================================================================
 *
 *           ★★★ RL (Reinforcement Learning) 인프라 ★★★
 *
 * [학습 루프] (GC 1회 = 에피소드 1회)
 *
 *   do_gc() 진입
 *     │
 *     ├─ ① rl_get_state(): 현재 SSD 상태를 54가지 중 하나로 이산화
 *     │
 *     ├─ ② rl_select_action(): epsilon-greedy로 36가지 중 action 선택
 *     │     ├─ 확률 ε: 랜덤 (탐험)
 *     │     └─ 확률 1-ε: argmax Q(state, a) (활용)
 *     │
 *     ├─ ③ rl_decode_action(): action → (alpha, delta, hot_thresh) 분해
 *     │
 *     ├─ ④ select_victim_unified(): 분해된 파라미터로 victim scoring
 *     │
 *     ├─ ⑤ GC 수행 (valid page 복사, erase)
 *     │     └─ gc_write_page() 내 rl_is_hot_page(): hot_thresh_level 사용
 *     │
 *     ├─ ⑥ rl_get_state(): GC 후 새 state 관찰
 *     │
 *     └─ ⑦ rl_update(): reward 계산 + Q(s,a) 갱신 + ε decay
 *
 * ================================================================ */

/* ================================================================
 * rl_get_state() - 현재 SSD 상태를 이산화된 state 인덱스로 변환
 *
 * [반환값] 0 ~ 53 사이의 정수
 *
 * [각 차원의 의미와 이산화 기준]
 *   → conv_ftl.h의 주석 참고
 *
 * [호출 시점]
 *   do_gc()에서 GC 직전(state 관찰)과 GC 직후(next state) 2회 호출.
 *
 * ================================================================ */
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

	/* S1: 공간 압박 정도 */
	free_pct = (lm->free_line_cnt * 100) / lm->tt_lines;
	if (free_pct > 30)      s1 = 0; /* 여유 */
	else if (free_pct > 10) s1 = 1; /* 보통 */
	else                    s1 = 2; /* 긴급 */

	/* S2: victim 큐의 평균 유효 페이지 비율 */
	for (i = 1; i <= q->size; i++) {
		struct line *c = (struct line *)q->d[i];
		if (c) vpc_sum += c->vpc;
	}
	vpc_avg_pct = (q->size > 0) ?
		(uint32_t)((vpc_sum * 100) / (q->size * spp->pgs_per_line)) : 50;
	if (vpc_avg_pct < 25)      s2 = 0; /* 더러움 */
	else if (vpc_avg_pct < 60) s2 = 1; /* 보통 */
	else                       s2 = 2; /* 깨끗 */

	/* S3: 워크로드의 hot/cold 편향 (avg_hot_degree 기반) */
	if (ftl->avg_hot_degree < 8)       s3 = 0; /* cold 위주 */
	else if (ftl->avg_hot_degree < 48) s3 = 1; /* 혼합 */
	else                               s3 = 2; /* hot 위주 */

	/* S4: wear 균등성 */
	for (i = 0; i < lm->tt_lines; i++) {
		uint32_t ec = lm->lines[i].erase_cnt;
		if (ec > erase_max) erase_max = ec;
		if (ec < erase_min) erase_min = ec;
	}
	s4 = ((erase_max - erase_min) > 20) ? 1 : 0;

	/* 4차원 좌표 → 1차원 인덱스 변환 */
	return s1 * (RL_NUM_S2 * RL_NUM_S3 * RL_NUM_S4) +
	       s2 * (RL_NUM_S3 * RL_NUM_S4) +
	       s3 * RL_NUM_S4 + s4;
}

/* ================================================================
 * rl_decode_action() - action 인덱스를 3개 파라미터 레벨로 분해
 *
 * action = a1×(4×3) + a2×3 + a3  (0~35)
 *
 * [결과]
 *   rl->alpha_level:      select_victim_unified()가 읽음
 *   rl->delta_level:      select_victim_unified()가 읽음
 *   rl->hot_thresh_level: rl_is_hot_page()가 읽음
 *
 * ================================================================ */
static void rl_decode_action(struct rl_config *rl, uint32_t action)
{
	rl->cur_action = action;
	rl->alpha_level      = action / (RL_NUM_A2 * RL_NUM_A3);
	rl->delta_level      = (action / RL_NUM_A3) % RL_NUM_A2;
	rl->hot_thresh_level = action % RL_NUM_A3;
}

/* ================================================================
 * rl_select_action() - Epsilon-Greedy 정책으로 action 선택
 *
 * [알고리즘]
 *   확률 ε: 랜덤 action (exploration, 새로운 조합 탐색)
 *   확률 1-ε: argmax Q(state, a) (exploitation, 최선 활용)
 *
 * [ε 변화]
 *   초기 0.3(30%) → 매 에피소드마다 ×0.999 → 최소 0.05(5%)
 *   학습 초기: 다양한 경험 축적
 *   학습 후기: 학습된 정책 활용, 최소한의 탐험으로 환경 변화 대응
 *
 * ================================================================ */
static uint32_t rl_select_action(struct rl_config *rl, uint32_t state)
{
	uint32_t rand_val, a, best_a = 0;
	int64_t best_q;

	rand_val = get_random_u32() % 1000;

	/* Explore: 랜덤 action */
	if (rand_val < rl->epsilon)
		return get_random_u32() % RL_NUM_ACTIONS;

	/* Exploit: Q값이 가장 높은 action 선택 */
	best_q = rl->q_table[state][0];
	for (a = 1; a < RL_NUM_ACTIONS; a++) {
		if (rl->q_table[state][a] > best_q) {
			best_q = rl->q_table[state][a];
			best_a = a;
		}
	}
	return best_a;
}

/* ================================================================
 * rl_update() - Reward 계산 + Q-table 갱신
 *
 * [Q-learning 공식]
 *   Q(s,a) ← Q(s,a) + α × [R + γ × max_a' Q(s',a') - Q(s,a)]
 *
 * [Reward 설계]
 *   R = -copy_ratio × 1000     (복사 비용: 낮을수록 좋음)
 *     - wear_delta × 10        (wear 불균형 악화: 0이 이상적)
 *     + reclaimed × 200 / ppl  (공간 회수: 많을수록 좋음)
 *
 *   copy_ratio = copied / (copied + reclaimed)
 *     → 0이면 복사 없이 전부 회수 (최고), 1이면 전부 복사 (최악)
 *
 *   wear_delta = (현재 max-min) - (이전 max-min)
 *     → 양수면 불균형 악화, 0이면 유지, 음수면 개선
 *
 * [호출 시점]
 *   do_gc()의 GC 완료 직후, gc_mode == GC_MODE_RL일 때만.
 *
 * ================================================================ */
static void rl_update(struct conv_ftl *ftl, uint32_t new_state)
{
	struct rl_config *rl = &ftl->rl;
	struct line_mgmt *lm = &ftl->lm;
	int64_t reward, max_q_next, old_q;
	uint64_t copied, reclaimed;
	uint32_t erase_max = 0, erase_min = UINT_MAX;
	uint32_t wear_delta;
	uint32_t i;

	/* 이번 GC에서 복사/회수한 양 */
	copied = ftl->gc_copied_pages - rl->prev_copied_pages;
	reclaimed = ftl->wfc.credits_to_refill; /* do_gc에서 victim->ipc로 설정됨 */

	/* 현재 wear 상태 */
	for (i = 0; i < lm->tt_lines; i++) {
		uint32_t ec = lm->lines[i].erase_cnt;
		if (ec > erase_max) erase_max = ec;
		if (ec < erase_min) erase_min = ec;
	}
	wear_delta = (erase_max - erase_min) -
		     (rl->prev_erase_max - rl->prev_erase_min);

	/* Reward 계산 */
	if (copied + reclaimed > 0)
		reward = -((int64_t)copied * 1000) / (int64_t)(copied + reclaimed);
	else
		reward = 0;
	reward -= (int64_t)wear_delta * 10;
	if (ftl->ssd->sp.pgs_per_line > 0)
		reward += ((int64_t)reclaimed * 200) / ftl->ssd->sp.pgs_per_line;

	/* max Q(s', a') - 다음 state에서의 최대 Q값 */
	max_q_next = rl->q_table[new_state][0];
	for (i = 1; i < RL_NUM_ACTIONS; i++) {
		if (rl->q_table[new_state][i] > max_q_next)
			max_q_next = rl->q_table[new_state][i];
	}

	/* Q(s,a) 갱신 (모든 값 ×1000 고정소수점) */
	old_q = rl->q_table[rl->cur_state][rl->cur_action];
	rl->q_table[rl->cur_state][rl->cur_action] = old_q +
		RL_ALPHA * (reward + RL_GAMMA * max_q_next / RL_Q_SCALE - old_q)
		/ RL_Q_SCALE;

	/* Epsilon decay */
	rl->epsilon = (rl->epsilon * RL_EPSILON_DECAY) / 1000;
	if (rl->epsilon < RL_EPSILON_MIN)
		rl->epsilon = RL_EPSILON_MIN;

	/* 통계 누적 */
	rl->total_episodes++;
	rl->total_reward += reward;

	/* 다음 에피소드를 위한 스냅샷 갱신 */
	rl->prev_copied_pages = ftl->gc_copied_pages;
	rl->prev_erase_max = erase_max;
	rl->prev_erase_min = erase_min;
}

/* ================================================================
 * rl_is_hot_page() - Hot/Cold 판별 (RL 모드에서 기준 조절 가능)
 *
 * [비-RL 모드]
 *   is_hot_page()로 위임 → avg_hot_degree × 1.0 기준 고정
 *
 * [RL 모드]
 *   hot_thresh_level에 따라 기준이 바뀜:
 *     level=0: avg × 0.5 (관대) → 대부분 hot 취급 → 분리 효과 약
 *     level=1: avg × 1.0 (기본) → 논문 기준 그대로
 *     level=2: avg × 1.5 (엄격) → 정말 hot인 것만 hot → 분리 강화
 *
 * [사용처]
 *   gc_write_page()에서 valid page를 gc_wp_hot과 gc_wp_cold 중
 *   어디에 쓸지 결정할 때 호출.
 *
 * ================================================================ */
static bool rl_is_hot_page(struct conv_ftl *ftl, uint64_t lpn, uint64_t now)
{
	uint64_t deg = calc_hot_degree(&ftl->page_meta[lpn], now);
	uint64_t threshold;

	if (gc_mode != GC_MODE_RL)
		return is_hot_page(ftl, lpn, now);

	switch (ftl->rl.hot_thresh_level) {
	case 0:  threshold = ftl->avg_hot_degree / 2; break;     /* avg×0.5 */
	case 2:  threshold = ftl->avg_hot_degree * 3 / 2; break; /* avg×1.5 */
	default: threshold = ftl->avg_hot_degree; break;          /* avg×1.0 */
	}

	return (deg * 16) >= threshold;
}

/* ================================================================
 *
 *           ★★★ 초기화 / 해제 함수 ★★★
 *
 * ================================================================ */

/*
 * init_lines() - 전체 라인 배열 및 PQ 초기화
 *
 * 모든 라인을 free_line_list에 넣고, victim/full 카운트를 0으로.
 * PQ는 Greedy용 Min-Heap 콜백으로 초기화하되,
 * CB/CAT/RL도 이 PQ 구조를 통해 원소를 관리함 (정렬은 무시하고 Linear Scan).
 */
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

static void remove_lines(struct conv_ftl *ftl)
{
	pqueue_free(ftl->lm.victim_line_pq);
	vfree(ftl->lm.lines);
}

/*
 * init_page_meta() - per-LPN 메타데이터 배열 할당 및 초기화
 *
 * 모든 LPN의 update_cnt=0, last_write_time=0으로 시작.
 * avg_hot_degree = 16 (×16 고정소수점에서 1.0에 해당).
 */
static void init_page_meta(struct conv_ftl *ftl)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	int i;

	ftl->page_meta = vmalloc(sizeof(struct page_meta) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++) {
		ftl->page_meta[i].update_cnt = 0;
		ftl->page_meta[i].last_write_time = 0;
	}
	ftl->avg_hot_degree = 16;
}

static void remove_page_meta(struct conv_ftl *ftl)
{
	vfree(ftl->page_meta);
}

/*
 * init_rl() - RL 에이전트 상태 초기화
 *
 * Q-table 전체를 0으로 → 학습 전에는 모든 action이 동등.
 * 기본 파라미터 = CAT와 동일한 설정(alpha=1, delta=2)으로 시작.
 */
static void init_rl(struct conv_ftl *ftl)
{
	struct rl_config *rl = &ftl->rl;

	memset(rl->q_table, 0, sizeof(rl->q_table));
	rl->cur_state = 0;
	rl->cur_action = 0;
	rl->alpha_level = 1;
	rl->delta_level = 2;
	rl->hot_thresh_level = 1;
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

/* ================================================================
 * Write Pointer 관리
 * ================================================================ */
static inline void check_addr(int a, int max)
{
	NVMEV_ASSERT(a >= 0 && a < max);
}

/*
 * get_next_free_line() - free 리스트에서 라인 하나 꺼내기
 *
 * FIFO 순서로 꺼냄 (list_first_entry).
 * 꺼낸 라인은 write pointer에 할당되어 "open block"이 됨.
 */
static struct line *get_next_free_line(struct conv_ftl *ftl)
{
	struct line_mgmt *lm = &ftl->lm;
	struct line *cur = list_first_entry_or_null(&lm->free_line_list,
						   struct line, entry);
	if (!cur) {
		NVMEV_ERROR("No free line!\n");
		return NULL;
	}
	list_del_init(&cur->entry);
	lm->free_line_cnt--;
	return cur;
}

/*
 * __get_wp() - io_type에 따라 적절한 write pointer를 반환
 *
 * USER_IO    → wp          (호스트 쓰기)
 * GC_IO_HOT  → gc_wp_hot   (GC hot page 복사)
 * GC_IO_COLD → gc_wp_cold  (GC cold page 복사)
 */
static struct write_pointer *__get_wp(struct conv_ftl *ftl, uint32_t io_type)
{
	switch (io_type) {
	case USER_IO:    return &ftl->wp;
	case GC_IO_HOT:  return &ftl->gc_wp_hot;
	case GC_IO_COLD: return &ftl->gc_wp_cold;
	default: NVMEV_ASSERT(0); return NULL;
	}
}

/*
 * prepare_write_pointer() - write pointer 초기화 + 첫 번째 open block 할당
 *
 * conv_init_ftl()에서 USER_IO, GC_IO_HOT, GC_IO_COLD 각각에 대해 호출.
 */
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

/*
 * advance_write_pointer() - 쓰기 포인터를 다음 위치로 이동
 *
 * [인터리빙 순서]
 *   pg → ch(채널) → lun → next_wordline → ... → 블록 끝 → 새 라인
 *
 * [블록 끝 처리]
 *   블록의 마지막 페이지를 넘어가면:
 *   - 현재 라인의 vpc == pgs_per_line → full_line_list로 이동
 *   - vpc < pgs_per_line → victim_line_pq로 이동 (GC 후보)
 *   - free 리스트에서 새 라인을 꺼내 open block으로 설정
 */
static void advance_write_pointer(struct conv_ftl *ftl, uint32_t io_type)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct line_mgmt *lm = &ftl->lm;
	struct write_pointer *wpp = __get_wp(ftl, io_type);

	check_addr(wpp->pg, spp->pgs_per_blk);
	wpp->pg++;

	/* 워드라인 내 다음 페이지면 여기서 종료 */
	if ((wpp->pg % spp->pgs_per_oneshotpg) != 0)
		return;

	/* 워드라인 끝 → 다음 채널로 */
	wpp->pg -= spp->pgs_per_oneshotpg;
	check_addr(wpp->ch, spp->nchs);
	wpp->ch++;
	if (wpp->ch != spp->nchs) return;

	/* 마지막 채널 → 다음 LUN으로 */
	wpp->ch = 0;
	check_addr(wpp->lun, spp->luns_per_ch);
	wpp->lun++;
	if (wpp->lun != spp->luns_per_ch) return;

	/* 마지막 LUN → 다음 워드라인으로 */
	wpp->lun = 0;
	wpp->pg += spp->pgs_per_oneshotpg;
	if (wpp->pg != spp->pgs_per_blk) return;

	/* ★ 블록 끝에 도달: 라인을 full/victim으로 전이 */
	wpp->pg = 0;

	if (wpp->curline->vpc == spp->pgs_per_line) {
		/* 모든 페이지가 유효 → full 라인 */
		NVMEV_ASSERT(wpp->curline->ipc == 0);
		list_add_tail(&wpp->curline->entry, &lm->full_line_list);
		lm->full_line_cnt++;
	} else {
		/* 무효 페이지 존재 → GC 후보 (victim) */
		NVMEV_ASSERT(wpp->curline->ipc > 0);
		pqueue_insert(lm->victim_line_pq, wpp->curline);
		lm->victim_line_cnt++;
	}

	/* 새 free 라인 할당 */
	check_addr(wpp->blk, spp->blks_per_pl);
	wpp->curline = get_next_free_line(ftl);
	wpp->blk = wpp->curline->id;
	check_addr(wpp->blk, spp->blks_per_pl);

	NVMEV_ASSERT(wpp->pg == 0 && wpp->lun == 0 &&
		     wpp->ch == 0 && wpp->pl == 0);
}

/*
 * get_new_page() - 현재 write pointer 위치에서 새 PPA 생성
 *
 * 이 PPA에 데이터가 기록될 예정.
 * advance_write_pointer()는 이 함수 호출 후에 별도로 호출해야 함.
 */
static struct ppa get_new_page(struct conv_ftl *ftl, uint32_t io_type)
{
	struct write_pointer *wp = __get_wp(ftl, io_type);
	struct ppa ppa = { .ppa = 0 };

	ppa.g.ch = wp->ch;
	ppa.g.lun = wp->lun;
	ppa.g.pg = wp->pg;
	ppa.g.blk = wp->blk;
	ppa.g.pl = wp->pl;

	NVMEV_ASSERT(ppa.g.pl == 0); /* 단일 플레인 가정 */
	return ppa;
}

/* ================================================================
 * 매핑/역매핑 테이블 초기화
 * ================================================================ */
static void init_maptbl(struct conv_ftl *ftl)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	int i;

	ftl->maptbl = vmalloc(sizeof(struct ppa) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++)
		ftl->maptbl[i].ppa = UNMAPPED_PPA;
}
static void remove_maptbl(struct conv_ftl *ftl) { vfree(ftl->maptbl); }

static void init_rmap(struct conv_ftl *ftl)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	int i;

	ftl->rmap = vmalloc(sizeof(uint64_t) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++)
		ftl->rmap[i] = INVALID_LPN;
}
static void remove_rmap(struct conv_ftl *ftl) { vfree(ftl->rmap); }

/* ================================================================
 * FTL 인스턴스 초기화/제거
 *
 * conv_init_ftl(): 모든 자료구조 할당 + write pointer 3개 준비
 * conv_remove_ftl(): 모든 vmalloc 해제
 *
 * [호출 순서]
 *   conv_init_namespace() → conv_init_ftl() (파티션별 1회)
 *   conv_remove_namespace() → conv_remove_ftl() (파티션별 1회)
 *
 * ================================================================ */
static void conv_init_ftl(struct conv_ftl *ftl, struct convparams *cpp,
			  struct ssd *ssd)
{
	ftl->cp = *cpp;
	ftl->ssd = ssd;
	ftl->gc_count = 0;
	ftl->gc_copied_pages = 0;

	init_maptbl(ftl);       /* LPN→PPA 매핑 테이블 */
	init_rmap(ftl);         /* PPA→LPN 역매핑 테이블 */
	init_page_meta(ftl);    /* per-LPN hot/cold 메타데이터 */
	init_lines(ftl);        /* 라인 배열 + PQ */
	init_rl(ftl);           /* RL Q-table 초기화 */

	/*
	 * 쓰기 포인터 3개 초기화.
	 * 각각 free 라인 1개를 소비하므로 총 3개의 open block이 생김.
	 * → gc_thres_lines를 3으로 설정한 이유.
	 */
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
	cpp->gc_thres_lines = 3;       /* open block 3개 필요 */
	cpp->gc_thres_lines_high = 3;
	cpp->enable_gc_delay = 1;      /* NAND 타이밍 시뮬레이션 활성화 */
	cpp->pba_pcent = (int)((1 + cpp->op_area_pcent) * 100);
}

/* ================================================================
 * 네임스페이스 초기화/제거
 *
 * NVMeVirt가 가상 NVMe 디바이스를 생성할 때 호출.
 * SSD_PARTITIONS(기본 1)개의 FTL 인스턴스를 생성하고,
 * PCIe와 write_buffer는 모든 인스턴스가 공유.
 *
 * ================================================================ */
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

	/* PCIe, Write buffer 공유 (0번 인스턴스의 것을 재사용) */
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
	ns->proc_io_cmd = conv_proc_nvme_io_cmd; /* IO 핸들러 등록 */

	NVMEV_INFO("FTL physical=%lld logical=%lld (ratio=%d) gc_mode=%d\n",
		   size, ns->size, cpp.pba_pcent, gc_mode);
}

void conv_remove_namespace(struct nvmev_ns *ns)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	const uint32_t nr_parts = SSD_PARTITIONS;
	uint32_t i;

	/* 공유 자원은 중복 해제 방지 */
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

/* ================================================================
 * PPA/LPN 유효성 검사
 * ================================================================ */
static inline bool valid_ppa(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;

	if (ppa->g.ch < 0 || ppa->g.ch >= spp->nchs)       return false;
	if (ppa->g.lun < 0 || ppa->g.lun >= spp->luns_per_ch) return false;
	if (ppa->g.pl < 0 || ppa->g.pl >= spp->pls_per_lun)   return false;
	if (ppa->g.blk < 0 || ppa->g.blk >= spp->blks_per_pl) return false;
	if (ppa->g.pg < 0 || ppa->g.pg >= spp->pgs_per_blk)   return false;
	return true;
}

static inline bool valid_lpn(struct conv_ftl *ftl, uint64_t lpn)
{
	return lpn < ftl->ssd->sp.tt_pgs;
}

static inline bool mapped_ppa(struct ppa *ppa)
{
	return ppa->ppa != UNMAPPED_PPA;
}

/* PPA로부터 해당 라인(블록) 구조체를 찾는 함수 */
static inline struct line *get_line(struct conv_ftl *ftl, struct ppa *ppa)
{
	return &ftl->lm.lines[ppa->g.blk];
}

/* ================================================================
 *
 *           ★★★ 페이지 상태 관리 ★★★
 *
 * NAND 페이지는 3가지 상태: PG_FREE → PG_VALID → PG_INVALID
 * (PG_FREE → PG_VALID: 쓰기, PG_VALID → PG_INVALID: 덮어쓰기)
 *
 * ================================================================ */

/*
 * mark_page_invalid() - 페이지를 무효(Invalid) 상태로 변경
 *
 * [호출 시점]
 *   conv_write()에서 이미 매핑된 LPN에 새 데이터를 쓸 때 (update write).
 *   기존 물리 페이지는 무효화되고, 새 물리 페이지에 데이터가 기록됨.
 *
 * [수행 내용]
 *   1) NAND page 상태: PG_VALID → PG_INVALID
 *   2) NAND block의 ipc++, vpc--
 *   3) line의 ipc++, vpc-- (또는 PQ 우선순위 갱신)
 *   4) full 라인이었으면 → victim 큐로 이동
 *   5) last_modified_time 갱신 → CB/CAT의 age 계산에 반영됨
 */
static void mark_page_invalid(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct line_mgmt *lm = &ftl->lm;
	struct nand_block *blk;
	struct nand_page *pg;
	struct line *line;
	bool was_full = false;

	/* NAND 페이지 상태 변경 */
	pg = get_pg(ftl->ssd, ppa);
	NVMEV_ASSERT(pg->status == PG_VALID);
	pg->status = PG_INVALID;

	/* NAND 블록 카운터 갱신 */
	blk = get_blk(ftl->ssd, ppa);
	blk->ipc++;
	blk->vpc--;

	/* 라인 카운터 갱신 */
	line = get_line(ftl, ppa);
	if (line->vpc == spp->pgs_per_line)
		was_full = true; /* 이전에 full 라인이었음 */

	line->ipc++;

	/*
	 * vpc 갱신: 라인이 PQ에 있으면 pqueue_change_priority()로,
	 * 없으면 직접 vpc--. PQ의 우선순위가 vpc이므로 PQ를 통해 갱신해야
	 * Heap 속성이 유지됨.
	 */
	if (line->pos) /* pos != 0 → 현재 PQ에 있음 */
		pqueue_change_priority(lm->victim_line_pq, line->vpc - 1, line);
	else
		line->vpc--;

	/* full → victim 전이 */
	if (was_full) {
		list_del_init(&line->entry);  /* full 리스트에서 제거 */
		lm->full_line_cnt--;
		pqueue_insert(lm->victim_line_pq, line); /* victim 큐에 삽입 */
		lm->victim_line_cnt++;
	}

	/* CB/CAT age 계산의 기준점 갱신 */
	line->last_modified_time = ktime_get_ns();
}

/*
 * mark_page_valid() - 페이지를 유효(Valid) 상태로 변경
 *
 * [호출 시점]
 *   conv_write(): 새 페이지에 호스트 데이터 기록 시
 *   gc_write_page(): GC가 valid page를 새 위치에 복사 시
 */
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

/*
 * mark_block_free() - 블록의 모든 페이지를 FREE로 리셋
 *
 * [호출 시점]
 *   do_gc()에서 victim 블록의 모든 valid page를 복사한 후,
 *   해당 블록을 erase할 때 호출.
 *
 * [NAND 레벨 erase_cnt]
 *   blk->erase_cnt++: 이것은 ssd.h의 nand_block 레벨 카운터.
 *   line->erase_cnt++는 mark_line_free()에서 별도로 수행.
 */
static void mark_block_free(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct nand_block *blk = get_blk(ftl->ssd, ppa);
	int i;

	for (i = 0; i < spp->pgs_per_blk; i++)
		blk->pg[i].status = PG_FREE;

	blk->ipc = 0;
	blk->vpc = 0;
	blk->erase_cnt++;
}

/*
 * mark_line_free() - 라인을 free 리스트로 복귀
 *
 * [호출 시점]
 *   do_gc()의 마지막 단계. 모든 블록 erase 후 호출.
 *
 * [erase_cnt 증가]
 *   이것이 CAT/RL scoring에서 사용하는 FTL 레벨의 erase 카운터.
 *   다음 GC에서 이 라인이 victim 후보가 되면,
 *   erase_cnt가 높을수록 scoring 분모가 커져 후순위로 밀림.
 */
static void mark_line_free(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct line *line = get_line(ftl, ppa);

	line->ipc = 0;
	line->vpc = 0;
	line->erase_cnt++; /* ★ CAT/RL wear-leveling의 핵심 */

	list_add_tail(&line->entry, &ftl->lm.free_line_list);
	ftl->lm.free_line_cnt++;
}

/* ================================================================
 *
 *           ★★★ GC 실행 경로 ★★★
 *
 * ================================================================ */

/*
 * gc_read_page() - GC가 victim 블록에서 valid page를 읽는 시뮬레이션
 */
static void gc_read_page(struct conv_ftl *ftl, struct ppa *ppa)
{
	if (ftl->cp.enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO,
			.cmd = NAND_READ,
			.stime = 0,
			.xfer_size = ftl->ssd->sp.pgsz,
			.interleave_pci_dma = false,
			.ppa = ppa,
		};
		ssd_advance_nand(ftl->ssd, &gcr);
	}
}

/*
 * gc_write_page() - valid page를 새 위치로 복사 (★ Hot/Cold 분기 포함)
 *
 * [핵심 흐름]
 *   1) rmap에서 물리 주소 → LPN 확인
 *   2) rl_is_hot_page()로 hot/cold 판별
 *   3) hot → gc_wp_hot 블록에, cold → gc_wp_cold 블록에 기록
 *   4) 매핑 테이블 갱신 (maptbl, rmap)
 *   5) 페이지 유효화 + write pointer 전진
 *   6) NAND 쓰기 시뮬레이션 (워드라인 끝이면)
 *
 * [이것이 논문 M6의 실제 구현부]
 *   gc_write_page가 hot/cold를 구분하여 다른 블록에 쓰기 때문에,
 *   GC 후 결과적으로:
 *   - gc_wp_hot 블록: hot 데이터만 → 빠르게 무효화 → 다음 GC 효율적
 *   - gc_wp_cold 블록: cold 데이터만 → 오래 유효 → GC 대상 안 됨
 */
static uint64_t gc_write_page(struct conv_ftl *ftl, struct ppa *old_ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	uint64_t lpn = get_rmap_ent(ftl, old_ppa); /* 이 물리 페이지의 LPN */
	uint64_t now = ktime_get_ns();
	uint32_t wp_type;
	struct ppa new_ppa;

	NVMEV_ASSERT(valid_lpn(ftl, lpn));

	/*
	 * ★ Hot/Cold 판별 → Write Pointer 분기
	 *
	 * rl_is_hot_page()는 gc_mode에 따라:
	 *   비-RL: is_hot_page() → avg_hot_degree × 1.0 기준
	 *   RL:    hot_thresh_level에 따라 × 0.5 / 1.0 / 1.5 기준
	 */
	wp_type = rl_is_hot_page(ftl, lpn, now) ? GC_IO_HOT : GC_IO_COLD;

	/* 해당 wp에서 새 PPA 할당 */
	new_ppa = get_new_page(ftl, wp_type);

	/* 매핑 테이블 갱신: LPN이 새 물리 위치를 가리키도록 */
	set_maptbl_ent(ftl, lpn, &new_ppa);
	set_rmap_ent(ftl, lpn, &new_ppa);

	mark_page_valid(ftl, &new_ppa);
	ftl->gc_copied_pages++; /* 통계 + RL reward 계산에 사용 */

	advance_write_pointer(ftl, wp_type);

	/* NAND 쓰기 시뮬레이션 */
	if (ftl->cp.enable_gc_delay) {
		struct nand_cmd gcw = {
			.type = GC_IO, /* NAND 타이밍은 항상 GC_IO */
			.cmd = NAND_NOP,
			.stime = 0,
			.interleave_pci_dma = false,
			.ppa = &new_ppa,
		};
		if (last_pg_in_wordline(ftl, &new_ppa)) {
			gcw.cmd = NAND_WRITE;
			gcw.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg;
		}
		ssd_advance_nand(ftl->ssd, &gcw);
	}
	return 0;
}

/*
 * clean_one_flashpg() - 하나의 flash page 단위로 valid page 복사
 *
 * flash page = 여러 NAND 서브페이지의 묶음 (pgs_per_flashpg개).
 * 먼저 유효 페이지 수를 세고, 일괄 읽기 후 하나씩 gc_write_page().
 */
static void clean_one_flashpg(struct conv_ftl *ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct nand_page *pg;
	int cnt = 0, i;
	struct ppa copy = *ppa;

	/* 유효 페이지 수 카운트 */
	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg = get_pg(ftl->ssd, &copy);
		NVMEV_ASSERT(pg->status != PG_FREE); /* victim에 FREE가 있으면 안 됨 */
		if (pg->status == PG_VALID)
			cnt++;
		copy.g.pg++;
	}

	copy = *ppa;
	if (cnt <= 0) return; /* 유효 페이지 없으면 스킵 */

	/* 유효 페이지 일괄 읽기 시뮬레이션 */
	if (ftl->cp.enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO,
			.cmd = NAND_READ,
			.stime = 0,
			.xfer_size = spp->pgsz * cnt,
			.interleave_pci_dma = false,
			.ppa = &copy,
		};
		ssd_advance_nand(ftl->ssd, &gcr);
	}

	/* 유효 페이지를 하나씩 새 위치에 복사 */
	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg = get_pg(ftl->ssd, &copy);
		if (pg->status == PG_VALID)
			gc_write_page(ftl, &copy); /* ★ 여기서 hot/cold 분기 */
		copy.g.pg++;
	}
}

/* ================================================================
 * do_gc() - GC 메인 함수
 *
 * [파라미터]
 *   ftl:       FTL 인스턴스
 *   force:     true면 효율성 체크 무시 (foreground GC)
 *   select_fn: victim 선택 함수 포인터 (RL 모드에서는 NULL, 내부에서 unified 사용)
 *
 * [전체 흐름]
 *   1) [RL 모드] state 관찰 → action 선택 → 파라미터 적용
 *   2) victim 선택 (select_fn 또는 select_victim_unified)
 *   3) victim 블록의 모든 flash page 순회:
 *      - 모든 채널/LUN에 대해 clean_one_flashpg() 호출
 *      - 마지막 flash page에서 mark_block_free() + NAND ERASE
 *   4) mark_line_free()로 라인을 free 리스트로 복귀
 *   5) [RL 모드] reward 계산 + Q-table 갱신
 *
 * ================================================================ */
static int do_gc(struct conv_ftl *ftl, bool force, victim_select_fn select_fn)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	struct line *victim = NULL;
	struct ppa ppa;
	int flashpg;
	uint32_t new_state;

	/* ────── RL 모드: GC 전 처리 ────── */
	if (gc_mode == GC_MODE_RL) {
		struct rl_config *rl = &ftl->rl;
		uint32_t state = rl_get_state(ftl);       /* ① state 관찰 */
		uint32_t action = rl_select_action(rl, state); /* ② action 선택 */

		rl->cur_state = state;
		rl_decode_action(rl, action);               /* ③ 파라미터 적용 */

		victim = select_victim_unified(ftl, force); /* ④ unified scoring */
	} else {
		/* 비-RL: 전달받은 함수 포인터로 victim 선택 */
		victim = select_fn(ftl, force);
	}

	if (!victim) return -1; /* victim 없음 = GC 불필요 또는 불가능 */

	ftl->gc_count++;
	ppa.g.blk = victim->id;

	/*
	 * credits_to_refill = victim의 ipc (무효 페이지 수)
	 * = 이 GC로 회수되는 공간.
	 * check_and_refill_write_credit()에서 write_credits에 더해짐.
	 */
	ftl->wfc.credits_to_refill = victim->ipc;

	/* ────── victim 블록의 모든 페이지 처리 ────── */
	for (flashpg = 0; flashpg < spp->flashpgs_per_blk; flashpg++) {
		int ch, lun;

		ppa.g.pg = flashpg * spp->pgs_per_flashpg;

		/* 모든 채널 × 모든 LUN 순회 (슈퍼블록 구조) */
		for (ch = 0; ch < spp->nchs; ch++) {
			for (lun = 0; lun < spp->luns_per_ch; lun++) {
				struct nand_lun *lunp;

				ppa.g.ch = ch;
				ppa.g.lun = lun;
				ppa.g.pl = 0;
				lunp = get_lun(ftl->ssd, &ppa);

				/* valid page 복사 (내부에서 hot/cold 분기) */
				clean_one_flashpg(ftl, &ppa);

				/* 마지막 flash page → 블록 erase */
				if (flashpg == (spp->flashpgs_per_blk - 1)) {
					mark_block_free(ftl, &ppa);

					if (ftl->cp.enable_gc_delay) {
						struct nand_cmd gce = {
							.type = GC_IO,
							.cmd = NAND_ERASE,
							.stime = 0,
							.interleave_pci_dma = false,
							.ppa = &ppa,
						};
						ssd_advance_nand(ftl->ssd, &gce);
					}
					lunp->gc_endtime = lunp->next_lun_avail_time;
				}
			}
		}
	}

	/* 라인을 free로 복귀 (erase_cnt 증가) */
	mark_line_free(ftl, &ppa);

	/* ────── RL 모드: GC 후 처리 ────── */
	if (gc_mode == GC_MODE_RL) {
		new_state = rl_get_state(ftl);  /* ⑥ 새 state 관찰 */
		rl_update(ftl, new_state);      /* ⑦ reward + Q 갱신 */
	}

	return 0;
}

/* ================================================================
 * foreground_gc() - 공간 부족 시 호출되는 최상위 GC 진입점
 *
 * [호출 경로]
 *   conv_write() → consume_write_credit() → check_and_refill_write_credit()
 *   → credits ≤ 0 → foreground_gc()
 *
 * [gc_mode에 따른 분기]
 *   0: Greedy  → do_gc(ftl, true, select_victim_greedy)
 *   1: CB      → do_gc(ftl, true, select_victim_cb)
 *   2: CAT     → do_gc(ftl, true, select_victim_cat)
 *   3: RL      → do_gc(ftl, true, NULL) → 내부에서 select_victim_unified
 *
 * ================================================================ */
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
		do_gc(ftl, true, NULL); /* RL은 내부에서 unified scoring 사용 */
		break;
	default:
		do_gc(ftl, true, select_victim_greedy);
		break;
	}
}

/* ================================================================
 *
 *           ★★★ NVMe IO 명령 처리 ★★★
 *
 * ================================================================ */

/*
 * is_same_flash_page() - 두 PPA가 같은 flash page 안에 있는지 확인
 *
 * 연속된 읽기를 하나로 합쳐 I/O 효율을 높이기 위해 사용.
 */
static bool is_same_flash_page(struct conv_ftl *ftl,
			       struct ppa p1, struct ppa p2)
{
	struct ssdparams *spp = &ftl->ssd->sp;
	return (p1.h.blk_in_ssd == p2.h.blk_in_ssd) &&
	       (p1.g.pg / spp->pgs_per_flashpg == p2.g.pg / spp->pgs_per_flashpg);
}

/* ================================================================
 * conv_read() - NVMe Read 명령 처리
 *
 * [흐름]
 *   1) LBA → LPN 변환
 *   2) LPN별로 매핑 테이블 조회 → PPA 획득
 *   3) 같은 flash page의 읽기는 하나로 합침 (최적화)
 *   4) ssd_advance_nand()로 NAND 읽기 타이밍 시뮬레이션
 *   5) 최종 완료 시각을 ret->nsecs_target에 설정
 *
 * ================================================================ */
static bool conv_read(struct nvmev_ns *ns, struct nvmev_request *req,
		      struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *ftl = &conv_ftls[0];
	struct ssdparams *spp = &ftl->ssd->sp;
	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba;
	uint64_t nr_lba = cmd->rw.length + 1;
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;
	uint64_t lpn, nsecs_start = req->nsecs_start;
	uint64_t nsecs_completed, nsecs_latest = nsecs_start;
	uint32_t xfer_size, i, nr_parts = ns->nr_parts;
	struct ppa prev_ppa;
	struct nand_cmd srd = {
		.type = USER_IO,
		.cmd = NAND_READ,
		.stime = nsecs_start,
		.interleave_pci_dma = true,
	};

	if ((end_lpn / nr_parts) >= spp->tt_pgs) return false;

	/* 펌웨어 처리 지연 추가 */
	srd.stime += (LBA_TO_BYTE(nr_lba) <= KB(4) * nr_parts) ?
		     spp->fw_4kb_rd_lat : spp->fw_rd_lat;

	/* 파티션별로 LPN을 분배하여 처리 */
	for (i = 0; (i < nr_parts) && (start_lpn <= end_lpn); i++, start_lpn++) {
		ftl = &conv_ftls[start_lpn % nr_parts];
		xfer_size = 0;
		prev_ppa = get_maptbl_ent(ftl, start_lpn / nr_parts);

		for (lpn = start_lpn; lpn <= end_lpn; lpn += nr_parts) {
			struct ppa cur = get_maptbl_ent(ftl, lpn / nr_parts);

			if (!mapped_ppa(&cur) || !valid_ppa(ftl, &cur))
				continue;

			/* 같은 flash page면 하나로 합침 */
			if (mapped_ppa(&prev_ppa) &&
			    is_same_flash_page(ftl, cur, prev_ppa)) {
				xfer_size += spp->pgsz;
				continue;
			}

			/* 이전까지 합쳐진 요청 발행 */
			if (xfer_size > 0) {
				srd.xfer_size = xfer_size;
				srd.ppa = &prev_ppa;
				nsecs_completed = ssd_advance_nand(ftl->ssd, &srd);
				nsecs_latest = max(nsecs_completed, nsecs_latest);
			}

			xfer_size = spp->pgsz;
			prev_ppa = cur;
		}

		/* 남은 요청 발행 */
		if (xfer_size > 0) {
			srd.xfer_size = xfer_size;
			srd.ppa = &prev_ppa;
			nsecs_completed = ssd_advance_nand(ftl->ssd, &srd);
			nsecs_latest = max(nsecs_completed, nsecs_latest);
		}
	}

	ret->nsecs_target = nsecs_latest;
	ret->status = NVME_SC_SUCCESS;
	return true;
}

/* ================================================================
 * conv_write() - NVMe Write 명령 처리
 *
 * [핵심 흐름]
 *   LPN 단위로 반복:
 *     1) 기존 매핑 있으면 → 구 페이지 무효화 (mark_page_invalid)
 *     2) 새 PPA 할당 (get_new_page, USER_IO)
 *     3) 매핑 테이블 갱신 (maptbl, rmap)
 *     4) 새 페이지 유효화 (mark_page_valid)
 *     5) ★ page_meta 갱신 (update_cnt++, last_write_time 설정)
 *     6) ★ avg_hot_degree EMA 갱신
 *     7) write pointer 전진
 *     8) 워드라인 끝이면 NAND WRITE 시뮬레이션
 *     9) 쓰기 크레딧 소모 → 부족하면 foreground_gc() 호출
 *
 * [page_meta 갱신이 중요한 이유]
 *   이 데이터가 나중에 GC에서 hot/cold 판별의 기초가 됨.
 *   update_cnt가 높고 last_write_time이 최근이면 → hot degree 높음 → hot
 *   이 LPN이 GC로 복사될 때 gc_wp_hot에 들어감.
 *
 * [avg_hot_degree EMA]
 *   매 쓰기마다 EMA(Exponential Moving Average)로 갱신:
 *     avg = (avg × 15 + new_degree × 16) / 16
 *   이것이 is_hot_page()에서 hot/cold 판별의 기준선이 됨.
 *
 * ================================================================ */
static bool conv_write(struct nvmev_ns *ns, struct nvmev_request *req,
		       struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *ftl = &conv_ftls[0];
	struct ssdparams *spp = &ftl->ssd->sp;
	struct buffer *wbuf = ftl->ssd->write_buffer;
	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba;
	uint64_t nr_lba = cmd->rw.length + 1;
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;
	uint64_t lpn, nsecs_latest, nsecs_xfer;
	uint32_t nr_parts = ns->nr_parts, alloc;
	struct nand_cmd swr = {
		.type = USER_IO,
		.cmd = NAND_WRITE,
		.interleave_pci_dma = false,
		.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg,
	};

	if ((end_lpn / nr_parts) >= spp->tt_pgs) return false;

	/* Write buffer에 데이터 적재 */
	alloc = buffer_allocate(wbuf, LBA_TO_BYTE(nr_lba));
	if (alloc < LBA_TO_BYTE(nr_lba)) return false;

	/* Write buffer로의 전송 시간 시뮬레이션 */
	nsecs_latest = ssd_advance_write_buffer(ftl->ssd, req->nsecs_start,
						LBA_TO_BYTE(nr_lba));
	nsecs_xfer = nsecs_latest; /* early completion 기준 시점 */
	swr.stime = nsecs_latest;

	/* LPN 단위 처리 루프 */
	for (lpn = start_lpn; lpn <= end_lpn; lpn++) {
		uint64_t local_lpn, nsecs_done = 0;
		struct ppa ppa;
		struct page_meta *pm;
		uint64_t degree;

		/* 파티셔닝: lpn % nr_parts로 FTL 인스턴스 선택 */
		ftl = &conv_ftls[lpn % nr_parts];
		local_lpn = lpn / nr_parts;

		/* ① 기존 매핑 확인 → update write면 구 페이지 무효화 */
		ppa = get_maptbl_ent(ftl, local_lpn);
		if (mapped_ppa(&ppa)) {
			mark_page_invalid(ftl, &ppa);           /* 구 페이지 → INVALID */
			set_rmap_ent(ftl, INVALID_LPN, &ppa);   /* 역매핑 해제 */
		}

		/* ② 새 PPA 할당 (유저 쓰기 = hot 취급 → wp 블록에 기록) */
		ppa = get_new_page(ftl, USER_IO);

		/* ③④ 매핑 갱신 + 유효화 */
		set_maptbl_ent(ftl, local_lpn, &ppa);
		set_rmap_ent(ftl, local_lpn, &ppa);
		mark_page_valid(ftl, &ppa);

		/* ⑤ page_meta 갱신: hot/cold 판별 데이터 축적 */
		pm = &ftl->page_meta[local_lpn];
		pm->update_cnt++;
		pm->last_write_time = ktime_get_ns();

		/* ⑥ avg_hot_degree EMA 갱신 (×16 고정소수점) */
		degree = calc_hot_degree(pm, pm->last_write_time);
		ftl->avg_hot_degree = (ftl->avg_hot_degree * 15 + degree * 16) / 16;

		/* ⑦ Write pointer 전진 */
		advance_write_pointer(ftl, USER_IO);

		/* ⑧ 워드라인 끝이면 NAND WRITE 발행 */
		if (last_pg_in_wordline(ftl, &ppa)) {
			swr.ppa = &ppa;
			nsecs_done = ssd_advance_nand(ftl->ssd, &swr);
			nsecs_latest = max(nsecs_done, nsecs_latest);
			schedule_internal_operation(req->sq_id, nsecs_done, wbuf,
						   spp->pgs_per_oneshotpg * spp->pgsz);
		}

		/* ⑨ 크레딧 소모 → 필요 시 GC */
		consume_write_credit(ftl);
		check_and_refill_write_credit(ftl);
	}

	/* 완료 시간 결정: FUA면 NAND까지 대기, 아니면 early completion */
	if ((cmd->rw.control & NVME_RW_FUA) || !spp->write_early_completion)
		ret->nsecs_target = nsecs_latest;
	else
		ret->nsecs_target = nsecs_xfer;

	ret->status = NVME_SC_SUCCESS;
	return true;
}

/* ================================================================
 * conv_flush() - NVMe Flush 명령 처리 + GC/RL 통계 출력
 *
 * SSD가 유휴 상태가 될 때까지 기다린 후 dmesg로 통계 출력.
 * fio 벤치마크 후 `dmesg | grep NVMeVirt`로 확인 가능.
 *
 * ================================================================ */
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

	/* 기본 GC 통계 */
	printk(KERN_INFO "NVMeVirt: [FLUSH] gc_mode=%d gc=%llu copied=%llu avg=%llu\n",
	       gc_mode, total_gc, total_cp,
	       total_gc > 0 ? total_cp / total_gc : 0);

	/* RL 모드 전용 통계 */
	if (gc_mode == GC_MODE_RL && ns->nr_parts > 0) {
		struct rl_config *rl = &conv_ftls[0].rl;
		printk(KERN_INFO "NVMeVirt: [RL] episodes=%llu epsilon=%u "
		       "avg_reward=%lld alpha=%u delta=%u hot_thresh=%u\n",
		       rl->total_episodes, rl->epsilon,
		       rl->total_episodes > 0 ?
			(int64_t)(rl->total_reward / (int64_t)rl->total_episodes) : 0,
		       rl->alpha_level, rl->delta_level, rl->hot_thresh_level);
	}

	/* Wear-leveling 통계 */
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

/* ================================================================
 * conv_proc_nvme_io_cmd() - NVMe IO 명령 디스패처
 *
 * NVMeVirt 코어가 NVMe 커맨드를 받으면 이 함수를 호출.
 * (conv_init_namespace에서 ns->proc_io_cmd에 등록됨)
 *
 * opcode에 따라 read / write / flush로 분기.
 *
 * ================================================================ */
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