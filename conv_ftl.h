// SPDX-License-Identifier: GPL-2.0-only
/*
 * ================================================================
 * conv_ftl.h - Conventional FTL 헤더 (NVMeVirt 가상 SSD용)
 * ================================================================
 *
 * [이 파일의 역할]
 *   NVMeVirt의 FTL(Flash Translation Layer) 계층에서 사용하는
 *   모든 자료구조와 상수를 정의한다.
 *   conv_ftl.c가 이 헤더를 include하여 실제 로직을 구현한다.
 *
 * [전체 아키텍처 개요]
 *
 *   호스트(fio)
 *     │  NVMe Write/Read 커맨드
 *     ▼
 *   NVMeVirt (커널 모듈)
 *     │  conv_proc_nvme_io_cmd()
 *     ▼
 *   conv_ftl.c  ◄── 이 헤더가 정의하는 구조체들을 사용
 *     │
 *     ├─ conv_write()  : 호스트 쓰기 처리
 *     │    ├─ page_meta 갱신 (update_cnt, last_write_time)
 *     │    ├─ 매핑 테이블 갱신 (maptbl, rmap)
 *     │    └─ write pointer 전진 (wp)
 *     │
 *     ├─ conv_read()   : 호스트 읽기 처리
 *     │    └─ 매핑 테이블 조회 → NAND 읽기 시뮬레이션
 *     │
 *     └─ foreground_gc() : 공간 부족 시 GC 수행
 *          ├─ victim 선택 (Greedy/CB/CAT/RL)
 *          ├─ valid page 복사 (gc_write_page)
 *          │    └─ hot/cold 판별 → gc_wp_hot 또는 gc_wp_cold로 분기
 *          ├─ 블록 erase
 *          └─ [RL 모드] Q-table 갱신
 *
 * [참고 논문]
 *   "Cleaning policies in mobile computers using flash memory"
 *   (M.-L. Chiang, R.-C. Chang, 1999)
 *   - CAT(Cost-Age-Times) 공식
 *   - M6 Fine-Grained Hot/Cold Data Redistribution
 *   - Age Transformation Function (Fig.7)
 *
 * ================================================================
 */

#ifndef _NVMEVIRT_CONV_FTL_H
#define _NVMEVIRT_CONV_FTL_H

#include <linux/types.h>
#include "pqueue/pqueue.h"  /* 우선순위 큐: Greedy의 Min-Heap에 사용 */
#include "ssd_config.h"     /* OP_AREA_PERCENT, SSD_PARTITIONS 등 상수 */
#include "ssd.h"            /* struct ssd, struct ppa, NAND 시뮬레이션 API */

/*
 * 전방 선언: struct conv_ftl이 아직 정의되기 전에
 * victim_select_fn 타입에서 포인터로 참조해야 하므로 필요.
 */
struct conv_ftl;

/*
 * victim_select_fn - GC 희생 라인 선택 함수의 타입
 *
 * [사용처]
 *   do_gc()의 파라미터로 전달된다.
 *   foreground_gc()에서 gc_mode에 따라 적절한 함수를 골라 넘긴다.
 *
 * [시그니처]
 *   @conv_ftl: FTL 인스턴스 (라인 정보, PQ 등에 접근)
 *   @force:    true면 효율성 체크 무시하고 무조건 하나 선택
 *              (foreground GC에서 공간이 급할 때 true)
 *   @return:   선택된 struct line* (NULL이면 후보 없음)
 *
 * [구현체 목록] (conv_ftl.c에 정의)
 *   select_victim_greedy()  — gc_mode=0
 *   select_victim_cb()      — gc_mode=1
 *   select_victim_cat()     — gc_mode=2
 *   select_victim_unified() — gc_mode=3 (RL 내부 전용)
 */
typedef struct line *(*victim_select_fn)(struct conv_ftl *, bool);

/* ================================================================
 * GC 모드 상수 (insmod gc_mode=N 으로 선택)
 *
 * [각 모드의 핵심 차이]
 *
 *   GREEDY (0):
 *     - VPC(유효 페이지)가 가장 적은 라인을 선택
 *     - PQ(Min-Heap)의 Root를 O(1)로 꺼냄
 *     - 장점: 빠르고 단순, uniform random 워크로드에 효율적
 *     - 단점: age 무시 → locality 높은 워크로드에서 hot 데이터 불필요 복사
 *
 *   CB (1) - Cost-Benefit:
 *     - Score = AgeWeight × IPC / (VPC + 1)
 *     - 전체 victim 큐를 Linear Scan O(N)
 *     - 장점: age 반영 → hot 데이터 보호, cold 우선 청소
 *     - 단점: erase 횟수 무시 → wear 불균형 발생 가능
 *
 *   CAT (2) - Cost-Age-Times:
 *     - Score = AgeWeight × IPC / ((VPC+1) × (EraseCount+1))
 *     - CB에 EraseCount를 분모에 추가 → 많이 지워진 블록 후순위
 *     - 장점: wear-leveling 포함
 *     - 단점: 워크로드가 바뀌면 고정된 공식이 최적이 아닐 수 있음
 *
 *   RL (3) - Reinforcement Learning:
 *     - Score = AgeWeight^α × IPC / ((VPC+1) × (EraseCount+1)^δ)
 *     - α, δ, hot_threshold를 Q-table이 state에 따라 자동 선택
 *     - 장점: 워크로드 변화에 적응 (phase별로 α, δ 조절)
 *     - 단점: 학습 초기 exploration으로 일시적 성능 저하
 *
 * ================================================================ */
#define GC_MODE_GREEDY  0
#define GC_MODE_CB      1
#define GC_MODE_CAT     2
#define GC_MODE_RL      3

/* ================================================================
 * GC Write Pointer 라우팅 상수
 *
 * [배경]
 *   GC가 valid page를 복사할 때, hot 페이지와 cold 페이지를
 *   서로 다른 블록(open line)에 분리해서 기록한다.
 *   이것이 논문의 "M6 Fine-Grained Data Redistribution".
 *
 * [사용 흐름]
 *   gc_write_page()에서:
 *     hot으로 판별됨 → get_new_page(ftl, GC_IO_HOT) → gc_wp_hot 블록에 기록
 *     cold로 판별됨  → get_new_page(ftl, GC_IO_COLD) → gc_wp_cold 블록에 기록
 *
 * [주의]
 *   이 값들은 conv_ftl 내부에서 write pointer를 고르는 데만 사용.
 *   NAND 타이밍 시뮬레이션(ssd_advance_nand)에는 항상 GC_IO를 넘긴다.
 *   (GC_IO는 ssd.h에 정의된 기존 상수)
 *
 * [값이 10, 11인 이유]
 *   ssd.h의 USER_IO(0), GC_IO(1)과 겹치지 않도록 충분히 큰 값 사용.
 *   __get_wp()의 switch문에서 USER_IO / GC_IO_HOT / GC_IO_COLD 3가지를 분기.
 *
 * ================================================================ */
#define GC_IO_HOT   10
#define GC_IO_COLD  11

/* ================================================================
 * RL (Reinforcement Learning) State/Action 공간 정의
 *
 * [설계 원칙]
 *   Q-table 크기 = NUM_STATES × NUM_ACTIONS = 54 × 36 = 1,944
 *   각 entry가 int64_t(8 bytes)이므로 총 ~15KB.
 *   실제 SSD 컨트롤러 DRAM(수백 MB)에서 충분히 수용 가능.
 *
 * ────────────────────────────────────────────
 * [State 설계] (54 = 3 × 3 × 3 × 2)
 *
 *   S1: free_line_ratio (3단계)
 *     ┌─────────┬────────────────────────────┐
 *     │ 0: 여유  │ free_line > 30% of total   │
 *     │ 1: 보통  │ 10% < free_line ≤ 30%      │
 *     │ 2: 긴급  │ free_line ≤ 10%            │
 *     └─────────┴────────────────────────────┘
 *     → 공간 압박 정도. 긴급하면 Greedy처럼 빠른 회수 필요.
 *
 *   S2: avg_victim_vpc_ratio (3단계)
 *     ┌──────────┬───────────────────────────────┐
 *     │ 0: 더러움 │ victim 평균 VPC < 25% of line │ ← GC 효율적
 *     │ 1: 보통   │ 25% ≤ VPC < 60%               │
 *     │ 2: 깨끗   │ VPC ≥ 60%                     │ ← GC 비효율적
 *     └──────────┴───────────────────────────────┘
 *     → victim들이 얼마나 "쓰레기가 많은지". 더러울수록 GC 이득.
 *
 *   S3: hot_cold_ratio (3단계)
 *     ┌────────────┬─────────────────────────────┐
 *     │ 0: cold 위주│ avg_hot_degree < 0.5 (×16)  │
 *     │ 1: 혼합     │ 0.5 ≤ avg < 3.0             │
 *     │ 2: hot 위주 │ avg ≥ 3.0                    │
 *     └────────────┴─────────────────────────────┘
 *     → 현재 워크로드의 hot/cold 편향.
 *       hot 위주면 age 가중치(α)를 높여 hot 보호 강화.
 *
 *   S4: wear_variance (2단계)
 *     ┌──────────┬──────────────────────────────┐
 *     │ 0: 균등   │ erase_max - erase_min ≤ 20   │
 *     │ 1: 불균등 │ 차이 > 20                     │
 *     └──────────┴──────────────────────────────┘
 *     → 불균등하면 δ(erase 가중치)를 높여 wear-leveling 강화.
 *
 * ────────────────────────────────────────────
 * [Action 설계] (36 = 3 × 4 × 3)
 *
 *   A1: alpha_level (3단계) — Age 가중치 지수
 *     ┌──────┬──────────┬──────────────────────────────┐
 *     │ 0    │ α = 0.5  │ sqrt(aw) — age 약하게 반영    │
 *     │ 1    │ α = 1.0  │ aw 그대로 — 기본 (= CB)      │
 *     │ 2    │ α = 2.0  │ aw² — age 강하게 (cold 우선)  │
 *     └──────┴──────────┴──────────────────────────────┘
 *
 *   A2: delta_level (4단계) — EraseCount 가중치 지수
 *     ┌──────┬──────────┬──────────────────────────────────┐
 *     │ 0    │ δ = 0.0  │ 1 (무시) — erase 안 봄 (= CB)   │
 *     │ 1    │ δ = 0.5  │ sqrt(ec+1) — 약간 반영          │
 *     │ 2    │ δ = 1.0  │ ec+1 그대로 (= CAT)             │
 *     │ 3    │ δ = 1.5  │ (ec+1)×sqrt(ec+1) — 강한 WL     │
 *     └──────┴──────────┴──────────────────────────────────┘
 *
 *   A3: hot_thresh_level (3단계) — Hot/Cold 분리 기준
 *     ┌──────┬──────────────┬─────────────────────────────────┐
 *     │ 0    │ avg × 0.5    │ 관대: 대부분 hot 취급 (분리 약)  │
 *     │ 1    │ avg × 1.0    │ 기본: 논문 기준 그대로           │
 *     │ 2    │ avg × 1.5    │ 엄격: 정말 hot인 것만 hot        │
 *     └──────┴──────────────┴─────────────────────────────────┘
 *
 *   이 3개 knob의 조합이 하나의 action.
 *   예: action=14 → alpha=1, delta=1, hot_thresh=2
 *       → CB 수준의 age 반영 + 약한 wear-leveling + 엄격한 hot 기준
 *
 * ────────────────────────────────────────────
 * [State 인덱싱 공식]
 *   state_idx = s1 × (NUM_S2 × NUM_S3 × NUM_S4)
 *             + s2 × (NUM_S3 × NUM_S4)
 *             + s3 × NUM_S4
 *             + s4
 *   → 0 ~ 53 범위의 정수
 *
 * [Action 인덱싱 공식]
 *   action_idx = a1 × (NUM_A2 × NUM_A3) + a2 × NUM_A3 + a3
 *   → 0 ~ 35 범위의 정수
 *
 * ================================================================ */
#define RL_NUM_S1       3
#define RL_NUM_S2       3
#define RL_NUM_S3       3
#define RL_NUM_S4       2
#define RL_NUM_STATES   (RL_NUM_S1 * RL_NUM_S2 * RL_NUM_S3 * RL_NUM_S4) /* 54 */

#define RL_NUM_A1       3   /* alpha levels */
#define RL_NUM_A2       4   /* delta levels */
#define RL_NUM_A3       3   /* hot threshold levels */
#define RL_NUM_ACTIONS  (RL_NUM_A1 * RL_NUM_A2 * RL_NUM_A3) /* 36 */

/*
 * Q-table 고정소수점 스케일
 *
 * 커널에서 float/double을 쓸 수 없으므로 모든 Q값, reward, 학습률을
 * 정수 × 1000 으로 표현한다.
 *
 * 예: Q값 2.5 → 2500으로 저장
 *     learning rate 0.1 → 100으로 저장
 *     reward -0.3 → -300으로 저장
 */
#define RL_Q_SCALE      1000

/*
 * RL 하이퍼파라미터 (모두 ×1000 고정소수점)
 *
 * RL_ALPHA (100 = 0.1):
 *   학습률. Q값을 새 경험에 얼마나 빠르게 반영할지.
 *   너무 크면 진동, 너무 작으면 수렴 느림.
 *   0.1은 tabular Q-learning의 일반적 기본값.
 *
 * RL_GAMMA (900 = 0.9):
 *   할인율. 미래 보상을 현재 가치로 얼마나 반영할지.
 *   0.9면 10스텝 후의 보상이 현재의 ~35%로 반영됨.
 *   GC는 한 번의 선택이 이후 여러 GC에 영향을 주므로 0.9가 적절.
 *
 * RL_EPSILON_INIT (300 = 0.3):
 *   초기 탐험율. 30%의 확률로 랜덤 action 선택.
 *   학습 초기에 다양한 (state, action) 쌍을 경험하기 위함.
 *
 * RL_EPSILON_MIN (50 = 0.05):
 *   최소 탐험율. 학습이 충분히 진행된 후에도 5%는 탐험.
 *   워크로드가 바뀔 때를 대비한 최소한의 적응력 유지.
 *
 * RL_EPSILON_DECAY (999):
 *   매 에피소드(GC 1회)마다 epsilon *= 0.999.
 *   GC 1000번 → epsilon이 0.3 → ~0.22 로 감소.
 *   GC 5000번 → ~0.02 근처로 수렴.
 */
#define RL_ALPHA        100
#define RL_GAMMA        900
#define RL_EPSILON_INIT 300
#define RL_EPSILON_MIN  50
#define RL_EPSILON_DECAY 999

/*
 * struct rl_config - RL 에이전트의 전체 상태
 *
 * [생명주기]
 *   init_rl()에서 초기화 → 매 GC(do_gc)마다 갱신 → rmmod 시 소멸
 *   (insmod 때마다 Q-table이 0으로 리셋됨.
 *    학습된 Q-table을 유지하려면 sysfs export 등이 필요하나 현재 미구현)
 *
 * [에피소드 = GC 1회]
 *   do_gc() 호출 1회가 RL의 1 에피소드에 해당.
 *   1) GC 직전: state 관찰 → action 선택 → 파라미터 적용
 *   2) GC 수행: unified scoring으로 victim 선택 + valid page 복사
 *   3) GC 직후: reward 계산 → Q(s,a) 갱신 → epsilon decay
 */
struct rl_config {
	/*
	 * Q-table: 핵심 학습 데이터
	 *
	 * q_table[state][action] = 해당 (state, action)의 예상 누적 보상.
	 * 값이 클수록 해당 상태에서 해당 action이 좋았다는 의미.
	 *
	 * 초기값 0 → 학습 초기에는 모든 action이 동일하게 평가됨
	 * → epsilon-greedy의 탐험이 초기 경험을 쌓아줌.
	 *
	 * 메모리: 54 × 36 × 8 = 15,552 bytes ≈ 15KB
	 */
	int64_t q_table[RL_NUM_STATES][RL_NUM_ACTIONS];

	/*
	 * 현재 에피소드의 state/action
	 *
	 * do_gc() 시작 시 설정되고, rl_update()에서 Q값 갱신에 사용.
	 * cur_state: rl_get_state()가 이산화한 환경 상태 (0~53)
	 * cur_action: rl_select_action()이 고른 action (0~35)
	 */
	uint32_t cur_state;
	uint32_t cur_action;

	/*
	 * 현재 action이 디코드된 파라미터 레벨
	 *
	 * rl_decode_action()에서 cur_action을 분해하여 여기에 저장.
	 * select_victim_unified()가 이 값들을 읽어 scoring 공식에 적용.
	 * rl_is_hot_page()가 hot_thresh_level을 읽어 분리 기준에 적용.
	 *
	 * 예: cur_action=14일 때
	 *   14 / (4×3) = 1 → alpha_level = 1 (α=1.0)
	 *   (14 / 3) % 4 = 1 → delta_level = 1 (δ=0.5)
	 *   14 % 3 = 2 → hot_thresh_level = 2 (avg×1.5)
	 */
	uint32_t alpha_level;      /* 0,1,2 → α = 0.5, 1.0, 2.0 */
	uint32_t delta_level;      /* 0,1,2,3 → δ = 0, 0.5, 1.0, 1.5 */
	uint32_t hot_thresh_level; /* 0,1,2 → avg×0.5, avg×1.0, avg×1.5 */

	/*
	 * epsilon: 현재 탐험율 (×1000 고정소수점)
	 *
	 * 매 에피소드마다 RL_EPSILON_DECAY(0.999)를 곱하여 감소.
	 * 300(30%)에서 시작 → GC 수천 회 후 50(5%)에 수렴.
	 *
	 * rl_select_action()에서:
	 *   random(0~999) < epsilon 이면 → 랜덤 action (탐험)
	 *   아니면 → argmax Q(state, a) (활용)
	 */
	uint32_t epsilon;

	/*
	 * Reward 계산용 스냅샷 (이전 GC 시점의 상태)
	 *
	 * GC 전후의 "변화량"으로 reward를 계산하기 위해
	 * 이전 GC 시점의 값을 기록해둔다.
	 *
	 * prev_copied_pages: 이전까지의 총 복사 페이지 수
	 *   → 현재 gc_copied_pages - prev = 이번 GC에서 복사한 양
	 *
	 * prev_erase_max/min: 이전까지의 erase_cnt 최대/최소
	 *   → 현재 max-min과 비교하여 wear 불균형 변화량 계산
	 */
	uint64_t prev_copied_pages;
	uint32_t prev_erase_max;
	uint32_t prev_erase_min;

	/*
	 * 통계 (dmesg 출력용, conv_flush에서 사용)
	 *
	 * total_episodes: GC 수행 횟수 = Q-table 갱신 횟수
	 * total_reward: 누적 reward (×1000). 평균 reward 계산에 사용.
	 */
	uint64_t total_episodes;
	uint64_t total_reward;
};

/* ================================================================
 * struct page_meta - per-LPN 메타데이터
 *
 * [존재 이유]
 *   GC 시 valid page를 hot/cold로 분류하려면 각 LPN이
 *   "얼마나 자주, 얼마나 최근에 수정되었는지" 알아야 한다.
 *
 * [논문 근거 - Section 3.2]
 *   "The hot degree of a block is defined as the number of times
 *    the block has been updated and decreases as the block's age grows.
 *    A block is defined as hot if its hot degree exceeds the average
 *    hot degree. Otherwise, the block is cold."
 *
 * [Hot Degree 계산] (conv_ftl.c의 calc_hot_degree)
 *   hot_degree = update_cnt × 1000 / get_age_weight(now - last_write_time)
 *
 *   - update_cnt가 클수록 → hot_degree ↑ (자주 수정됨)
 *   - 오래될수록 age_weight가 커져서 → hot_degree ↓ (식어감)
 *   - 1000은 정수 나눗셈 정밀도를 위한 스케일링
 *
 * [갱신 시점]
 *   conv_write()에서 매 LPN 쓰기마다:
 *     pm->update_cnt++
 *     pm->last_write_time = ktime_get_ns()
 *
 * [사용 시점]
 *   gc_write_page()에서 valid page 복사 직전:
 *     calc_hot_degree(pm, now) → avg와 비교 → hot이면 gc_wp_hot, cold면 gc_wp_cold
 *
 * [메모리 사용량]
 *   LPN당 12 bytes (uint32_t + uint64_t)
 *   예: 11.2GB SSD, 4KB 페이지 → ~2.8M LPN → ~33MB
 *   NVMeVirt는 호스트 DRAM을 쓰므로 문제없음.
 *   실제 SSD에서는 DRAM L2P 테이블 옆에 배치 가능.
 *
 * ================================================================ */
struct page_meta {
	uint32_t update_cnt;      /* 이 LPN이 덮어쓰기된 누적 횟수 */
	uint64_t last_write_time; /* 마지막 쓰기 시각 (나노초, ktime_get_ns) */
};

/* ================================================================
 * struct convparams - FTL 동작 파라미터
 *
 * conv_init_params()에서 설정되고 conv_ftl.cp에 복사된다.
 * 이후 GC 트리거 조건 등에서 참조된다.
 * ================================================================ */
struct convparams {
	/*
	 * gc_thres_lines / gc_thres_lines_high
	 *
	 * free_line_cnt가 이 값 이하로 떨어지면 GC가 트리거된다.
	 *
	 * 현재 둘 다 3으로 설정 (conv_init_params).
	 * 3인 이유: open block이 3개(user_wp, gc_wp_hot, gc_wp_cold)이므로
	 *           최소 3개의 free line이 항상 남아있어야 다음 블록을 열 수 있다.
	 *
	 * gc_thres_lines:      should_gc()에서 사용 (background GC용, 현재 미사용)
	 * gc_thres_lines_high: should_gc_high()에서 사용 (foreground GC 트리거)
	 *
	 * [호출 경로]
	 *   conv_write() → consume_write_credit() → check_and_refill_write_credit()
	 *   → foreground_gc() → should_gc_high() → 여기서 비교
	 */
	uint32_t gc_thres_lines;
	uint32_t gc_thres_lines_high;

	/*
	 * enable_gc_delay
	 *
	 * true면 GC 과정(read, write, erase)에서 NAND 타이밍 시뮬레이션 수행.
	 * false면 즉시 완료 (타이밍 무시, 순수 로직 테스트용).
	 * 현재 항상 1(true)로 설정.
	 */
	bool enable_gc_delay;

	/*
	 * op_area_pcent: Over-Provisioning 비율 (예: 0.07 = 7%)
	 *   호스트에 보이는 논리 용량 = 물리 용량 / (1 + op_area_pcent)
	 *   OP 영역은 GC가 valid page를 복사할 여유 공간으로 사용됨.
	 *
	 * pba_pcent: (물리공간 / 논리공간) × 100
	 *   conv_init_namespace()에서 ns->size 계산에 사용.
	 */
	double op_area_pcent;
	int pba_pcent;
};

/* ================================================================
 * struct line - 하나의 "라인" (= 슈퍼블록) 관리 단위
 *
 * [라인이란?]
 *   NVMeVirt에서 "라인"은 하나의 erase 단위(블록)에 대응.
 *   실제 SSD에서는 여러 채널/LUN에 걸친 슈퍼블록이 라인이 되지만,
 *   NVMeVirt에서는 blk_id = line_id로 1:1 매핑.
 *
 * [라인의 생명주기]
 *
 *   free_line_list  ──(할당)──→  open block (write pointer가 가리킴)
 *         ▲                              │
 *         │                        (페이지가 다 채워짐)
 *         │                              ▼
 *   mark_line_free()  ◄──(GC)──  victim_line_pq  ◄──  full_line_list
 *   (erase_cnt++)                  (ipc > 0)           (vpc == pgs_per_line)
 *
 *   1. free_line_list에서 꺼내져 write pointer에 할당됨
 *   2. 페이지가 하나씩 채워지다가 블록 끝에 도달하면:
 *      - 모든 페이지가 유효(vpc == pgs_per_line) → full_line_list로
 *      - 무효 페이지 존재(ipc > 0) → victim_line_pq로
 *   3. victim_line_pq에서 GC victim으로 선택되면:
 *      - valid page 복사 → 블록 erase → mark_line_free()
 *      - erase_cnt 증가 → 다시 free_line_list로 복귀
 *
 * ================================================================ */
struct line {
	int id;  /* 라인 ID = 물리적 블록 번호. 0 ~ tt_lines-1 */

	/*
	 * ipc / vpc - Invalid/Valid Page Count
	 *
	 * ipc + vpc = 해당 블록에 기록된 총 페이지 수 (≤ pgs_per_line)
	 * ipc: 덮어쓰기로 무효화된 페이지 수. GC가 회수할 수 있는 공간.
	 * vpc: 아직 유효한 페이지 수. GC 시 복사해야 할 양.
	 *
	 * [Greedy에서의 역할]
	 *   vpc가 작을수록 복사할 게 적으니 GC 효율적 → PQ에서 최소 vpc를 꺼냄.
	 *
	 * [CB/CAT에서의 역할]
	 *   scoring 공식의 분자(ipc)와 분모(vpc+1)에 사용.
	 *
	 * [갱신 시점]
	 *   mark_page_valid(): vpc++  (새 페이지 기록 시)
	 *   mark_page_invalid(): ipc++, vpc--  (덮어쓰기로 구버전 무효화 시)
	 *   mark_line_free(): 둘 다 0으로 리셋 (GC 완료 후 erase 시)
	 */
	int ipc;
	int vpc;

	/*
	 * entry: free_line_list 또는 full_line_list에 연결되는 리스트 노드.
	 *        victim_line_pq에 들어가 있을 때는 리스트에서 분리되어 있음.
	 */
	struct list_head entry;

	/*
	 * pos: victim_line_pq(우선순위 큐) 내부에서의 인덱스 위치.
	 *      0이면 큐에 없음을 의미.
	 *      pqueue_insert() 시 설정되고, pqueue_remove() 시 0으로 초기화.
	 *
	 *      mark_page_invalid()에서 line->pos가 0이 아니면
	 *      pqueue_change_priority()로 우선순위를 갱신한다.
	 */
	size_t pos;

	/*
	 * last_modified_time: 이 라인에 마지막으로 무효화(invalidation)가 발생한 시각.
	 *
	 * [CB/CAT에서의 역할]
	 *   age = now - last_modified_time
	 *   → get_age_weight(age)로 변환하여 scoring에 사용.
	 *   age가 클수록 = 오래전에 수정됨 = cold 데이터 = GC 우선.
	 *
	 * [갱신 시점]
	 *   mark_page_invalid()의 마지막 줄에서 ktime_get_ns()로 갱신.
	 *   → 페이지 덮어쓰기가 발생할 때마다 이 라인의 "최종 활동 시각"이 업데이트.
	 *
	 * [주의: page-level vs line-level age]
	 *   이것은 "라인 단위" age임. 라인 내 어떤 페이지든 무효화되면 갱신됨.
	 *   "페이지 단위" age는 page_meta.last_write_time이 별도로 관리.
	 *   CB/CAT victim 선택은 line-level age를, hot/cold 판별은 page-level age를 사용.
	 */
	uint64_t last_modified_time;

	/*
	 * erase_cnt: 이 라인(블록)이 erase된 누적 횟수.
	 *
	 * [CAT에서의 역할]
	 *   scoring 분모에 (erase_cnt + 1)이 들어감.
	 *   많이 지워진 블록은 점수가 낮아져 GC 후순위 → wear-leveling.
	 *
	 * [RL에서의 역할]
	 *   delta_level에 따라 (erase_cnt+1)^δ로 변환되어 분모에 적용.
	 *   state의 S4(wear_variance) 계산에도 사용: max - min > 20이면 불균등.
	 *
	 * [갱신 시점]
	 *   mark_line_free()에서 GC 완료 후 erase_cnt++ 로 증가.
	 *   mark_block_free()에서도 nand_block.erase_cnt가 별도로 증가하지만,
	 *   그것은 NAND 하드웨어 레벨의 카운터이고, 이것은 FTL 레벨의 카운터.
	 */
	uint32_t erase_cnt;
};

/* ================================================================
 * struct write_pointer - 다음 쓰기 위치를 가리키는 포인터
 *
 * [개수: 총 3개]
 *   wp:          유저 쓰기 (conv_write)
 *   gc_wp_hot:   GC hot page 복사 (gc_write_page에서 hot으로 판별 시)
 *   gc_wp_cold:  GC cold page 복사 (gc_write_page에서 cold로 판별 시)
 *
 * [동작 원리]
 *   각 wp는 하나의 "open line(블록)"을 잡고 있다.
 *   get_new_page()가 현재 wp 위치에서 새 PPA를 생성하고,
 *   advance_write_pointer()가 ch → lun → pg 순서로 인터리빙하며 전진.
 *   블록 끝에 도달하면 현재 라인을 full/victim으로 보내고 새 free line을 할당.
 *
 * [인터리빙 순서]
 *   pg → ch → lun → next_wordline → ... → 블록 끝 → 새 라인
 *   (채널/LUN 인터리빙으로 병렬성 극대화)
 *
 * ================================================================ */
struct write_pointer {
	struct line *curline; /* 현재 이 wp가 쓰고 있는 라인 */
	uint32_t ch;          /* 현재 채널 번호 */
	uint32_t lun;         /* 현재 LUN 번호 */
	uint32_t pg;          /* 현재 페이지 번호 (블록 내) */
	uint32_t blk;         /* 현재 블록 번호 (= curline->id) */
	uint32_t pl;          /* 현재 플레인 번호 (항상 0, 단일 플레인 가정) */
};

/* ================================================================
 * struct line_mgmt - 전체 라인 상태 관리
 *
 * [3가지 컨테이너]
 *   free_line_list:   빈 라인들의 연결 리스트 (FIFO)
 *   victim_line_pq:   GC 후보 라인들의 우선순위 큐 (Min-Heap by vpc)
 *   full_line_list:   모든 페이지가 유효한(vpc == max) 라인들의 리스트
 *
 * [victim_line_pq의 PQ 콜백]
 *   Greedy: vpc 기준 Min-Heap (작은 vpc가 위로)
 *   CB/CAT/RL: Linear Scan하므로 PQ의 정렬 순서에 의존하지 않음.
 *              다만 중간 원소 제거(pqueue_remove)와
 *              우선순위 갱신(pqueue_change_priority)은 PQ 구조가 필요.
 *
 * ================================================================ */
struct line_mgmt {
	struct line *lines;              /* 전체 라인 배열. lines[0] ~ lines[tt_lines-1] */
	struct list_head free_line_list; /* 빈 라인 리스트 */
	pqueue_t *victim_line_pq;        /* GC 후보 우선순위 큐 */
	struct list_head full_line_list; /* 꽉 찬 라인 리스트 */

	uint32_t tt_lines;        /* 전체 라인 수 */
	uint32_t free_line_cnt;   /* 현재 free 라인 수 */
	uint32_t victim_line_cnt; /* 현재 victim 후보 수 */
	uint32_t full_line_cnt;   /* 현재 full 라인 수 */
	/* 항상 성립: free + victim + full + 3(open blocks) = tt_lines */
};

/* ================================================================
 * struct write_flow_control - 쓰기 유량 제어
 *
 * [목적]
 *   호스트 쓰기가 계속되면 free line이 고갈된다.
 *   일정량 쓸 때마다 크레딧을 소모하고, 바닥나면 foreground GC를 강제.
 *
 * [동작 흐름]
 *   conv_write()의 매 LPN마다:
 *     consume_write_credit() → write_credits--
 *     check_and_refill_write_credit():
 *       if (write_credits <= 0):
 *         foreground_gc()  ← GC 수행
 *         write_credits += credits_to_refill  ← 리필
 *
 * [credits_to_refill의 동적 조절]
 *   do_gc()에서 victim의 ipc(회수 가능 페이지)로 설정.
 *   → GC로 많이 회수했으면 크레딧 많이 리필 → 다음 GC까지 여유.
 *   → GC로 적게 회수했으면 크레딧 적게 리필 → 빨리 다시 GC.
 *
 * ================================================================ */
struct write_flow_control {
	uint32_t write_credits;     /* 남은 쓰기 크레딧 */
	uint32_t credits_to_refill; /* GC 후 리필할 양 (= 이전 GC의 회수량) */
};

/* ================================================================
 * struct conv_ftl - FTL 인스턴스의 최상위 구조체
 *
 * [인스턴스 수]
 *   SSD_PARTITIONS(기본 1)개 생성. 멀티 파티션 시 각각 독립 운영.
 *   모든 파티션이 PCIe와 write_buffer를 공유.
 *
 * [메모리 할당]
 *   conv_init_ftl()에서:
 *     maptbl:    vmalloc(sizeof(ppa) × tt_pgs)    ← 수십~수백 MB
 *     rmap:      vmalloc(sizeof(uint64_t) × tt_pgs)
 *     page_meta: vmalloc(sizeof(page_meta) × tt_pgs) ← ~33MB (11.2GB SSD 기준)
 *     lm.lines:  vmalloc(sizeof(line) × tt_lines)
 *     rl.q_table: struct 내부 배열 (~15KB, 별도 할당 불필요)
 *
 * ================================================================ */
struct conv_ftl {
	struct ssd *ssd; /* 하부 SSD 하드웨어 모델. NAND 타이밍, 파라미터 등. */

	struct convparams cp; /* FTL 파라미터 (GC 임계값 등) */

	/*
	 * maptbl: LPN → PPA 매핑 테이블 (페이지 단위)
	 *
	 * maptbl[lpn] = 해당 논리 페이지가 저장된 물리 주소.
	 * UNMAPPED_PPA면 아직 한 번도 쓰이지 않은 LPN.
	 *
	 * [갱신 시점]
	 *   conv_write():    새 PPA 할당 후 maptbl[lpn] = new_ppa
	 *   gc_write_page(): valid page를 새 위치로 복사 후 갱신
	 */
	struct ppa *maptbl;

	/*
	 * rmap: PPA → LPN 역매핑 테이블
	 *
	 * rmap[pgidx] = 해당 물리 페이지에 저장된 논리 페이지 번호.
	 * INVALID_LPN이면 빈 물리 페이지.
	 *
	 * [사용 시점]
	 *   gc_write_page(): 물리 주소에서 LPN을 찾아 maptbl 갱신
	 *   (실제 SSD에서는 OOB 영역에 저장되는 정보)
	 */
	uint64_t *rmap;

	/*
	 * 쓰기 포인터 3개 (= open block 3개)
	 *
	 * wp:          유저 쓰기 전용. conv_write()에서 사용.
	 *              "Data newly written are treated as hot" (논문)
	 *              → hot 데이터가 자연스럽게 여기에 모임.
	 *
	 * gc_wp_hot:   GC에서 hot으로 판별된 valid page가 복사되는 곳.
	 *              이 블록은 hot 데이터만 모이므로, 곧 많이 무효화됨
	 *              → 다음 GC에서 회수 효율적 (낮은 vpc).
	 *
	 * gc_wp_cold:  GC에서 cold로 판별된 valid page가 복사되는 곳.
	 *              이 블록은 cold 데이터만 모이므로, 오래 유효 상태 유지
	 *              → GC 대상이 잘 안 됨 (높은 vpc) → 불필요한 복사 방지.
	 *
	 * [이것이 논문 M6의 핵심]
	 *   hot/cold 분리 없이 gc_wp 1개만 쓰면 → hot+cold 뒤섞임
	 *   → cold가 딸려서 복사됨 → erase 증가 → 성능 하락
	 */
	struct write_pointer wp;
	struct write_pointer gc_wp_hot;
	struct write_pointer gc_wp_cold;

	struct line_mgmt lm;             /* 라인 상태 관리 */
	struct write_flow_control wfc;   /* 쓰기 유량 제어 */

	/*
	 * page_meta: per-LPN 메타데이터 배열
	 *
	 * page_meta[lpn].update_cnt / last_write_time으로
	 * 각 LPN의 hot degree를 계산한다.
	 *
	 * [갱신] conv_write()에서 매 쓰기마다
	 * [조회] gc_write_page()에서 hot/cold 판별 시
	 */
	struct page_meta *page_meta;

	/*
	 * avg_hot_degree: 전체 LPN의 평균 hot degree (×16 고정소수점)
	 *
	 * EMA(Exponential Moving Average)로 유지:
	 *   avg = (avg × 15 + new_degree × 16) / 16
	 *   → 최근 값에 1/16(6.25%) 가중, 기존 값에 15/16(93.75%) 가중.
	 *
	 * ×16인 이유: 정수 나눗셈에서 정밀도 확보. avg=16이면 실제 평균=1.0.
	 *
	 * [갱신] conv_write()에서 매 쓰기마다
	 * [사용]
	 *   is_hot_page(): (degree × 16) >= avg_hot_degree 이면 hot
	 *   rl_is_hot_page(): hot_thresh_level에 따라 avg를 0.5~1.5배 조절
	 */
	uint64_t avg_hot_degree;

	/*
	 * RL 에이전트 상태
	 *
	 * gc_mode == GC_MODE_RL일 때만 활성적으로 사용.
	 * 다른 모드에서는 init_rl()에서 초기화만 되고 이후 무시.
	 * (메모리는 struct 안에 포함이라 별도 할당/해제 불필요)
	 */
	struct rl_config rl;

	/*
	 * 통계 카운터
	 *
	 * gc_count: 총 GC 수행 횟수. do_gc()에서 증가.
	 * gc_copied_pages: GC로 복사된 총 페이지 수. gc_write_page()에서 증가.
	 *
	 * conv_flush()에서 dmesg로 출력.
	 * RL reward 계산에서도 사용 (이번 GC의 복사량 = 현재 - prev).
	 */
	uint64_t gc_count;
	uint64_t gc_copied_pages;
};

/* ================================================================
 * 외부 인터페이스 함수 선언
 *
 * conv_init_namespace():
 *   NVMeVirt가 네임스페이스를 생성할 때 호출.
 *   SSD 파라미터 초기화 → FTL 인스턴스 생성 → IO 핸들러 등록.
 *
 * conv_remove_namespace():
 *   rmmod 시 호출. 모든 메모리 해제.
 *
 * conv_proc_nvme_io_cmd():
 *   NVMe IO 커맨드 디스패처. opcode에 따라 read/write/flush 분기.
 *   ns->proc_io_cmd에 등록되어 NVMeVirt 코어에서 호출됨.
 *
 * ================================================================ */
void conv_init_namespace(struct nvmev_ns *ns, uint32_t id, uint64_t size,
			 void *mapped_addr, uint32_t cpu_nr_dispatcher);
void conv_remove_namespace(struct nvmev_ns *ns);
bool conv_proc_nvme_io_cmd(struct nvmev_ns *ns, struct nvmev_request *req,
			   struct nvmev_result *ret);

#endif /* _NVMEVIRT_CONV_FTL_H */