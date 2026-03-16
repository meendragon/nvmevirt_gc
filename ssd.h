// SPDX-License-Identifier: GPL-2.0-only

#ifndef _NVMEVIRT_SSD_H
#define _NVMEVIRT_SSD_H

#include <linux/types.h>
#include "pqueue/pqueue.h"
#include "ssd_config.h"
#include "channel_model.h"

/*
    [메모리 할당 크기 예시 (Default malloc size)]
    섹터 크기가 512B일 때의 계층별 개수 예시입니다.
    이 구조에 따라 거대한 배열들이 동적으로 할당됩니다.
    
    Channel = 40 * 8 = 320
    LUN     = 40 * 8 = 320
    Plane   = 16 * 1 = 16
    Block   = 32 * 256 = 8192
    Page    = 16 * 256 = 4096
    Sector  = 4 * 8 = 32
    ...
*/

// 유효하지 않은 주소/LPN을 나타내는 매직 넘버 (모트 비트가 1)
#define INVALID_PPA (~(0ULL))
#define INVALID_LPN (~(0ULL))
#define UNMAPPED_PPA (~(0ULL))

/* 낸드 플래시 명령 타입 */
enum {
    NAND_READ = 0,
    NAND_WRITE = 1,
    NAND_ERASE = 2,
    NAND_NOP = 3, // No Operation (시간 동기화용)
};

/* IO 요청의 주체 구분 */
enum {
    USER_IO = 0, // 호스트(사용자)가 보낸 요청 (처리 우선순위 높음)
    GC_IO = 1,   // 내부 GC가 생성한 요청 (Valid Page Copy 등)
    MIG_IO = 2
};

/* 섹터 및 페이지 상태 */
enum {
    SEC_FREE = 0,    // 비어 있음
    SEC_INVALID = 1, // 구버전 데이터 (쓰레기)
    SEC_VALID = 2,   // 최신 유효 데이터

    PG_FREE = 0,
    PG_INVALID = 1,
    PG_VALID = 2
};

/* 셀 타입 (Cell Type) - MLC/TLC/QLC 특성 반영 */
// LSB/CSB/MSB 페이지에 따라 읽기/쓰기 속도가 다름을 시뮬레이션하기 위함
enum { CELL_TYPE_LSB, CELL_TYPE_MSB, CELL_TYPE_CSB, MAX_CELL_TYPES };

/* * [PPA 비트맵 구조 정의]
 * 64비트 정수 하나에 물리적 주소(Channel, LUN, Block, Page)를 구겨 넣는 방식입니다.
 * 비트 연산을 통해 빠르게 주소를 해석합니다.
 */
#define TOTAL_PPA_BITS (64)
#define BLK_BITS (16)   // 블록 번호용 비트 수
#define PAGE_BITS (16)  // 페이지 번호용 비트 수
#define PL_BITS (8)     // 플레인 번호용 비트 수
#define LUN_BITS (8)    // LUN 번호용 비트 수
#define CH_BITS (8)     // 채널 번호용 비트 수
// 남는 비트 (Reserved)
#define RSB_BITS (TOTAL_PPA_BITS - (BLK_BITS + PAGE_BITS + PL_BITS + LUN_BITS + CH_BITS))

/* * @brief 물리적 페이지 주소 (Physical Page Address) 구조체
 * Union을 사용하여 같은 64비트 데이터를 3가지 방식으로 접근 가능하게 함
 */
struct ppa {
    union {
        // 1. 개별 필드로 접근 (비트 필드 구조체)
        struct {
            uint64_t pg : PAGE_BITS; // 4KB 단위 페이지 번호
            uint64_t blk : BLK_BITS; // 블록 번호
            uint64_t pl : PL_BITS;   // 플레인 번호
            uint64_t lun : LUN_BITS; // LUN(Die) 번호
            uint64_t ch : CH_BITS;   // 채널 번호
            uint64_t rsv : RSB_BITS; // 예약됨
        } g;

        // 2. 계층적 접근 (페이지 vs 나머지 상위 주소)
        struct {
            uint64_t : PAGE_BITS;
            uint64_t blk_in_ssd : BLK_BITS + PL_BITS + LUN_BITS + CH_BITS; // SSD 전체에서의 블록 고유 번호
            uint64_t rsv : RSB_BITS;
        } h;

        // 3. 통짜 정수로 접근 (비교/대입용)
        uint64_t ppa;
    };
};

typedef int nand_sec_status_t;

/* 낸드 페이지 구조체 (메타데이터) */
struct nand_page {
    nand_sec_status_t *sec; // 섹터별 상태 배열
    int nsecs;  // 페이지 당 섹터 수
    int status; // 페이지 상태 (FREE/VALID/INVALID)
};

/* * @brief 낸드 블록 구조체
 * GC의 핵심 관리 단위입니다.
 */
struct nand_block {
    struct nand_page *pg; // 페이지 배열 포인터
    int npgs;             // 블록 당 페이지 수
    
    int ipc; /* Invalid Page Count: 무효 페이지 수 (GC 이득 계산용) */
    int vpc; /* Valid Page Count: 유효 페이지 수 (GC 비용 계산용) */
    
    int erase_cnt; /* Erase Count: 지운 횟수 (수명/Wear-leveling 관리용) */
    int wp;        /* Write Pointer: 현재 쓰고 있는 페이지 위치 (순차 쓰기용) */
};

/* 낸드 플레인 구조체 (블록의 집합) */
struct nand_plane {
    struct nand_block *blk; // 블록 배열
    uint64_t next_pln_avail_time; // 플레인 사용 가능 시간
    int nblks;
};

/* * @brief 낸드 LUN (Die) 구조체
 * 독립적으로 명령을 수행할 수 있는 최소 단위입니다.
 */
struct nand_lun {
    struct nand_plane *pl;
    int npls;
    
    // [중요] LUN Busy Time 시뮬레이션
    // 이 LUN이 언제까지 바쁜지 기록하여, 다음 명령의 시작 시간을 지연시킴
    uint64_t next_lun_avail_time; 
    bool busy;
    uint64_t gc_endtime;
};

/* SSD 채널 구조체 (버스) */
struct ssd_channel {
    struct nand_lun *lun; // LUN 배열
    int nluns;
    uint64_t gc_endtime;
    
    // 채널 대역폭 모델 (데이터 전송 지연 계산용)
    struct channel_model *perf_model;
};

/* PCIe 인터페이스 구조체 */
struct ssd_pcie {
    struct channel_model *perf_model; // PCIe 대역폭 모델
};

/* 내부 낸드 명령 구조체 (시뮬레이터 전달용) */
struct nand_cmd {
    int type; // USER_IO or GC_IO
    int cmd;  // READ/WRITE/ERASE
    uint64_t xfer_size; // 전송 크기 (byte)
    uint64_t stime; /* 요청 도착 시간 (Start Time) */
    bool interleave_pci_dma; // PCIe 전송과 겹쳐서 수행할지 여부
    struct ppa *ppa; // 대상 주소
};

/* 쓰기 버퍼 구조체 (DRAM 시뮬레이션) */
struct buffer {
    size_t size;      // 전체 크기
    size_t remaining; // 남은 크기
    spinlock_t lock;  // 동시성 제어용 락
};

/*
 * [용어 정리]
 * pg (page): FTL 매핑 단위 (4KB, 논리적 최소 단위)
 * flashpg (flash page): 낸드 물리적 읽기 단위 (Sensing Unit, tR)
 * oneshotpg (oneshot page): 낸드 물리적 쓰기 단위 (Program Unit, tPROG)
 * - 예: TLC는 3개의 페이지를 한 번에 씀 (One-shot Program)
 * blk (block): 지우기(Erase) 단위
 * lun (die): 동작 수행 단위 (병렬성 단위)
 * ch (channel): 데이터 전송 통로
 */

/* * @brief SSD 파라미터 구조체 (스펙 시트)
 * SSD의 모든 기하학적 구조(Geometry)와 타이밍(Latency) 정보가 저장됩니다.
 * ssd_config.h의 설정을 바탕으로 ssd.c에서 계산되어 채워집니다.
 */
struct ssdparams {
    int secsz; /* 섹터 크기 (Bytes) */
    int secs_per_pg; /* 페이지 당 섹터 수 */
    int pgsz; /* 매핑 단위 크기 (Bytes, 보통 4KB) */
    
    int pgs_per_flashpg; /* 물리 페이지(Flash Page) 당 논리 페이지(4KB) 수 */
    int flashpgs_per_blk; /* 블록 당 물리 페이지 수 */
    int slc_flashpgs_per_blk;

    int pgs_per_oneshotpg; /* 원샷 프로그램 단위 당 논리 페이지 수 */
    int slc_pgs_per_oneshotpg;
    int oneshotpgs_per_blk; /* 블록 당 원샷 프로그램 수 */
    int slc_oneshotpgs_per_blk;
    
    int pgs_per_blk; /* 블록 당 총 논리 페이지 수 */
    int slc_pgs_per_blk;
    int blks_per_pl; /* 플레인 당 블록 수 */
    int slc_blks_per_pl;
    int pls_per_lun; /* LUN 당 플레인 수 */
    int luns_per_ch; /* 채널 당 LUN 수 */
    int nchs; /* 전체 채널 수 */
    int cell_mode; /* 셀 모드 (SLC/MLC/TLC) */

    /* NVMe 쓰기 단위 (이 단위의 배수로 전송됨) */
    int write_unit_size;
    bool write_early_completion; // 버퍼에만 쓰면 완료 처리할지 여부

    /* 지연 시간 (Latency) 정보 - 나노초(ns) 단위 */
    int pg_4kb_rd_lat[MAX_CELL_TYPES]; // 4KB 부분 읽기 시간
    int pg_rd_lat[MAX_CELL_TYPES];     // 전체 페이지 읽기 시간 (tR)
    int pg_wr_lat;                     // 페이지 쓰기 시간 (tPROG)
    int blk_er_lat;                    // 블록 지우기 시간 (tBERS)
    int max_ch_xfer_size;              // 채널 최대 전송 크기
    
    /* 지연 시간 (Latency) 정보 - 나노초(ns) 단위 */
    int slc_pg_4kb_rd_lat; // 4KB 부분 읽기 시간
    int slc_pg_rd_lat;     // 전체 페이지 읽기 시간 (tR)
    int slc_pg_wr_lat;                     // 페이지 쓰기 시간 (tPROG)
    int slc_blk_er_lat;                    // 블록 지우기 시간 (tBERS)

    /* 펌웨어(F/W) 오버헤드 시뮬레이션 값 */
    int fw_4kb_rd_lat; 
    int fw_rd_lat; 
    int fw_wbuf_lat0; 
    int fw_wbuf_lat1; 
    int fw_ch_xfer_lat; 

    uint64_t ch_bandwidth;   /* 낸드 채널 대역폭 (MiB/s) */
    uint64_t pcie_bandwidth; /* PCIe 대역폭 (MiB/s) */

    /* [계산된 총계 값들 (Total Counts)] - 초기화 시 자동 계산됨 */
    unsigned long secs_per_blk;
    unsigned long slc_secs_per_blk;
    unsigned long secs_per_pl;
    unsigned long slc_secs_per_pl;
    unsigned long secs_per_lun;
    unsigned long slc_secs_per_lun;
    unsigned long secs_per_ch;
    unsigned long slc_secs_per_ch;
    unsigned long tt_secs; /* SSD 전체 섹터 수 */
    unsigned long slc_tt_secs;
    unsigned long tlc_tt_secs;

    unsigned long pgs_per_pl;
    unsigned long slc_pgs_per_pl;
    unsigned long pgs_per_lun;
    unsigned long slc_pgs_per_lun;
    unsigned long pgs_per_ch;
    unsigned long slc_pgs_per_ch;
    unsigned long tt_pgs; /* SSD 전체 페이지 수 */
    unsigned long slc_tt_pgs;
    unsigned long tlc_tt_pgs;

    unsigned long blks_per_lun;
    unsigned long blks_per_ch;
    unsigned long tt_blks; /* SSD 전체 블록 수 */


    unsigned long secs_per_line;
    unsigned long slc_secs_per_line;
    unsigned long pgs_per_line;
    unsigned long slc_pgs_per_line;
    unsigned long blks_per_line;
    
    unsigned long tt_lines; /* 전체 라인(Superblock) 수 */
    unsigned long slc_tt_lines; //slc 라인수
    unsigned long tlc_tt_lines; //tlc 라인수

    unsigned long pls_per_ch;
    unsigned long tt_pls;

    unsigned long tt_luns;

    unsigned long long write_buffer_size; // 쓰기 버퍼 크기
};

/* SSD 최상위 구조체 */
struct ssd {
    struct ssdparams sp;   // 파라미터 정보
    struct ssd_channel *ch; // 채널 배열 포인터
    struct ssd_pcie *pcie;  // PCIe 인터페이스
    struct buffer *write_buffer; // 쓰기 버퍼
    unsigned int cpu_nr_dispatcher; // 연결된 CPU 코어 번호
};

/* * [Inline Helper Functions]
 * PPA(주소)를 받아서 해당 계층의 구조체 포인터를 반환하는 함수들
 * 반복적인 배열 인덱싱 코드를 줄여줌
 */

// PPA에 해당하는 채널 구조체 반환
static inline struct ssd_channel *get_ch(struct ssd *ssd, struct ppa *ppa)
{
    return &(ssd->ch[ppa->g.ch]);
}

// PPA에 해당하는 LUN 구조체 반환
static inline struct nand_lun *get_lun(struct ssd *ssd, struct ppa *ppa)
{
    struct ssd_channel *ch = get_ch(ssd, ppa);
    return &(ch->lun[ppa->g.lun]);
}

// PPA에 해당하는 플레인 구조체 반환
static inline struct nand_plane *get_pl(struct ssd *ssd, struct ppa *ppa)
{
    struct nand_lun *lun = get_lun(ssd, ppa);
    return &(lun->pl[ppa->g.pl]);
}

// PPA에 해당하는 블록 구조체 반환
static inline struct nand_block *get_blk(struct ssd *ssd, struct ppa *ppa)
{
    struct nand_plane *pl = get_pl(ssd, ppa);
    return &(pl->blk[ppa->g.blk]);
}

// PPA에 해당하는 페이지 구조체 반환
static inline struct nand_page *get_pg(struct ssd *ssd, struct ppa *ppa)
{
    struct nand_block *blk = get_blk(ssd, ppa);
    return &(blk->pg[ppa->g.pg]);
}

// PPA의 페이지 번호를 보고 셀 타입(LSB/CSB/MSB) 계산
static inline uint32_t get_cell(struct ssd *ssd, struct ppa *ppa)
{
    struct ssdparams *spp = &ssd->sp;
    // 페이지 번호를 플래시 페이지 단위로 나눈 뒤, 셀 모드로 나머지 연산
    return (ppa->g.pg / spp->pgs_per_flashpg) % (spp->cell_mode);
}

/* 함수 원형 선언 (ssd.c 구현) */
void ssd_init_params(struct ssdparams *spp, uint64_t capacity, uint32_t nparts);
void ssd_init(struct ssd *ssd, struct ssdparams *spp, uint32_t cpu_nr_dispatcher);
void ssd_remove(struct ssd *ssd);

uint64_t ssd_advance_nand(struct ssd *ssd, struct nand_cmd *ncmd);
uint64_t ssd_advance_pcie(struct ssd *ssd, uint64_t request_time, uint64_t length);
uint64_t ssd_advance_write_buffer(struct ssd *ssd, uint64_t request_time, uint64_t length);
uint64_t ssd_next_idle_time(struct ssd *ssd);

void buffer_init(struct buffer *buf, size_t size);
uint32_t buffer_allocate(struct buffer *buf, size_t size);
bool buffer_release(struct buffer *buf, size_t size);
void buffer_refill(struct buffer *buf);

void adjust_ftl_latency(int target, int lat);
#endif