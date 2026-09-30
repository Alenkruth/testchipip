// See LICENSE.SiFive for license details.

#include "testchip_dtm.h"
#include <riscv/encoding.h>
#include <fesvr/dtm.h>
#include <riscv/debug_defines.h>
#include <fstream>
#include <vector>

#define RV_X(x, s, n) \
  (((x) >> (s)) & ((1 << (n)) - 1))
#define ENCODE_ITYPE_IMM(x) \
  (RV_X(x, 0, 12) << 20)
#define LOAD(xlen, dst, base, imm) \
  (((xlen) == 64 ? 0x00003003 : 0x00002003) \
   | ((dst) << 7) | ((base) << 15) | (uint32_t)ENCODE_ITYPE_IMM(imm))
#define FLOAD(flen, dst, base, imm)                                     \
  (((flen) == 64 ? 0x00003007 : 0x00002007)                             \
   | ((dst) << 7) | ((base) << 15) | (uint32_t)ENCODE_ITYPE_IMM(imm))
#define ADDI(dst, src, imm) (0x13 | ((dst) << 7) | ((src) << 15) | (uint32_t)ENCODE_ITYPE_IMM(imm))
#define VL1R(dst, base) (0x02800007 | ((dst) << 7) | (base << 15))
#define VSETVL(dst, rs1, rs2) (0x80007057 | ((dst) << 7) | ((rs1) << 15) | ((rs2) << 20))
#define EBREAK  0x00100073
#define X0 0
#define S0 8
#define S1 9
#define WRITE 1
#define SET 2
#define CLEAR 3
#define CSRRx(type, dst, csr, src) (0x73 | ((type) << 12) | ((dst) << 7) | ((src) << 15) | (uint32_t)((csr) << 20))

#define AC_AR_REGNO(x) ((0x1000 | x) << AC_ACCESS_REGISTER_REGNO_OFFSET)
#define AC_AR_SIZE(x)  (((x == 128)? 4 : (x == 64 ? 3 : 2)) << AC_ACCESS_REGISTER_AARSIZE_OFFSET)

#define AC_RUN_OR_ABORT(a, b, c, d, e) {                                \
  uint32_t cmderr = run_abstract_command(a, b, c, d, e);                \
  if (cmderr) {                                                         \
    printf("Error %x\n", cmderr);                                       \
    abort();                                                            \
  }                                                                     \
  }

#define MSIP_BASE 0x2000000

#define MAX_VLEN (8 * 32) // Limited by debug module capacity

testchip_dtm_t::testchip_dtm_t(int argc, char** argv, bool can_have_loadmem)
  : testchip_dtm_arg_filter_t(argc, argv),
    dtm_t(filt_argc, filt_argv.data()),
    loadarch_done(false)
{
  has_loadmem = false;
  is_loadmem = false;
  loadarch_file = "";
  reconfig_csrs_file = "";
  restore_cache_wipe = false;
  std::vector<std::string> args(argv + 1, argv + argc);
  for (auto& arg : args) {
    if (arg.find("+loadmem=") == 0)
      has_loadmem = can_have_loadmem;
    if (arg.find("+loadarch=") == 0)
      loadarch_file = arg.substr(strlen("+loadarch="));
    if (arg.find("+reconfig_csrs=") == 0)
      reconfig_csrs_file = arg.substr(strlen("+reconfig_csrs="));
    if (arg.find("+restore-cache-wipe") == 0)
      restore_cache_wipe = true;   // [reconf-fix Phase C]
  }
  if (reconfig_csrs_file != "") {
    std::ifstream rcf(reconfig_csrs_file);
    if (!rcf.is_open()) {
      printf("reconfig_csrs: could not open %s\n", reconfig_csrs_file.c_str());
      abort();
    }
    std::string line;
    while (std::getline(rcf, line)) {
      if (line.empty() || line[0] == '#') continue;
      auto eq = line.find('=');
      if (eq == std::string::npos) continue;
      uint32_t addr = (uint32_t)std::stoull(line.substr(0, eq), nullptr, 0);
      uint64_t val  = std::stoull(line.substr(eq + 1), nullptr, 0);
      reconfig_csrs.push_back({addr, val});
    }
    printf("reconfig_csrs: loaded %zu writes from %s\n", reconfig_csrs.size(), reconfig_csrs_file.c_str());
  }
  testchip_htif_t::parse_htif_args(args);
}


// [2026-08-10] DO NOT "simplify" this predicate back to bare `is_loadmem`.
//
// WHY: dtm_t::{read,write}_chunk is the DMI path -- it HALTS the hart, drives the access
// via abstract commands / the program buffer, and resumes. After loadarch, FESVR polls
// tohost/fromhost through read_chunk for the whole run. MEASURED with branch counters:
// >100,000 read_chunk calls per run. Sending those down DMI stalls the core continuously
// and evicts the working set from both L1s.
//
// MEASURED on 648.exchange2_s/sp_0 (same bitstream/checkpoint/plusargs, 100M-instr window):
//     `is_loadmem || (loadarch_done && has_loadmem)` -> IPC 1.4497, D$  0.0073 MPKI (6 runs)
//     bare `is_loadmem`                              -> IPC 0.5488, D$ 49.49   MPKI (3 runs)
// 2.6x, reproducible in both directions. is_loadmem alone is insufficient: it is set only
// inside load_program() and cleared immediately after, so it is false for the entire run.
// loadarch_done is what makes the post-restore traffic take the fast path.
//
// TRAP FOR FUTURE DEBUGGING: firesim_dtm_t::load_mem_read/load_mem_write fprintf(stdout)
// on EVERY call, yet those lines appear ZERO times in the uartlog even though temporary
// branch counters measured >100k calls. That stdout is swallowed somewhere. Do NOT infer
// "never called" from their absence -- an earlier investigation did exactly that and
// wrongly retracted this explanation. If you need call counts, add temporary counters
// printing to STDERR (they were removed here to keep the hot path clean).
//
// Full history: claude-artifacts/37-session-changelog-2026-08-09.md
void testchip_dtm_t::write_chunk(addr_t taddr, size_t nbytes, const void* src)
{
  if (is_loadmem || (loadarch_done && has_loadmem)) {
    load_mem_write(taddr, nbytes, src);
  } else {
    dtm_t::write_chunk(taddr, nbytes, src);
  }
}

void testchip_dtm_t::read_chunk(addr_t taddr, size_t nbytes, void* dst)
{
  if (is_loadmem || (loadarch_done && has_loadmem)) {
    load_mem_read(taddr, nbytes, dst);
  } else {
    dtm_t::read_chunk(taddr, nbytes, dst);
  }
}

reg_t read_priv(std::string priv_str) {
  reg_t prv;
  if (priv_str == "U") {
    prv = 0;
  } else if (priv_str == "S") {
    prv = 1;
  } else if (priv_str == "M") {
    prv = 3;
  } else {
    printf("loadarch illegal privilege mode %s\n", priv_str.c_str());
    abort();
  }
  return prv;
}


void testchip_dtm_t::loadarch_restore_reg(uint32_t regno, reg_t reg) {
  uint32_t data[2];
  data[0] = (uint32_t) reg;
  data[1] = (uint32_t) (reg >> 32);
  uint32_t cmd = (AC_ACCESS_REGISTER_TRANSFER |
                  AC_ACCESS_REGISTER_WRITE |
                  AC_AR_SIZE(64) |
                  AC_AR_REGNO(regno));
  printf("loadarch restoring reg %d=%lx\n", regno, reg);
  AC_RUN_OR_ABORT(cmd, 0, 0, data, 2);
}

void testchip_dtm_t::loadarch_restore_freg(uint32_t regno, reg_t reg) {
  uint32_t prog[2] = {
    FLOAD(64, regno, X0, get_data_base()),
    EBREAK
  };
  uint32_t data[2];
  data[0] = (uint32_t) reg;
  data[1] = (uint32_t) (reg >> 32);
  uint32_t cmd = (AC_ACCESS_REGISTER_POSTEXEC |
                  AC_ACCESS_REGISTER_TRANSFER |
                  AC_ACCESS_REGISTER_WRITE |
                  AC_AR_SIZE(64) |
                  AC_AR_REGNO(0));
  printf("loadarch restoring freg %d=%lx\n", regno, reg);
  AC_RUN_OR_ABORT(cmd, prog, 2, data, 2);
}

// TODO: Check that DM_ABSTRACTCS_DATACOUNT is large enough for a vreg
void testchip_dtm_t::loadarch_restore_vreg(uint32_t regno, unsigned char* reg, size_t bytes) {
  uint32_t data[MAX_VLEN / (8 * 4)];
  for (size_t i = 0; i < bytes; i += 4) {
    memcpy(&data[i/4], reg + i, 4);
  }
  uint32_t prog[3] = {
    ADDI(S0, X0, get_data_base()),
    VL1R(regno, S0),
    EBREAK
  };
  uint32_t cmd = (AC_ACCESS_REGISTER_POSTEXEC |
		  AC_ACCESS_REGISTER_TRANSFER |
		  AC_ACCESS_REGISTER_WRITE |
		  AC_AR_SIZE(64) |
		  AC_AR_REGNO(0));
  printf("loadarch restoring vreg %d=", regno);
  for (size_t i = bytes; i-- > 0; ) {
    printf("%02x", reg[i]);
  }
  printf("\n");
  AC_RUN_OR_ABORT(cmd, prog, 3, data, bytes / 4);
}

void testchip_dtm_t::loadarch_restore_vtype(uint32_t vtype) {
  uint32_t prog[2] = {
    VSETVL(X0, X0, S0),
    EBREAK
  };
  uint32_t data[1] = {vtype};
  uint32_t cmd = (AC_ACCESS_REGISTER_POSTEXEC |
		  AC_ACCESS_REGISTER_TRANSFER |
		  AC_ACCESS_REGISTER_WRITE |
		  AC_AR_SIZE(64) |
		  AC_AR_REGNO(S0));
  printf("loadarch restoring vtype=%x\n", vtype);
  AC_RUN_OR_ABORT(cmd, prog, 2, data, 1);
}

void testchip_dtm_t::loadarch_restore_csr(uint32_t regno, reg_t reg) {
  uint32_t prog[5] = {
    CSRRx(WRITE, S0, CSR_DSCRATCH0, S0),
    LOAD(64, S0, X0, get_data_base()),
    CSRRx(WRITE, S0, regno, S0),
    CSRRx(WRITE, S0, CSR_DSCRATCH0, S0),
    EBREAK
  };
  uint32_t data[2];
  data[0] = (uint32_t) reg;
  data[1] = (uint32_t) (reg >> 32);
  uint32_t cmd = (AC_ACCESS_REGISTER_POSTEXEC |
                  AC_ACCESS_REGISTER_TRANSFER |
                  AC_ACCESS_REGISTER_WRITE |
                  AC_AR_SIZE(64) |
                  AC_AR_REGNO(0));
  printf("loadarch restoring csr %x=%lx\n", regno, reg);
  AC_RUN_OR_ABORT(cmd, prog, 5, data, 2);
}

// [reconf-fix Phase C] RESTORE-ONLY cache wipe (doc 30 Task 9).
// Order is load-bearing: L1 discard walk FIRST (so no L1 release can be in
// flight into a wiping L2), then the L2 directory wipe.  Both flows DISCARD —
// every cached line is doomed pre-restore state; DRAM holds (or is about to
// hold) the checkpoint image, and nothing cached may alias it.  Runs with the
// hart halted, before any architectural state is restored (the program-buffer
// helpers clobber S0, which is fine here and only here).
// Signaling rule: no target tohost stores happen between the wipe and the
// loadmem — all host<->target traffic in this window is DMI only.
#define CSR_CF_CACHECTL       0xbc4
#define CSR_CF_DCACHE_STATUS  0xbcd
#define CF_L2_WIPE_ADDR       0x2010100

// Every access below is a bare abstract command on the ALREADY-halted hart.
// dtm_t's conveniences (read_csr/write_csr/read_chunk/write_chunk) all wrap
// themselves in halt()..resume() — a resume here would set the wiped,
// not-yet-restored hart running wild, so none of them may be used in this
// window.  S0/S1 are clobbered freely: flush runs BEFORE any architectural
// state is restored.
#define CF_LUI(dst, imm20) (0x37 | ((dst) << 7) | ((uint32_t)(imm20) << 12))
#define CF_LW(dst, base, imm) \
  (0x00002003 | ((dst) << 7) | ((base) << 15) | (uint32_t)ENCODE_ITYPE_IMM(imm))
#define CF_SW(src, base, imm) \
  (0x00002023 | ((src) << 20) | ((base) << 15) | \
   (RV_X(imm, 0, 5) << 7) | (RV_X(imm, 5, 7) << 25))
#define CF_SD(src, base, imm) \
  (0x00003023 | ((src) << 20) | ((base) << 15) | \
   (RV_X(imm, 0, 5) << 7) | (RV_X(imm, 5, 7) << 25))

// Read a CSR on the halted hart, returning its low 32 bits (enough for the
// 18-bit 0xbcd status).  csrrs rd, csr, x0 reads without writing.
uint32_t testchip_dtm_t::loadarch_read_csr32(uint32_t regno)
{
  // [fix 2026-08-12] s0 MUST be preserved.  These readbacks run AFTER the
  // checkpoint's GPRs have been restored and immediately before resume(), so a bare
  // `csrrs s0, regno, x0` resumes the target with s0 holding a CSR value instead of
  // the program's data.  Measured: mcf sp_0 dies ~10 instructions after resume for
  // ANY +reconfig_csrs (even a no-op 0xbc1=0 or an unrelated mscratch write), which
  // masqueraded as a reconfiguration bug.  Same dscratch0 bracket that
  // loadarch_restore_csr uses; still 5 program-buffer words.
  uint32_t prog[5] = {
    CSRRx(WRITE, S0, CSR_DSCRATCH0, S0),  // save s0 (swap into dscratch0)
    CSRRx(SET, S0, regno, X0),            // csrrs s0, regno, x0
    CF_SW(S0, X0, get_data_base()),       // low 32b -> DM data buffer
    CSRRx(WRITE, S0, CSR_DSCRATCH0, S0),  // restore s0 (swap back)
    EBREAK
  };
  uint32_t data[2] = {0, 0};
  uint32_t cmd = (AC_ACCESS_REGISTER_POSTEXEC |
                  AC_ACCESS_REGISTER_TRANSFER |
                  AC_ACCESS_REGISTER_WRITE |
                  AC_AR_SIZE(64) |
                  AC_AR_REGNO(0));
  AC_RUN_OR_ABORT(cmd, prog, 5, data, 2);
  return read(DM_DATA0);
}

// [reconf-fix] 64-bit halted CSR read (abstract command, no resume). Mirrors
// loadarch_read_csr32 but uses `sd` and reads both DM data words, so trap CSRs
// (mepc/mtval/mcause with high-half addresses) are not truncated. Assumes the
// hart is already halted; leaves it halted.
reg_t testchip_dtm_t::loadarch_read_csr64(uint32_t regno)
{
  // [fix 2026-08-12] s0 preserved via dscratch0 — see loadarch_read_csr32.
  uint32_t prog[5] = {
    CSRRx(WRITE, S0, CSR_DSCRATCH0, S0),  // save s0
    CSRRx(SET, S0, regno, X0),            // csrrs s0, regno, x0
    CF_SD(S0, X0, get_data_base()),       // 64b -> DM data buffer
    CSRRx(WRITE, S0, CSR_DSCRATCH0, S0),  // restore s0
    EBREAK
  };
  uint32_t data[2] = {0, 0};
  uint32_t cmd = (AC_ACCESS_REGISTER_POSTEXEC |
                  AC_ACCESS_REGISTER_TRANSFER |
                  AC_ACCESS_REGISTER_WRITE |
                  AC_AR_SIZE(64) |
                  AC_AR_REGNO(0));
  AC_RUN_OR_ABORT(cmd, prog, 5, data, 2);
  return ((reg_t)read(DM_DATA0)) | (((reg_t)read(DM_DATA0 + 1)) << 32);
}

void testchip_dtm_t::flush_caches_for_restore(int hartsel)
{
  printf("restore-wipe: hart %d L1 discard walk (csrw 0xbc4, 0x2)\n", hartsel);
  loadarch_restore_csr(CSR_CF_CACHECTL, 0x2);
  // One DMI round-trip >> the <=~256-target-cycle walk, so the first read
  // normally already shows idle; the loop is a safety net, not a spin.
  size_t polls = 0;
  while (loadarch_read_csr32(CSR_CF_DCACHE_STATUS) & (1u << 17)) {
    if (++polls > 1000) { printf("restore-wipe: L1 walk stuck busy\n"); abort(); }
  }
  printf("restore-wipe: L2 directory wipe (0x%x <- 1)\n", CF_L2_WIPE_ADDR);
  {
    uint32_t prog[4] = {
      CF_LUI(S0, 0x2010),               // s0 = 0x02010000
      ADDI(S1, X0, 1),
      CF_SW(S1, S0, 0x100),             // sw 1, 0x100(s0): trigger wipe
      EBREAK
    };
    uint32_t data[2] = {0, 0};
    uint32_t cmd = (AC_ACCESS_REGISTER_POSTEXEC |
                    AC_ACCESS_REGISTER_TRANSFER |
                    AC_ACCESS_REGISTER_WRITE |
                    AC_AR_SIZE(64) |
                    AC_AR_REGNO(0));
    AC_RUN_OR_ABORT(cmd, prog, 4, data, 2);
  }
  polls = 0;
  uint32_t busy = 1;
  do {
    uint32_t prog[4] = {
      CF_LUI(S0, 0x2010),
      CF_LW(S1, S0, 0x100),             // lw s1, 0x100(s0): busy bit
      CF_SW(S1, X0, get_data_base()),
      EBREAK
    };
    uint32_t data[2] = {0, 0};
    uint32_t cmd = (AC_ACCESS_REGISTER_POSTEXEC |
                    AC_ACCESS_REGISTER_TRANSFER |
                    AC_ACCESS_REGISTER_WRITE |
                    AC_AR_SIZE(64) |
                    AC_AR_REGNO(0));
    AC_RUN_OR_ABORT(cmd, prog, 4, data, 2);
    busy = read(DM_DATA0);
    if (++polls > 100000) { printf("restore-wipe: L2 wipe stuck busy\n"); abort(); }
  } while (busy & 1);
  printf("restore-wipe: caches wiped (%zu L2 polls)\n", polls);
}

// fence.i on the halted hart via the program buffer: after a mid-run restore
// the I$ may hold stale lines aliasing the fresh DRAM image; the L2 was wiped,
// so post-fence refills are image-fresh.  Runs after all state restore, right
// before resume (touches no architectural register, and DPC is unaffected by
// program-buffer execution).
void testchip_dtm_t::loadarch_fencei()
{
  uint32_t prog[2] = { 0x0000100f /* fence.i */, EBREAK };
  // Same encoding as dtm_t::fence_i (transfer-to-X0 is the proven path; its
  // own TODO warns transfer=0 is untested) — but unlike dtm_t::fence_i this
  // runs on an already-halted hart and does NOT resume it.
  uint32_t cmd = (AC_ACCESS_REGISTER_POSTEXEC |
                  AC_ACCESS_REGISTER_TRANSFER |
                  AC_ACCESS_REGISTER_WRITE |
                  AC_AR_SIZE(64) |
                  AC_AR_REGNO(0));
  printf("restore-wipe: fence.i before resume\n");
  AC_RUN_OR_ABORT(cmd, prog, 2, 0, 0);
}

void testchip_dtm_t::reset()
{
  testchip_htif_t::perform_init_accesses();
  if (loadarch_file != "") {
    printf("loadarch attempting to load architectural state from %s\n", loadarch_file.c_str());
    std::string line;
    std::ifstream in(loadarch_file);

    if (!in.is_open()) {
      printf("loadarch could not load architectural state file %s\n", loadarch_file.c_str());
      abort();
    }

    std::vector<std::string> lines;
    while (std::getline(in, line)) {
      lines.push_back(line);
    }

    size_t id = 0;
    size_t nharts = std::stoull(lines[id++], nullptr, 0);
    id++; // skip past garbage ':' character

    loadarch_state.resize(nharts);

    for (size_t hartsel = 0; hartsel < nharts; hartsel++) {
      loadarch_state_t &state = loadarch_state[hartsel];
      state.pc       = std::stoull(lines[id++], nullptr, 0);
      state.prv      = read_priv(lines[id++]);

      state.fcsr     = std::stoull(lines[id++], nullptr, 0);

      state.vstart   = std::stoull(lines[id++], nullptr, 0);
      state.vxsat    = std::stoull(lines[id++], nullptr, 0);
      state.vxrm     = std::stoull(lines[id++], nullptr, 0);
      state.vcsr     = std::stoull(lines[id++], nullptr, 0);
      state.vtype    = std::stoull(lines[id++], nullptr, 0);

      state.stvec    = std::stoull(lines[id++], nullptr, 0);
      state.sscratch = std::stoull(lines[id++], nullptr, 0);
      state.sepc     = std::stoull(lines[id++], nullptr, 0);
      state.scause   = std::stoull(lines[id++], nullptr, 0);
      state.stval    = std::stoull(lines[id++], nullptr, 0);
      state.satp     = std::stoull(lines[id++], nullptr, 0);

      state.mstatus  = std::stoull(lines[id++], nullptr, 0);
      state.medeleg  = std::stoull(lines[id++], nullptr, 0);
      state.mideleg  = std::stoull(lines[id++], nullptr, 0);
      state.mie      = std::stoull(lines[id++], nullptr, 0);
      state.mtvec    = std::stoull(lines[id++], nullptr, 0);
      state.mscratch = std::stoull(lines[id++], nullptr, 0);
      state.mepc     = std::stoull(lines[id++], nullptr, 0);
      state.mcause   = std::stoull(lines[id++], nullptr, 0);
      state.mtval    = std::stoull(lines[id++], nullptr, 0);
      state.mip      = std::stoull(lines[id++], nullptr, 0);

      state.mcycle   = std::stoull(lines[id++], nullptr, 0);
      state.minstret = std::stoull(lines[id++], nullptr, 0);

      state.mtime    = std::stoull(lines[id++], nullptr, 0);
      state.mtimecmp = std::stoull(lines[id++], nullptr, 0);

      for (size_t i = 0; i < 32; i++) {
        // Spike prints 128b-wide floats, which this doesn't support
	state.FPR[i] = std::stoull(lines[id++].substr(18), nullptr, 16);
      }
      reg_t regs[32];
      for (size_t i = 0; i < 32; i++) {
	state.XPR[i] = std::stoull(lines[id++], nullptr, 0);
      }

      bool has_vector = false;
      if (lines[id].find("VLEN") != std::string::npos) {
        has_vector = true;
        std::string vlen = lines[id].substr(lines[id].find("VLEN="));
        vlen = vlen.substr(0, vlen.find(" ")).substr(strlen("VLEN="));
        std::string elen = lines[id].substr(lines[id].find("ELEN="));
        elen = elen.substr(0, elen.find(" ")).substr(strlen("ELEN="));

        state.VLEN = std::stoull(vlen, nullptr, 0);
        state.ELEN = std::stoull(elen, nullptr, 0);
        id++;
        if (state.VLEN > MAX_VLEN) {
          printf("Loadarch VLEN %d > %d. Aborting\n", state.VLEN, MAX_VLEN);
          abort();
        }

        for (size_t i = 0; i < 32; i++) {
          state.VPR[i] = (unsigned char*) malloc(state.VLEN / 8);
          std::string elems_s = lines[id].substr(lines[id].find("0x"));
          for (size_t j = state.VLEN / state.ELEN; j-- > 0;) {
            reg_t elem = std::stoull(elems_s.substr(0, elems_s.find(' ')), nullptr, 0);
            if (j > 0)
              elems_s = elems_s.substr(elems_s.find("0x", 2));
            memcpy(state.VPR[i] + j * (state.ELEN / 8), &elem, state.ELEN / 8);
          }
          id++;
        }
      } else {
        id++;
      }

      // mtime goes to CLINT
      write_chunk(0x2000000 + 0xbff8, 8, &state.mtime);
      // mtimecmp goes to CLINT
      write_chunk(0x2000000 + 0x4000 + hartsel * 8, 8, &state.mtimecmp);


      halt(hartsel);

      // [reconf-fix Phase C] wipe L1+L2 while halted, BEFORE restoring state.
      if (restore_cache_wipe) flush_caches_for_restore(hartsel);

      loadarch_restore_csr(CSR_MSTATUS , MSTATUS_FS | MSTATUS_XS | MSTATUS_VS);
      if (state.mstatus & MSTATUS_FS) {
        for (size_t i = 0; i < 32; i++) {
          loadarch_restore_freg(i, state.FPR[i]);
        }
	loadarch_restore_csr(CSR_FCSR     , state.fcsr);
      }
      if (has_vector && state.mstatus & MSTATUS_VS) {
	for (size_t i = 0; i < 32; i++) {
	  loadarch_restore_vreg(i, state.VPR[i], (size_t)state.VLEN / 8);
	}
	loadarch_restore_csr(CSR_VSTART , state.vstart);
	loadarch_restore_csr(CSR_VXSAT  , state.vxsat);
	loadarch_restore_csr(CSR_VXRM   , state.vxrm);
	loadarch_restore_csr(CSR_VCSR   , state.vcsr);
	loadarch_restore_vtype(state.vtype); // vsetvl
      }

      loadarch_restore_csr(CSR_STVEC    , state.stvec);
      loadarch_restore_csr(CSR_SSCRATCH , state.sscratch);
      loadarch_restore_csr(CSR_SEPC     , state.sepc);
      loadarch_restore_csr(CSR_STVAL    , state.stval);
      loadarch_restore_csr(CSR_SATP     , state.satp);
      // [restore-fix 2026-08-09] Counter-enable CSRs. rocket's reg_mcounteren /
      // reg_scounteren are plain Reg() with NO RegInit (CSR.scala:530-537), so nothing
      // initialises them -- on the FPGA they come up zero. CSR.scala:912 gates every
      // counter read on them:
      //     allow_counter = (prv > S || read_mcounteren(addr)) &&
      //                     (!usingSupervisor || prv >= S || read_scounteren(addr))
      // so with them zero EVERY S/U-mode rdcycle/rdtime/rdinstret is an illegal
      // instruction and traps to M-mode. The checkpoint's OS and benchmark both read
      // the timers, so the restored image spends its life in the trap handler:
      // measured 648.exchange2_s/sp_0 at IPC 0.548 with D$ 49.5 MPKI, versus
      // 1.448 / 0.007 MPKI once these are set -- a 2.6x whole-suite slowdown that
      // looks exactly like a cache regression (the handler thrashes both L1s).
      // These are NOT in the 95-line loadarch file; spike checkpoints assume counters
      // are readable, so enable all of them unconditionally. This restores behaviour
      // the pre-2026-08-08 driver had and that was lost in the Phase C rewrite.
      loadarch_restore_csr(CSR_SCOUNTEREN, ~(reg_t)0);
      loadarch_restore_csr(CSR_MCOUNTEREN, ~(reg_t)0);
      loadarch_restore_csr(CSR_MSTATUS  , state.mstatus);
      loadarch_restore_csr(CSR_MEDELEG  , state.medeleg);
      loadarch_restore_csr(CSR_MIDELEG  , state.mideleg);
      loadarch_restore_csr(CSR_MIE      , state.mie);
      loadarch_restore_csr(CSR_MTVEC    , state.mtvec);
      loadarch_restore_csr(CSR_MSCRATCH , state.mscratch);
      loadarch_restore_csr(CSR_MEPC     , state.mepc);
      loadarch_restore_csr(CSR_MCAUSE   , state.mcause);
      loadarch_restore_csr(CSR_MTVAL    , state.mtval);
      loadarch_restore_csr(CSR_MIP      , state.mip);
      loadarch_restore_csr(CSR_MCYCLE   , state.mcycle);
      loadarch_restore_csr(CSR_MINSTRET , state.minstret);

      loadarch_restore_csr(CSR_DCSR     , state.prv);
      loadarch_restore_csr(CSR_DPC      , state.pc);

      for (size_t i = 0; i < 32; i++) {
	loadarch_restore_reg(i, state.XPR[i]);
      }
    }
    assert(id == lines.size());

    printf("loadarch resuming harts\n");
    loadarch_done = true;
    for (size_t hartsel = 0; hartsel < nharts; hartsel++) {
      for (auto& csr : reconfig_csrs) {
        loadarch_restore_csr(csr.first, (reg_t)csr.second);
        // [diag 2026-08-11] The documented contract is `csrw 0xbc4,0x2; fence` --
        // the fence blocks on io.lsu.ordered while dcache_wipe_busy. The DTM emits
        // bare abstract-command writes with NO fence, so poll 0xbcd[17] here to
        // supply that ordering explicitly, and report whether the walk ever ran.
        if (csr.first == CSR_CF_CACHECTL && (((reg_t)csr.second) & 0x2)) {
          uint32_t st0 = loadarch_read_csr32(CSR_CF_DCACHE_STATUS);
          size_t polls = 0;
          while (loadarch_read_csr32(CSR_CF_DCACHE_STATUS) & (1u << 17)) {
            if (++polls > 100000) { printf("reconfig: L1 walk STUCK busy\n"); break; }
          }
          printf("reconfig: csrw 0xbc4=0x%lx -> 0xbcd=0x%08x busy_at_write=%d settled_polls=%zu\n",
                 (unsigned long)csr.second, st0, (st0 >> 17) & 1, polls);
          fflush(stdout);
        }
      }
        // [removed 2026-08-12] Diagnostic 0xbcd/0xbc1/mcycle readbacks deleted.  They ran
        // after the GPRs were restored and, via the then-unbracketed loadarch_read_csr32,
        // clobbered s0 -- corrupting the resumed program for ANY +reconfig_csrs run
        // (even a no-op 0xbc1=0).  loadarch_read_csr32 now preserves s0 via dscratch0;
        // these readbacks are removed regardless: instrumentation, not function.
      // [reconf-fix Phase C] I$ may alias the fresh image after a mid-run
      // restore; fence.i as the last act before resume.
      if (restore_cache_wipe) loadarch_fencei();
      resume(hartsel);
    }
  } else {
    if (!reconfig_csrs.empty()) {
      halt(0);
      for (auto& csr : reconfig_csrs) {
        loadarch_restore_csr(csr.first, (reg_t)csr.second);
      }
      resume(0);
    }
    if (write_hart0_msip) {
      // The dtm_t::reset skips the rest of the bootrom
      // Use CLINT to interrupt the core instead
      //dtm_t::reset();
      uint32_t one = 1;
      write_chunk(MSIP_BASE, sizeof(uint32_t), &one);
    }
  }
}
