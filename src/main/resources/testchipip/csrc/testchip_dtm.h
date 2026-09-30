#ifndef __TESTCHIP_DTM_H
#define __TESTCHIP_DTM_H

#include <string>
#include <fesvr/dtm.h>
#include <utility>
#include <vector>
#include <riscv/processor.h>
#include "testchip_htif.h"

// Base class initialized before dtm_t to strip +reconfig_csrs= from argv
// before fesvr's htif_t arg parser treats it as the target binary path.
struct testchip_dtm_arg_filter_t {
  std::vector<char*> filt_argv;
  int filt_argc;
  testchip_dtm_arg_filter_t(int argc, char** argv) : filt_argc(0) {
    for (int i = 0; i < argc; i++)
      if (std::string(argv[i]).find("+reconfig_csrs=") != 0)
        filt_argv.push_back(argv[i]);
    filt_argc = (int)filt_argv.size();
  }
};

struct loadarch_state_t {
  reg_t pc;
  reg_t prv;

  reg_t fcsr;

  reg_t vstart;
  reg_t vxsat;
  reg_t vxrm;
  reg_t vcsr;
  reg_t vtype;

  reg_t stvec;
  reg_t sscratch;
  reg_t sepc;
  reg_t scause;
  reg_t stval;
  reg_t satp;

  reg_t mstatus;
  reg_t medeleg;
  reg_t mideleg;
  reg_t mie;
  reg_t mtvec;
  reg_t mscratch;
  reg_t mepc;
  reg_t mcause;
  reg_t mtval;
  reg_t mip;

  reg_t mcycle;
  reg_t minstret;
  reg_t mtime;
  reg_t mtimecmp;

  reg_t XPR[32];
  reg_t FPR[32];

  reg_t VLEN;
  reg_t ELEN;
  unsigned char* VPR[32];
};

class testchip_dtm_t : private testchip_dtm_arg_filter_t, public dtm_t, public testchip_htif_t
{
 public:
  testchip_dtm_t(int argc, char** argv, bool has_loadmem);
  virtual ~testchip_dtm_t() {};

  void write_chunk(addr_t taddr, size_t nbytes, const void* src) override;
  void read_chunk(addr_t taddr, size_t nbytes, void* dst) override;
  void load_program() {
    is_loadmem = has_loadmem;
    dtm_t::load_program();
    is_loadmem = false;
  };
  void reset() override;

  std::vector<loadarch_state_t> loadarch_state;
  bool loadarch_done;

 protected:
  virtual void load_mem_write(addr_t taddr, size_t nbytes, const void* src) { };
  virtual void load_mem_read(addr_t taddr, size_t nbytes, void* dst) { };
  bool has_loadmem;
  // [reconf-fix] 64-bit halted CSR read (see .cc); protected so subclasses
  // (firesim_dtm_t trap-CSR dump) can use it.
  reg_t loadarch_read_csr64(uint32_t regno);

 private:
  bool is_loadmem;
  std::string loadarch_file;
  std::string reconfig_csrs_file;
  std::vector<std::pair<uint32_t, uint64_t>> reconfig_csrs;
  // [reconf-fix Phase C] +restore-cache-wipe: before restoring architectural
  // state, discard-invalidate L1 D$ (CSR 0xbc4 walk) and wipe the L2
  // directory (L2ctrl+0x100), then fence.i before resume.  Requires the
  // reconf-enabled core (0xbc4/0xbcd exist only there) and an L2 built with
  // the wipe register.  RESTORE-ONLY: drops all dirty cached state.
  bool restore_cache_wipe;
  void flush_caches_for_restore(int hartsel);
  void loadarch_fencei();
  uint32_t loadarch_read_csr32(uint32_t regno);

  void loadarch_restore_csr(uint32_t regno, reg_t reg);
  void loadarch_restore_reg(uint32_t regno, reg_t reg);
  void loadarch_restore_freg(uint32_t regno, reg_t reg);
  void loadarch_restore_vreg(uint32_t regno, unsigned char* reg, size_t bytes);
  void loadarch_restore_vtype(uint32_t vtype);
};

#endif
