// See LICENSE for license details.

#ifndef MM_EMULATOR_H
#define MM_EMULATOR_H

#include <stdint.h>
#include <cstring>
#include <queue>
#include <cassert>
#include <fesvr/memif.h>

struct backing_data_t
{
  uint8_t* data;
  size_t size;
};

class mm_t
{
 public:
  mm_t(size_t mem_bs, size_t mem_sz, size_t word_sz, size_t line_sz, backing_data_t& dat) :
    data(dat.data), mem_base(mem_bs), mem_size(mem_sz), word_size(word_sz), line_size(line_sz) {
    assert(dat.size == mem_sz);
  }

  virtual bool ar_ready() = 0;
  virtual bool aw_ready() = 0;
  virtual bool w_ready() = 0;
  virtual bool b_valid() = 0;
  virtual uint64_t b_resp() = 0;
  virtual uint64_t b_id() = 0;
  virtual bool r_valid() = 0;
  virtual uint64_t r_resp() = 0;
  virtual uint64_t r_id() = 0;
  virtual void *r_data() = 0;
  virtual bool r_last() = 0;

  virtual void tick
  (
    bool reset,

    bool ar_valid,
    uint64_t ar_addr,
    uint64_t ar_id,
    uint64_t ar_size,
    uint64_t ar_len,

    bool aw_valid,
    uint64_t aw_addr,
    uint64_t aw_id,
    uint64_t aw_size,
    uint64_t aw_len,

    bool w_valid,
    uint64_t w_strb,
    void *w_data,
    bool w_last,

    bool r_ready,
    bool b_ready
  ) = 0;

  virtual void* get_data() { return data; }
  virtual size_t get_size() { return mem_size; }
  virtual size_t get_base() { return mem_base; }
  virtual size_t get_word_size() { return word_size; }
  virtual size_t get_line_size() { return line_size; }

  void write(uint64_t addr, uint8_t *data, uint64_t strb, uint64_t size);
  std::vector<char> read(uint64_t addr);

  virtual ~mm_t();
  uint8_t* data;

 protected:
  size_t mem_base;
  size_t mem_size;
  int word_size;
  int line_size;
};

struct mm_rresp_t
{
  uint64_t id;
  std::vector<char> data;
  bool last;

  uint64_t ready;   // [MMJIT] first tick count at which this beat may be presented

  mm_rresp_t(uint64_t id, std::vector<char> data, bool last, uint64_t ready = 0)
  {
    this->id = id;
    this->data = data;
    this->last = last;
    this->ready = ready;
  }

  mm_rresp_t()
  {
    this->id = 0;
    this->last = false;
    this->ready = 0;
  }
};

class mm_magic_t : public mm_t
{
 public:
  mm_magic_t(size_t mem_base, size_t mem_sz, size_t word_sz, size_t line_sz, backing_data_t& dat) :
    mm_t(mem_base, mem_sz, word_sz, line_sz, dat), store_inflight(false), cycle(0),
    jitter_max(0), rng(0x9E3779B97F4A7C15ULL) {}

  // [MMJIT] seeded response-latency jitter (harness only, no RTL).  Each read burst and each
  // write ack is held for an extra U[0, jmax] ticks, drawn from a per-channel xorshift64*
  // seeded by `seed`.  Responses stay in FIFO order (head-of-line), so AXI ordering holds.
  // jmax = 0 (the default) gives exactly the stock timing.
  void set_jitter(uint64_t jmax, uint64_t seed) {
    jitter_max = jmax;
    rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
  }

  virtual bool ar_ready() { return true; }
  virtual bool aw_ready() { return !store_inflight; }
  virtual bool w_ready() { return store_inflight; }
  virtual bool b_valid() { return !bresp.empty() && cycle > bresp_ready.front(); }
  virtual uint64_t b_resp() { return 0; }
  virtual uint64_t b_id() { return b_valid() ? bresp.front() : 0; }
  virtual bool r_valid() { return !rresp.empty() && cycle > rresp.front().ready; }
  virtual uint64_t r_resp() { return 0; }
  virtual uint64_t r_id() { return r_valid() ? rresp.front().id: 0; }
  virtual void *r_data() { return r_valid() ? &rresp.front().data[0] : (void*) data; }
  virtual bool r_last() { return r_valid() ? rresp.front().last : false; }

  virtual void tick
  (
    bool reset,

    bool ar_valid,
    uint64_t ar_addr,
    uint64_t ar_id,
    uint64_t ar_size,
    uint64_t ar_len,

    bool aw_valid,
    uint64_t aw_addr,
    uint64_t aw_id,
    uint64_t aw_size,
    uint64_t aw_len,

    bool w_valid,
    uint64_t w_strb,
    void *w_data,
    bool w_last,

    bool r_ready,
    bool b_ready
  );

 protected:
  bool store_inflight;
  uint64_t store_addr;
  uint64_t store_id;
  uint64_t store_size;
  uint64_t store_count;
  std::queue<uint64_t> bresp;
  std::queue<uint64_t> bresp_ready;   // [MMJIT] parallel to bresp

  std::queue<mm_rresp_t> rresp;

  uint64_t cycle;

  // [MMJIT]
  uint64_t jitter_max;
  uint64_t rng;
  uint64_t draw_jitter() {
    if (jitter_max == 0) return 0;
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return ((rng * 0x2545F4914F6CDD1DULL) >> 32) % (jitter_max + 1);
  }
};

#endif
