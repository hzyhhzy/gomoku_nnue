#pragma once

#include "global.h"

enum Bound : int8_t {
  BOUND_NONE,
  BOUND_UPPER,
  BOUND_LOWER,
  BOUND_EXACT = BOUND_UPPER | BOUND_LOWER
};

struct alignas(int64_t) TTEntry
{
  uint16_t key;
  int16_t  value;
  Loc      best;
  int8_t   depth;
  Bound    bound : 2;
  int8_t   gen : 5;
  bool     pv : 1;

  void save(Key k, int v, Bound b, int d, bool pv, Loc l);
};
static_assert(sizeof(TTEntry) == 8);

class HashTable
{
  friend struct TTEntry;
  TTEntry *tbl  = nullptr;
  size_t   size = 0;
  int8_t   gen  = 0;

public:
  void                       resize(size_t sizeMb);
  void                       clear();
  void                       incGen() { gen = (gen + 1) % 32; }
  std::pair<TTEntry *, bool> probe(Key key) const;
};

extern HashTable TT;