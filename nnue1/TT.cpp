#include "TT.h"

#include <cstring>

HashTable TT;

static unsigned long upper_power_of_two(unsigned long v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

void TTEntry::save(Key k, int v, Bound b, int d, bool isPv, Loc l)
{
  if (l != NULL_LOC)
    best = l;
  else if (k != key)
    best = NULL_LOC;

  if (b == BOUND_EXACT || (uint16_t)k != key || d + 3 > depth) {
    key   = (uint16_t)k;
    value = (int16_t)v;
    depth = (int8_t)d;
    bound = b;
    gen   = TT.gen;
    pv    = isPv;
  }
}

void HashTable::resize(size_t sizeMb)
{
  if (tbl)
    delete[] tbl;

  size = sizeMb * 1024 * (1024 / sizeof(TTEntry));
  size = upper_power_of_two(size);
  tbl  = new TTEntry[size];

  clear();
}

void HashTable::clear()
{
  std::memset(tbl, 0, sizeof(TTEntry) * size);
  for (size_t i = 0; i < size; i++) {
    tbl[i].best = NULL_LOC;
  }
  gen = 0;
}

std::pair<TTEntry *, bool> HashTable::probe(Key key) const
{
  TTEntry *tte = &tbl[(key >> 32) & (size - 1)];
  return std::make_pair(tte, tte->key == (uint16_t)key);
}