#include "NNUEHashTable.h"
#include "../MCTSsearch.h"

using namespace std;

NNUEHashTable::NNUEHashTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo)
{
  tableSize              = ((uint64_t)1) << sizePowerOfTwo;
  tableMask              = tableSize - 1;
  entries                = new Entry[tableSize];
  uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  mutexPoolMask          = mutexPoolSize - 1;
  mutexPool              = new MutexPool(mutexPoolSize);
}
NNUEHashTable::~NNUEHashTable()
{
  delete[] entries;
  delete mutexPool;
}

bool NNUEHashTable::get(Hash128 hash, MCTSnode &node)
{
  // Free ret BEFORE locking, to avoid any expensive operations while locked.

  uint64_t idx      = hash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry &  entry    = entries[idx];
#ifndef FORGOMOCUP
  std::mutex &mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);
#endif
  if (entry.hash1 == hash.hash1) {
    node.sureResult       = entry.sureResult;
    node.legalChildrennum = entry.legalChildrennum;
    for (int child = 0; child < MAX_MCTS_CHILDREN; child++) {
      node.children[child].loc = entry.locs[child];
      node.children[child].policy = entry.policy[child];
    }
    return true;
  }
  else return false;
}

void NNUEHashTable::set(Hash128 hash, const MCTSnode &node)
{
  // Immediately copy p right now, before locking, to avoid any expensive operations while
  // locked.

  uint64_t idx = hash.hash0 & tableMask;

#ifndef FORGOMOCUP
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
#endif
  Entry &entry = entries[idx];
#ifndef FORGOMOCUP
  std::mutex &mutex = mutexPool->getMutex(mutexIdx);
#endif

  {
#ifndef FORGOMOCUP
    std::lock_guard<std::mutex> lock(mutex);
#endif
    // Perform a swap, to avoid any expensive free under the mutex.
    entry.hash1           = hash.hash1;
    entry.sureResult       = node.sureResult;
    entry.legalChildrennum = node.legalChildrennum;
    for (int child = 0; child < MAX_MCTS_CHILDREN; child++) {
      entry.locs[child] = node.children[child].loc;
      entry.policy[child] = node.children[child].policy;
    }
  }

  // No longer locked, allow buf to fall out of scope now, will free whatever used to be
  // present in the table.
}
