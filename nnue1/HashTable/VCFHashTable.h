#pragma once
#include <iostream>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "hash.h"
#include "mutexpool.h"
/*
class MutexPool {
  std::mutex* mutexes;
  uint32_t numMutexes;

public:
  MutexPool(uint32_t n);
  ~MutexPool();

  uint32_t getNumMutexes() const;
  std::mutex& getMutex(uint32_t idx);
};*/

class VCFHashTable {
  struct Entry {
    uint64_t hash1;//HASH��ײ�ĸ��ʼ��ͣ����Ժ���
    int64_t result;
    Entry():hash1(),result(0){}
    ~Entry(){}
  };

  Entry* entries;
  MutexPool* mutexPool;
  uint64_t tableSize;
  uint64_t tableMask;
  uint32_t mutexPoolMask;

public:
  VCFHashTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);
  //VCFHashTable()
  //{
  //  VCFHashTable(20, 13);
  //}
  ~VCFHashTable();

  VCFHashTable(const VCFHashTable& other) = delete;
  VCFHashTable& operator=(const VCFHashTable& other) = delete;

  //These are thread-safe. For get, ret will be set to nullptr upon a failure to find.
  int64_t get(Hash128 hash);
  void set(Hash128 hash, int64_t result);
};
