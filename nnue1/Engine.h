#pragma once
#include "Evaluator.h"
#include "PVSsearch.h"
#include "Search.h"
#include "global.h"
class Engine
{
public:
  Evaluator *   evaluator;
  PVSsearch *   search;
  Color         nextColor;
  std::ofstream logfile;
  Engine(std::string evaluator_type, std::string weightfile, int TTsize);
  Engine(const Engine &e) = delete;
  Engine(Engine &&e)      = delete;
  ~Engine()
  {
    delete evaluator;
    delete search;
    logfile.close();
  }

  int timeout_turn;
  int timeout_match;
  int time_left;

  static constexpr int ReservedTime          = 50;
  static constexpr int AsyncWaitReservedTime = 100;

  std::string genmove();

  void protocolLoop();
};

typedef int64_t Time;  // value in milliseconds
Time            now();
