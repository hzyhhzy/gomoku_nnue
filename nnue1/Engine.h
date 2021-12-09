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
  const bool    writeLogEnable;
  std::ofstream logfile;
  Engine(std::string evaluator_type,
    std::string weightfileB,
    std::string weightfileW,
    int         TTsize,
    bool        writeLogEnable);
  Engine(std::string evaluator_type,
    std::string weightfile,
    int         TTsize,
    bool        writeLogEnable):Engine(evaluator_type,weightfile,weightfile,TTsize,writeLogEnable){}
  Engine(const Engine &e) = delete;
  Engine(Engine &&e)      = delete;
  ~Engine()
  {
    delete evaluator;
    delete search;
    logfile.close();
  }

  int64_t  timeout_turn;
  int64_t  timeout_match;
  int64_t  time_left;
  uint64_t max_nodes;
  int      max_depth;

  static constexpr int ReservedTime          = 30;
  static constexpr int AsyncWaitReservedTime = 70;

  void        writeLog(std::string str);
  std::string genmove();

  void protocolLoop();
};
