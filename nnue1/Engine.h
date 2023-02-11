#pragma once
//功能同engine，只是可以在不删engine代码的同时进行测试
#include "MCTSsearch.h"
#include "Evaluator.h"
#include "Search.h"
#include "NNUEglobal.h"

class Engine
{
public:
  Evaluator *   evaluator;
  MCTSsearch *      search;
  Color         nextColor;
  const bool    writeLogEnable;
  std::ofstream logfile;
  Engine(std::string evaluator_type, std::string weightfile, std::string configfile,bool writeLogEnable);
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


  static constexpr int ReservedTime          = 30;
  static constexpr int AsyncWaitReservedTime = 70;

  std::string genmove();

  void        writeLog(std::string str);

  void protocolLoop();
};
