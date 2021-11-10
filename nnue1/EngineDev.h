#pragma once
//功能同engine，只是可以在不删engine代码的同时进行测试
#include "Evaluator.h"
#include "Search.h"
#include "AllSearch.h"
#include "global.h"

class EngineDev
{
public:
  Evaluator *   evaluator;
  Search *   search;
  Color         nextColor;
  std::ofstream logfile;
  EngineDev(std::string evaluator_type, std::string weightfile, int TTsize);
  EngineDev(const EngineDev &e) = delete;
  EngineDev(EngineDev &&e)   = delete;
  ~EngineDev()
  {
    delete evaluator;
    delete search;
    logfile.close();
  }

  int timeout_turn;
  int timeout_match;
  int time_left;

  std::string genmove();

  void protocolLoop();
};
