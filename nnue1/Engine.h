#pragma once
#include "global.h"
#include "Evaluator.h"
#include "Search.h"
#include "PVSsearch.h"
class Engine
{
public:
  Evaluator *evaluator;
  PVSsearch *search;
  Color      nextColor;
  Engine(std::string evaluator_type, std::string weightfile,int TTsize);
  Engine(const Engine &e) = delete;
  Engine(Engine &&e) = delete;
  ~Engine() { delete evaluator; delete search; }

  void protocolLoop();
};
