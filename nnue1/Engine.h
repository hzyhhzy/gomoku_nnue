#pragma once
#include "global.h"
#include "Evaluator.h"
#include "Search.h"
class Engine
{
public:
  Evaluator* evaluator;
  Search* search;
  Engine(std::string evaluator_type, std::string search_type);
  ~Engine() { delete evaluator; delete search; }


  Color* getBoard() { return evaluator->blackEvaluator->board; }
};
