#pragma once
#include "global.h"
#include "Evaluator.h"
class Search
{
public:
  Evaluator* evaluator;//在engine里析构这个evaluator，不在这里析构
  const Color* boardPointer;
  std::atomic_bool terminate;
  Search(Evaluator* e):evaluator(e),boardPointer(e->blackEvaluator->board){}
  virtual float fullsearch(Color color, double factor, Loc& bestmove) = 0;//factor是搜索因数（可以指代层数，节点数等），越大则搜索量越大
  virtual void play(Color color, Loc loc) { evaluator->play(color, loc); }
  virtual void undo( Loc loc) { evaluator->undo(loc); }
  virtual void     stop() { terminate.store(true, std::memory_order_relaxed); }
};
