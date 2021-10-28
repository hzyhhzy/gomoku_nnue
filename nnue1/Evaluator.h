#pragma once
#include "global.h"
#include "EvaluatorOneSide.h"
class Evaluator
{
public:
  EvaluatorOneSide* blackEvaluator;
  EvaluatorOneSide* whiteEvaluator;

  Evaluator(std::string type, std::string filepath);
  ~Evaluator() {delete blackEvaluator; delete whiteEvaluator;}


  bool loadParam(std::string filepathB, std::string filepathW);
  void clear() {blackEvaluator->clear();whiteEvaluator->clear();}
  void recalculate() { blackEvaluator->recalculate(); whiteEvaluator->recalculate(); }
  ValueType evaluate(Color color, PolicyType* policy)
  {
    if (color == C_BLACK)return blackEvaluator->evaluate(policy);
    else return whiteEvaluator->evaluate(policy);
  }
  void play(Color color, Loc loc) { blackEvaluator->play(color, loc); whiteEvaluator->play(3 - color, loc); }
  void undo(Loc loc) { blackEvaluator->undo(loc); whiteEvaluator->undo(loc); }
  Color* board() const { return blackEvaluator->board; }
};

