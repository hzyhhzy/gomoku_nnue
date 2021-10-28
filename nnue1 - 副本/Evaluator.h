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
  double evaluate(Color color, PolicyType* policy)
  {
    if (color == C_BLACK)return blackEvaluator->evaluate(policy);
    else return whiteEvaluator->evaluate(policy);
  }
  void play(uint8_t color, uint16_t loc) { blackEvaluator->play(color, loc); whiteEvaluator->play(3 - color, loc); }
  void undo(uint16_t loc) { blackEvaluator->undo(loc); whiteEvaluator->undo(loc); }
  Color* board() const { return blackEvaluator->board; }
};

