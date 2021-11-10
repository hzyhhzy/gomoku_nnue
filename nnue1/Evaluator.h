#pragma once
#include "global.h"
#include "EvaluatorOneSide.h"
class Evaluator
{
public:
  EvaluatorOneSide* blackEvaluator;
  EvaluatorOneSide* whiteEvaluator;
  Key               zobrist[2][BS * BS];
  Key               key;

  Evaluator(std::string type, std::string filepath);
  ~Evaluator() {delete blackEvaluator; delete whiteEvaluator;}

  void initZobrist(uint64_t seed);
  bool loadParam(std::string filepathB, std::string filepathW);
  void clear() {blackEvaluator->clear();whiteEvaluator->clear();}
  void recalculate()
  {
    blackEvaluator->recalculate();
    whiteEvaluator->recalculate();
  }
  ValueType evaluateFull(Color color, PolicyType *policy)
  {
    if (color == C_BLACK)
      return blackEvaluator->evaluateFull(policy);
    else
      return whiteEvaluator->evaluateFull(policy);
  }
  void evaluatePolicy(Color color, PolicyType *policy)
  {
    if (color == C_BLACK)
      blackEvaluator->evaluatePolicy(policy);
    else
      whiteEvaluator->evaluatePolicy(policy);
  }
  ValueType evaluateValue(Color color)
  {
    if (color == C_BLACK)
      return blackEvaluator->evaluateValue();
    else
      return whiteEvaluator->evaluateValue();
  }
  void play(Color color, Loc loc) { 
      key ^= zobrist[color - C_BLACK][loc];
      blackEvaluator->play(color, loc); 
      whiteEvaluator->play(~color, loc); 
  }
  void undo(Color color, Loc loc)
  {
      key ^= zobrist[color - C_BLACK][loc];
      blackEvaluator->undo(loc); 
      whiteEvaluator->undo(loc); 
  }
  Color* board() const { return blackEvaluator->board; }
};

