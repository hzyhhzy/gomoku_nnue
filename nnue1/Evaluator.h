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
  void recalculate() { blackEvaluator->recalculate(); whiteEvaluator->recalculate(); }
  ValueType evaluate(Color color, PolicyType* policy)
  {
    if (color == C_BLACK)return blackEvaluator->evaluate(policy);
    else return whiteEvaluator->evaluate(policy);
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

