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

  Color board[BS * BS];


  Evaluator(std::string type, std::string filepath);
  ~Evaluator() {delete blackEvaluator; delete whiteEvaluator;}

  void initZobrist(uint64_t seed);
  bool loadParam(std::string filepathB, std::string filepathW);
  void clear();
  
  
  ValueType evaluateFull(Color color, PolicyType *policy)
  {
    clearCache(color);
    if (color == C_BLACK)
      return blackEvaluator->evaluateFull(policy);
    else
      return whiteEvaluator->evaluateFull(policy);
  }
  void evaluatePolicy(Color color, PolicyType *policy)
  {
    clearCache(color);
    if (color == C_BLACK)
      blackEvaluator->evaluatePolicy(policy);
    else
      whiteEvaluator->evaluatePolicy(policy);
  }
  ValueType evaluateValue(Color color)
  {
    clearCache(color);
    if (color == C_BLACK)
      return blackEvaluator->evaluateValue();
    else
      return whiteEvaluator->evaluateValue();
  }
  void play(Color color, Loc loc); 
  void undo(Loc loc);
  
  //Color* board() const { return blackEvaluator->board; }

private:

  //每次调用play或者undo时，先不在EvaluatorOneSide里面走，因为开销很大。先缓存。
  //MCTS的时候，经常“走回头路”，使用cache可以提速。
  struct MoveCache
  {
    bool isUndo;
    Color color;
    Loc loc;
    MoveCache() :isUndo(false), color(C_EMPTY), loc(LOC_NULL){}
    MoveCache(bool isUndo,Color color,Loc loc) :isUndo(isUndo), color(color), loc(loc){}
  };

  MoveCache moveCacheB[BS * BS], moveCacheW[BS * BS];
  int moveCacheBlength, moveCacheWlength;

  void clearCache(Color color);//把所有缓存的步数清空，使得evaluatorOneSide的board与这里的board相同
  void addCache(bool isUndo, Color color, Loc loc);
  bool isContraryMove(MoveCache a, MoveCache b);//是不是可以抵消的一对操作
};

