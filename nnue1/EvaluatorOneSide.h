#pragma once
#include "global.h"
class EvaluatorOneSide
{
public:
  Color mySide;//这个估值器是哪边的。无论上一手是谁走的，都返回下一手为mySide的估值
  Color board[BS * BS];
  virtual bool loadParam(std::ifstream& fs) = 0;
  virtual void clear() = 0;
  virtual void recalculate() = 0;//根据board完全重新计算棋形表
  
  //计算拆分为两部分，第一部分是可增量计算的，放在play函数里。第二部分是不易增量计算的，放在evaluate里。
  virtual void play(Color color, Loc loc) = 0;
  virtual ValueType evaluate(PolicyType* policy) = 0;//policy通过函数参数返回

  virtual void undo(Loc loc) = 0;//play的逆过程
  virtual void debug_print() {}//debug用
};


