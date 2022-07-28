#pragma once
#include "Eva_nnuev2.h"
#include "EvaluatorOneSide.h"
#include "VCF/VCFsolver.h"

#include <vector>

class Eva_nnuev2VCF : public Eva_nnuev2
{
public:
  VCFsolver vcfsolver;

  VCF::SearchResult vcfResult;  //如果之前算过VCF就记录在这里
  Loc               vcfWinLoc;  //如果之前算过VCF就记录在这里

  virtual bool loadParam(std::string filepath);
  virtual void clear();
  virtual void recalculate();  //根据board完全重新计算棋形表

  //计算拆分为两部分，第一部分是可增量计算的，放在play函数里。第二部分是不易增量计算的，放在evaluate里。
  virtual void      play(Color color, Loc loc);
  virtual ValueType evaluateFull(PolicyType *policy);    // policy通过函数参数返回
  virtual void      evaluatePolicy(PolicyType *policy);  // policy通过函数参数返回
  virtual ValueType evaluateValue();                     //

  virtual void undo(Loc loc);  // play的逆过程

  // virtual void debug_print();
};
