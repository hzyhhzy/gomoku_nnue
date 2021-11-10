#pragma once
#include "global.h"
#include "Search.h"
class ABsearch
  :public Search
{
public:
  ABsearch(Evaluator* e) :Search(e) {}
  virtual float fullsearch(Color color, double factor, Loc& bestmove);
  virtual void  stop() {}

private:
  float searchRec(Color color, int depth, float maxEval,float minEval,Loc& bestmove);//发现一条分支大于maxeval则直接返回（用于ab剪枝）。minEval传递到下层

  int explorationFactor(int depth);//倒数第depth层，探索policy下降不超过explorationFactor的选点
  int minExploreChildren(int depth);//倒数第depth层，最少搜多少选点
  bool isWin(Color color, Loc toplayLoc);

  int policyRankBuf[BS * BS][BS * BS];//policyRankBuf[depth][loc],避免子节点的policy把父节点覆盖
  PolicyType policyBuf[BS * BS][BS * BS];
  void sortPolicy(const PolicyType* policy,int* policyRank);
};

