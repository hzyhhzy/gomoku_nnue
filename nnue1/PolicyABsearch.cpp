#include "PolicyABsearch.h"
const double temp = 2;
double PolicyABsearch::fullsearch(Color color, double factor, Loc& bestmove)
{
  currentDepth = 0;
  double remainPolicy = quantFactor * factor;
  return searchRec(color, remainPolicy, WIN_VALUE, LOSE_VALUE, bestmove);
}

double PolicyABsearch::searchRec(Color color, double remainPolicy, double maxEval, double minEval, Loc& bestmove)
{
  //std::cout << "Debug: remainPolicy=" << remainPolicy << " maxEval=" << maxEval << " minEval=" << minEval << std::endl;//debug
  PolicyType* policyPtr = policyBuf[currentDepth];
  int* policyRankPtr = policyRankBuf[currentDepth];
  double value = evaluator->evaluate(color, policyPtr).winlossrate();
  //if (color == C_BLACK)evaluator->blackEvaluator->debug_print();//debug
  //if (color == C_WHITE)evaluator->whiteEvaluator->debug_print();//debug
  if (remainPolicy<=0)return value;
  sortPolicy(policyPtr, policyRankPtr);
  
  //计算总policy
  PolicyType maxp = policyPtr[policyRankPtr[0]];
  double totalPolicy = 0;
  for (int i = 0; i < 10; i++)totalPolicy += exp((policyPtr[policyRankPtr[i]]/temp - maxp) / quantFactor);
  PolicyType totalp = maxp + log(totalPolicy) * quantFactor;
  //std::cout << totalp << "\n";

  bestmove = NULL_LOC;
  double bestValue = -1e100;
  for (int i = 0; i < BS * BS; i++)
  {
    Loc loc = policyRankPtr[i];
    PolicyType nextRemainPolicy = remainPolicy + policyPtr[policyRankPtr[i]] / temp - totalp;// +quantFactor * (log(maxEval - std::max(minEval, bestValue)) - log(maxEval - minEval));//如果小于0，则这个是最后一个搜索的节点

    //std::cout << nextRemainPolicy << "\n";
    if (nextRemainPolicy < -32&&i!=0)break;
    if (boardPointer[loc] != C_EMPTY) { std::cout << "This nonempty point has a very big policy\n"; continue; }

    double value;
    if (isWin(color, loc))value = WIN_VALUE;
    else
    {
      evaluator->play(color, loc);
      Loc nextBestMove;
      currentDepth += 1;
      value = -searchRec(3 - color, nextRemainPolicy, -std::max(minEval,bestValue), -maxEval, nextBestMove);
      currentDepth -= 1;
      evaluator->undo(loc);
    }
    //std::cout << " Value=" << value;
    if (value > bestValue)
    {
      bestValue = value;
      bestmove = loc;
      if (bestValue >= maxEval)break;
    }
  }
  return bestValue;
}


int PolicyABsearch::explorationFactor(int depth)
{
  return (depth - 1) * 50;//最后一层只走一步
}

int PolicyABsearch::minExploreChildren(int depth)
{
  return depth * 1;
}

bool PolicyABsearch::isWin(Color color, Loc toplayLoc)
{
  const Color* board = (color == C_BLACK) ? evaluator->blackEvaluator->board : evaluator->whiteEvaluator->board;
  int x0 = toplayLoc % BS, y0 = toplayLoc / BS;
  int dirX[4] = { 1,0,1,1 };
  int dirY[4] = { 0,1,1,-1 };
  for (int dir = 0; dir < 4; dir++)
  {
    int x = x0 + dirX[dir], y = y0 + dirY[dir];
    int len = 1;
    while (x >= 0 && x < BS && y >= 0 && y < BS && board[x + BS * y] == C_MY)
    {
      len++;
      x = x + dirX[dir], y = y + dirY[dir];
    }
    x = x0 - dirX[dir], y = y0 - dirY[dir];
    while (x >= 0 && x < BS && y >= 0 && y < BS && board[x + BS * y] == C_MY)
    {
      len++;
      x = x - dirX[dir], y = y - dirY[dir];
    }
    if (len >= 5)return true;
  }
  return false;
}

void PolicyABsearch::sortPolicy(const PolicyType* policy, int* policyRank)//assume policyBuf is ready
{
  std::iota(policyRank, policyRank + BS * BS, 0);
  std::sort(policyRank, policyRank + BS * BS, [&](int a, int b) { return policy[a] > policy[b]; });
}
