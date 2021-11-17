#include "VCFsolver.h"
using namespace VCF;
VCFsolver::VCFsolver(int h, int w, Color pla) :H(h), W(w), isWhite(pla == C_WHITE)
{
  reset();
}

void VCFsolver::reset()
{
  //board
  for (int i = 0; i < ArrSize; i++)
    board[i] = C_WALL;
  for (int x = 0; x < W; x++)
    for (int y = 0; y < H; y++)
    {
      Loc loc = xytoshapeloc(x, y);
      board[loc] = 0;
    }
  //shape
  for (int j = 0; j < 4; j++)
    for (int i = 0; i < ArrSize; i++)
      shape[j][i] = 4096;

  for (int x = 0; x < W; x++)
    for (int y = 0; y < H; y++)
    {
      Loc loc = xytoshapeloc(x, y);
      if (x >= 2 && x < W - 2)shape[0][loc] = 0;//横
      if (y >= 2 && y < H - 2)shape[1][loc] = 0;//竖
      if (x >= 2 && x < W - 2 && y >= 2 && y < H - 2)shape[2][loc] = 0;//正斜线
      if (x >= 2 && x < W - 2 && y >= 2 && y < H - 2)shape[3][loc] = 0;//反斜线
    }

  //other
  ptNum = 0;
}

void VCFsolver::setBoard(Color* b, bool katagoType)
{
  for (int x = 0; x < W; x++)
    for (int y = 0; y < H; y++)
    {
      Loc locSrc = katagoType ? x + 1 + (y + 1) * (W + 1) : x + y * BS;
      Loc locDst = xytoshapeloc(x, y);
      Color c = b[locSrc];
      if (c == C_EMPTY)continue;
      playOutside(locDst, c, 0);
    }

}


VCF::SearchResult VCFsolver::fullSearch(float factor, Loc& bestmove, bool katagoType )
{
  nodeNumThisSearch = 0;
  if (katagoType) {
    std::cout << "Support katago loc in the future";
    return SR_Uncertain;
  }
  bestmove = LOC_NULL;
  //制作活三表，并检查是否有naive的结果
  Loc onlyLoc;
  SearchResult SR_beforeSearch=resetPts(onlyLoc);
  if (SR_beforeSearch == SR_Win||SR_beforeSearch==SR_Lose)
  {
    bestmove = (onlyLoc/(BS+6)-3)*BS+(onlyLoc%(BS+6)-3);
    return SR_beforeSearch;
  }
  for (int maxNoThree =0;; maxNoThree++)
  {
    Loc winMove = LOC_NULL;
    SearchResult sr = search(maxNoThree, onlyLoc, winMove);
    std::cout << maxNoThree << " " << nodeNumThisSearch << std::endl;
    if (sr == SR_Win)
    {
      bestmove =  (winMove/(BS+6)-3)*BS+(winMove%(BS+6)-3);
      return sr;
    }
    else if (sr == SR_Lose)
    {
      bestmove = LOC_NULL;
      return sr;
    }
    else if (nodeNumThisSearch>factor)
    {
      bestmove = LOC_NULL;
      return sr;
    }
  }
}

void VCFsolver::playOutside(Loc loc, Color color, int locType)
{
  if (color == C_EMPTY)return;
  //loc换算
  if (locType == 1)
  {
    int x = loc % BS, y = loc / BS;
    loc = xytoshapeloc(x, y);
  }
  else if (locType == 2)
  {
    int x = (loc % (W + 1)) - 1, y = (loc / (W + 1)) - 1;
    loc = xytoshapeloc(x, y);
  }

  int d = (isWhite ^ (color == C_BLACK)) ? 1 : 64;
  board[loc] = color;

#define OpPerShape(DIR,DIF,INC) shape[DIR][loc+(DIF)]+=INC
  OpPerShape(0, -2 * dir0, d);
  OpPerShape(0, -1 * dir0, d);
  OpPerShape(0, 0, d);
  OpPerShape(0, 1 * dir0, d);
  OpPerShape(0, 2 * dir0, d);
  OpPerShape(1, -2 * dir1, d);
  OpPerShape(1, -1 * dir1, d);
  OpPerShape(1, 0, d);
  OpPerShape(1, 1 * dir1, d);
  OpPerShape(1, 2 * dir1, d);
  OpPerShape(2, -2 * dir2, d);
  OpPerShape(2, -1 * dir2, d);
  OpPerShape(2, 0, d);
  OpPerShape(2, 1 * dir2, d);
  OpPerShape(2, 2 * dir2, d);
  OpPerShape(3, -2 * dir3, d);
  OpPerShape(3, -1 * dir3, d);
  OpPerShape(3, 0, d);
  OpPerShape(3, 1 * dir3, d);
  OpPerShape(3, 2 * dir3, d);

  //长连时只采用坐标较大的五，避免重复
  OpPerShape(0, -3 * dir0, 8 * d);
  OpPerShape(1, -3 * dir1, 8 * d);
  OpPerShape(2, -3 * dir2, 8 * d);
  OpPerShape(3, -3 * dir3, 8 * d);

  //六不胜时解除底下的注释
  //OpPerShape(0, 3*dir0,8*d);
  //OpPerShape(1, 3*dir1,8*d);
  //OpPerShape(2, 3*dir2,8*d);
  //OpPerShape(3, 3*dir3,8*d);



#undef OpPerShape
}

void VCFsolver::undoOutside(Loc loc, int locType)
{
  //loc换算
  if (locType == 1)
  {
    int x = loc % BS, y = loc / BS;
    loc = xytoshapeloc(x, y);
  }
  else if (locType == 2)
  {
    int x = (loc % (W + 1)) - 1, y = (loc / (W + 1)) - 1;
    loc = xytoshapeloc(x, y);
  }
  undo(loc);
}

SearchResult VCFsolver::resetPts(Loc& onlyLoc)
{
  ptNum = 0;
  onlyLoc = LOC_NULL;
  bool oppDoubleFour = false;
  for (int d = 0; d < 4; d++)
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++)
      {
        Loc loc = xytoshapeloc(x, y);
        int16_t s = shape[d][loc];
        if (shape_isMyFive(s))
        {
          std::cout << "Error: already my five";
          return SR_Win;
        }
        else if (shape_isOppFive(s))
        {
          std::cout << "Error: already opp five";
          return SR_Lose;
        }
        else if (shape_isMyFour(s))
        {
          onlyLoc = findEmptyPoint1(loc, dirs[d]);
          return SR_Win;
        }
        else if (shape_isOppFour(s))
        {
          if (onlyLoc == LOC_NULL)onlyLoc = findEmptyPoint1(loc, dirs[d]);
          else if (onlyLoc != findEmptyPoint1(loc, dirs[d]))oppDoubleFour = true;
        }
        else if (shape_isMyThree(s))
        {
          pts[ptNum] = findEmptyPoint2(loc, dirs[d]);
          ptNum++;
        }
      }
  if (oppDoubleFour)return SR_Lose;
  return SR_Uncertain;
}

VCF::PT VCFsolver::findEmptyPoint2(Loc loc, Loc bias)
{
  PT pt;
  bool secondEmpty = false;
  loc = loc - 2 * bias;
  for (int i = 0; i < 5; i++)
  {
    if (board[loc] == C_EMPTY)
    {
      if (secondEmpty)
      {
        pt.loc2 = loc;
        return pt;
      }
      pt.loc1 = loc;
      secondEmpty = true;
    }
    loc += bias;
  }
  std::cout << "ERROR no two empty points";
  return pt;
}

Loc VCFsolver::findEmptyPoint1(Loc loc, Loc bias)
{
  loc = loc - 2 * bias;
  for (int i = 0; i < 5; i++)
  {
    if (board[loc] == C_EMPTY)
    {
      return loc;
    }
    loc += bias;
  }
  std::cout << "ERROR no one empty points";
  return LOC_NULL;
}

VCF::PlayResult VCFsolver::playTwo(Loc loc1, Loc loc2, Loc& nextForceLoc)
{
  bool newThree = false;
  bool isWin = false;//只检测自己是否双四
  bool isLose = false;//只检测对手是否双四，优先级低于isWin
  nextForceLoc = LOC_NULL;

  //自己落子------------------------------------------------------------------------------
  board[loc1] = C_MY;
  //处理长连问题
#define OpSix(DIR,DIF) shape[DIR][loc1+3*(DIF)]+=8
  OpSix(0, -dir0);
  OpSix(1, -dir1);
  OpSix(2, -dir2);
  OpSix(3, -dir3);
  //六不胜去掉下面注释
  //OpSix(0, dir0);
  //OpSix(1, dir1);
  //OpSix(2, dir2);
  //OpSix(3, dir3);
#undef OpSix

  //DIR方向编号，DX方向编号对应的指针改变量，DIS是移动距离
#define OpPerShape(DIR,DX,DIS) {\
  Loc loc=loc1+(DIS*DX);\
  int s=shape[DIR][loc];\
  s+=1;\
  shape[DIR][loc]=s;\
  if(shape_isMyFour(s)){\
    Loc defendLoc=findEmptyPoint1(loc,DX);\
    if(defendLoc!=loc2)isWin=true;\
  }\
  else if(shape_isMyThree(s)){\
    PT pt=findEmptyPoint2(loc,DX);\
    if(pt.loc1!=loc2&&pt.loc2!=loc2){\
      newThree=true;\
      pts[ptNum]=pt;\
      ptNum++;\
    }\
  }\
}

  OpPerShape(0, dir0, -2);
  OpPerShape(0, dir0, -1);
  OpPerShape(0, dir0, 0);
  OpPerShape(0, dir0, 1);
  OpPerShape(0, dir0, 2);
  OpPerShape(1, dir1, -2);
  OpPerShape(1, dir1, -1);
  OpPerShape(1, dir1, 0);
  OpPerShape(1, dir1, 1);
  OpPerShape(1, dir1, 2);
  OpPerShape(2, dir2, -2);
  OpPerShape(2, dir2, -1);
  OpPerShape(2, dir2, 0);
  OpPerShape(2, dir2, 1);
  OpPerShape(2, dir2, 2);
  OpPerShape(3, dir3, -2);
  OpPerShape(3, dir3, -1);
  OpPerShape(3, dir3, 0);
  OpPerShape(3, dir3, 1);
  OpPerShape(3, dir3, 2);

#undef OpPerShape
  

  //对手落子------------------------------------------------------------------------------

  board[loc2] = C_OPP;
  //处理长连问题
#define OpSix(DIR,DIF) shape[DIR][loc2+3*(DIF)]+=8*64
  OpSix(0, -dir0);
  OpSix(1, -dir1);
  OpSix(2, -dir2);
  OpSix(3, -dir3);
  //六不胜去掉下面注释
  //OpSix(0, dir0);
  //OpSix(1, dir1);
  //OpSix(2, dir2);
  //OpSix(3, dir3);
#undef OpSix

  //DIR方向编号，DX方向编号对应的指针改变量，DIS是移动距离
#define OpPerShape(DIR,DX,DIS) {\
  Loc loc=loc2+(DIS*DX);\
  int s=shape[DIR][loc];\
  s+=64;\
  shape[DIR][loc]=s;\
  if(shape_isOppFour(s)){\
    if(nextForceLoc==LOC_NULL){\
      nextForceLoc=findEmptyPoint1(loc,DX);\
    }\
    else{\
      if(findEmptyPoint1(loc,DX)!=nextForceLoc)isLose=true;\
    }\
  }\
}

  OpPerShape(0, dir0, -2);
  OpPerShape(0, dir0, -1);
  OpPerShape(0, dir0, 0);
  OpPerShape(0, dir0, 1);
  OpPerShape(0, dir0, 2);
  OpPerShape(1, dir1, -2);
  OpPerShape(1, dir1, -1);
  OpPerShape(1, dir1, 0);
  OpPerShape(1, dir1, 1);
  OpPerShape(1, dir1, 2);
  OpPerShape(2, dir2, -2);
  OpPerShape(2, dir2, -1);
  OpPerShape(2, dir2, 0);
  OpPerShape(2, dir2, 1);
  OpPerShape(2, dir2, 2);
  OpPerShape(3, dir3, -2);
  OpPerShape(3, dir3, -1);
  OpPerShape(3, dir3, 0);
  OpPerShape(3, dir3, 1);
  OpPerShape(3, dir3, 2);

#undef OpPerShape

  isLose = (!isWin) && isLose;

  if (isWin)return PR_Win;
  else if (isLose)return PR_Lose;
  else if (newThree)return PR_OneFourWithThree;
  else return PR_OneFourWithoutThree;


}

void VCFsolver::undo(Loc loc)
{
  Color color = board[loc];
  int d = (color == C_MY) ? 1 : 64;
  board[loc] = C_EMPTY;

#define OpPerShape(DIR,DIF,INC) shape[DIR][loc+(DIF)]-=INC
  OpPerShape(0, -2 * dir0, d);
  OpPerShape(0, -1 * dir0, d);
  OpPerShape(0, 0, d);
  OpPerShape(0, 1 * dir0, d);
  OpPerShape(0, 2 * dir0, d);
  OpPerShape(1, -2 * dir1, d);
  OpPerShape(1, -1 * dir1, d);
  OpPerShape(1, 0, d);
  OpPerShape(1, 1 * dir1, d);
  OpPerShape(1, 2 * dir1, d);
  OpPerShape(2, -2 * dir2, d);
  OpPerShape(2, -1 * dir2, d);
  OpPerShape(2, 0, d);
  OpPerShape(2, 1 * dir2, d);
  OpPerShape(2, 2 * dir2, d);
  OpPerShape(3, -2 * dir3, d);
  OpPerShape(3, -1 * dir3, d);
  OpPerShape(3, 0, d);
  OpPerShape(3, 1 * dir3, d);
  OpPerShape(3, 2 * dir3, d);

  //长连时只采用坐标较大的五，避免重复
  OpPerShape(0, -3 * dir0, 8 * d);
  OpPerShape(1, -3 * dir1, 8 * d);
  OpPerShape(2, -3 * dir2, 8 * d);
  OpPerShape(3, -3 * dir3, 8 * d);

  //六不胜时解除底下的注释
  //OpPerShape(0, 3*dir0,8*d);
  //OpPerShape(1, 3*dir1,8*d);
  //OpPerShape(2, 3*dir2,8*d);
  //OpPerShape(3, 3*dir3,8*d);



#undef OpPerShape
}

VCF::SearchResult VCFsolver::search(int maxNoThree, Loc forceLoc, Loc& winLoc)
{
  //std::cout << nodeNumThisSearch << std::endl;
  //printboard();
  nodeNumThisSearch++;
  int ptNumOld = ptNum;
  SearchResult result = SR_Lose;
  for (int i = 0; i < ptNumOld; i++)
  {
    PT pt = pts[i];
    if (board[pt.loc1] != C_EMPTY || board[pt.loc2] != C_EMPTY)continue;//眠三作废了
    for (int j = 0; j < 2; j++)
    {
      Loc loc1 = pt.loc1, loc2 = pt.loc2;
      if (j)loc1 = pt.loc2, loc2 = pt.loc1;
      if (forceLoc != LOC_NULL && forceLoc != loc1)continue;//冲四不挡

      //落子，递归
      Loc nextForceLoc;
      ptNum = ptNumOld;
      PlayResult  pr = playTwo(loc1, loc2, nextForceLoc);
      if (pr == PR_Win)//直接return
      {
        undo(loc1);
        undo(loc2);
        winLoc = loc1;
        ptNum = ptNumOld;
        return SR_Win;
      }
      else if (pr == PR_Lose)//这步不能走
      {
        undo(loc1);
        undo(loc2);
        continue;
      }
      else//正常情况
      {
        bool noThree = (pr == PR_OneFourWithoutThree) && (forceLoc == LOC_NULL);
        int newMaxNoThree = noThree ? maxNoThree - 1 : maxNoThree;
        if (newMaxNoThree < 0)
        {
          result = SearchResult(1);//说明还是有可以走的，只是剪枝了
          undo(loc1);
          undo(loc2);
          continue;
        }
        Loc nextWinLoc;//无用
        SearchResult sr = search(newMaxNoThree, nextForceLoc, nextWinLoc);//递归搜索
        if(sr==SR_Win)//直接return
        {
          undo(loc1);
          undo(loc2);
          winLoc = loc1;
          ptNum = ptNumOld;
          return SR_Win;
        }
        else if(sr!=SR_Lose)//暂时无解
        {
          result = SearchResult(maxNoThree + 1);
        }
        undo(loc1);
        undo(loc2);
      }
    }

  }

  ptNum = ptNumOld;
  return result;
}

void VCFsolver::printboard()
{
  using namespace std;
  for (int y = 0; y < H; y++)
  {
    for (int x = 0; x < W; x++)
    {
      Color c = board[xytoshapeloc(x,y)];
      if (c == C_EMPTY)cout << ". ";
      else if (c == C_MY)cout << "x ";
      else if (c == C_OPP)cout << "o ";
    }
    cout << endl;
  }
  cout << endl;
}
