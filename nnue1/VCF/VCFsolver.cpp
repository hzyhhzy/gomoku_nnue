#include "VCFsolver.h"
#include <random>
using namespace VCF;
VCFHashTable VCFsolver::hashtable(20, 5);//���Ҫ���̣߳����԰ѵڶ������Ĵ�
VCF::zobristTable VCFsolver::zobrist(1919810);

VCF::zobristTable::zobristTable(int64_t seed)
{
  std::mt19937_64 r(seed);
  r();
  r();
  isWhite = Hash128(r(), r());
  for (int i = 0; i < 3; i++)
    basicRuleHash[i] = Hash128(r(), r());
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < (BS+6)*(BS+6); j++)
    {
      boardhash[i][j] = Hash128(r(), r());
      //std::cout << boardhash[i][j] << std::endl;
    }

}

VCFsolver::VCFsolver(int h, int w, BasicRule basicRule, Color pla)
    : H(h)
    , W(w)
    , basicRule(basicRule)
    , isWhite(pla == C_WHITE)
{
  reset();
}

void VCFsolver::reset()
{

  //hash
  boardHash = Hash128();
  boardHash ^= zobrist.basicRuleHash[basicRule];
  if (isWhite)boardHash ^= zobrist.isWhite;

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
      if (x >= 2 && x < W - 2)shape[0][loc] = 0;//��
      if (y >= 2 && y < H - 2)shape[1][loc] = 0;//��
      if (x >= 2 && x < W - 2 && y >= 2 && y < H - 2)shape[2][loc] = 0;//��б��
      if (x >= 2 && x < W - 2 && y >= 2 && y < H - 2)shape[3][loc] = 0;//��б��
    }

  //other
  ptNum = 0;
}

void VCFsolver::setBoard(const Color* b, bool katagoType, bool colorType)
{
  reset();
  for (int x = 0; x < W; x++)
    for (int y = 0; y < H; y++)
    {
      Loc locSrc = katagoType ? x + 1 + (y + 1) * (W + 1) : x + y * BS;
      Loc locDst = xytoshapeloc(x, y);
      Color c = b[locSrc];
      if (c == C_EMPTY)continue;
      playOutside(locDst, c, 0,colorType);
    }

}


VCF::SearchResult VCFsolver::fullSearch(float factor, int maxLayer,Loc& bestmove, bool katagoType )
{
  if (maxLayer == 0)maxLayer = 10000;
  if (katagoType) {
    std::cout << "Support katago loc in the future";
    return SR_Uncertain;
  }
  bestmove = LOC_NULL;
  //����������������Ƿ���naive�Ľ��
  Loc onlyLoc;
  SearchResult SR_beforeSearch=resetPts(onlyLoc);
  if (SR_beforeSearch == SR_Win||SR_beforeSearch==SR_Lose)
  {
    bestmove = (onlyLoc/(BS+6)-3)*BS+(onlyLoc%(BS+6)-3);
    return SR_beforeSearch;
  }
  for (int search_n =0;search_n<=maxLayer; search_n++)
  {
    nodeNumThisSearch = 0;
    SearchResult sr = search(bound_n(search_n), onlyLoc);
    //std::cout << bound_n(search_n) << " " << nodeNumThisSearch << std::endl;
    if (sr == SR_Win)
    {
      Loc winMove = PV[0];
      bestmove =  (winMove/(BS+6)-3)*BS+(winMove%(BS+6)-3);
      return sr;
    }
    else if (sr == SR_Lose)
    {
      bestmove = LOC_NULL;
      return sr;
    }
    else if (nodeNumThisSearch>factor*searchFactor(search_n))
    {
      bestmove = LOC_NULL;
      return sr;
    }
  }
}

std::vector<Loc> VCFsolver::getPV()
{
  std::vector<Loc> PVvector(PVlen);
  for (int i = 0; i < PVlen; i++)
  {
    Loc oldloc = PV[i];
    Loc loc =  (oldloc/(BS+6)-3)*BS+(oldloc%(BS+6)-3);
    PVvector[i] = loc;
  }
  return PVvector;

}

std::vector<Loc> VCFsolver::getPVreduced()
{
  std::vector<Loc> PVvector((PVlen+1)/2);
  for (int i = 0; i < PVlen; i++)
  {
    if (i % 2 == 1)continue;
    Loc oldloc = PV[i];
    Loc loc =  (oldloc/(BS+6)-3)*BS+(oldloc%(BS+6)-3);
    PVvector[i/2] = loc;
  }
  return PVvector;
}

void VCFsolver::playOutside(Loc loc, Color color, int locType,bool colorType)
{
  if (color == C_EMPTY)return;

  //loc����
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

  //color����
  if (colorType && isWhite)color = getOpp(color);


  //board
  board[loc] = color;

  //hash
  boardHash ^= zobrist.boardhash[color - 1][loc];

  //shape
  int d = (color == C_BLACK) ? 1 : 64;
  bool isSixNotWin = basicRule == BASICRULE_STANDARD || (BASICRULE_RENJU && (isWhite^(color==C_MY)));


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

  //����ʱֻ��������ϴ���壬�����ظ�
  OpPerShape(0, -3 * dir0, 8 * d);
  OpPerShape(1, -3 * dir1, 8 * d);
  OpPerShape(2, -3 * dir2, 8 * d);
  OpPerShape(3, -3 * dir3, 8 * d);

  if (isSixNotWin) {
    OpPerShape(0, 3 * dir0, 8 * d);
    OpPerShape(1, 3 * dir1, 8 * d);
    OpPerShape(2, 3 * dir2, 8 * d);
    OpPerShape(3, 3 * dir3, 8 * d);
  }



#undef OpPerShape
}

void VCFsolver::undoOutside(Loc loc, int locType)
{
  //loc����
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
  movenum = 0;
  PVlen = 0;
  for (int i = 0; i < BS * BS; i++)PV[i] = LOC_NULL;

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
          pts[ptNum] = findEmptyPoint2(loc, d);
          ptNum++;
        }
      }
  if (oppDoubleFour)return SR_Lose;
  return SR_Uncertain;
}

VCF::PT VCFsolver::findEmptyPoint2(Loc loc, int dir)
{
  PT pt;
  pt.shapeDir = dir;
  pt.shapeLoc = loc;
  bool secondEmpty = false;
  Loc bias = dirs[dir];
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
  movenum += 2;
  //�Լ�����------------------------------------------------------------------------------
  board[loc1] = C_MY;
  //hash
  boardHash ^= zobrist.boardhash[C_MY - 1][loc1];
  //����������
#define OpSix(DIR,DIF) shape[DIR][loc1+3*(DIF)]+=8
  OpSix(0, -dir0);
  OpSix(1, -dir1);
  OpSix(2, -dir2);
  OpSix(3, -dir3);

  if (basicRule == BASICRULE_STANDARD
      || (BASICRULE_RENJU && (!isWhite))) {
    OpSix(0, dir0);
    OpSix(1, dir1);
    OpSix(2, dir2);
    OpSix(3, dir3);
  }
#undef OpSix

  //DIR�����ţ�DX�����Ŷ�Ӧ��ָ��ı�����DIS���ƶ�����
#define OpPerShape(DIR,DX,DIS) shape[DIR][loc1+(DIS*DX)]+=1

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

  //��������------------------------------------------------------------------------------

  board[loc2] = C_OPP;
  //hash
  boardHash ^= zobrist.boardhash[C_OPP-1][loc2];
  //����������
#define OpSix(DIR,DIF) shape[DIR][loc2+3*(DIF)]+=8*64
  OpSix(0, -dir0);
  OpSix(1, -dir1);
  OpSix(2, -dir2);
  OpSix(3, -dir3);
  if (basicRule == BASICRULE_STANDARD || (BASICRULE_RENJU && (isWhite))) {
    OpSix(0, dir0);
    OpSix(1, dir1);
    OpSix(2, dir2);
    OpSix(3, dir3);
  }
#undef OpSix

  //DIR�����ţ�DX�����Ŷ�Ӧ��ָ��ı�����DIS���ƶ�����
#define OpPerShape(DIR,DX,DIS) shape[DIR][loc2+(DIS*DX)]+=64

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


  //����Լ�����------------------------------------------------------------------------------

  bool newThree = false;//�Ƿ���������
  bool newTwo = false;//�Ƿ������߶�
  bool isWin = false;//ֻ����Լ��Ƿ�˫��

  //DIR�����ţ�DX�����Ŷ�Ӧ��ָ��ı�����DIS���ƶ�����
#define OpPerShape(DIR,DX,DIS) {\
  Loc loc=loc1+(DIS*DX);\
  int s=shape[DIR][loc];\
  if(shape_isMyFour(s)){\
  isWin = true; \
  PV[movenum]=findEmptyPoint1(loc,DX);\
  }\
  else if(shape_isMyThree(s)){\
      newThree=true;\
      PT pt=findEmptyPoint2(loc,DIR);\
      pts[ptNum]=pt;\
      ptNum++;\
  }\
  else if(shape_isMyTwo(s))newTwo=true;\
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






  //����������------------------------------------------------------------------------------


  bool isLose = false;//ֻ�������Ƿ�˫�ģ����ȼ�����isWin
  nextForceLoc = LOC_NULL;//�ǲ��ǵ�����

#define OpPerShape(DIR,DX,DIS) {\
    Loc loc=loc2+(DIS*DX);\
    int s=shape[DIR][loc];\
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







  isLose = (!isWin) && isLose;

  if (isWin)return PR_Win;
  else if (isLose)return PR_Lose;
  else if (newThree)return PR_OneFourWithThree;
  else if (newTwo)return PR_OneFourWithTwo;
  else return PR_OneFourWithoutTwo;


}

void VCFsolver::undo(Loc loc)
{
  movenum--;
  Color color = board[loc];
  int   d           = (color == C_MY) ? 1 : 64;
  bool  isSixNotWin = basicRule == BASICRULE_STANDARD
                     || (BASICRULE_RENJU && (isWhite ^ (color == C_MY)));

  board[loc] = C_EMPTY;

  //hash
  boardHash ^= zobrist.boardhash[color - 1][loc];

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

  //����ʱֻ��������ϴ���壬�����ظ�
  OpPerShape(0, -3 * dir0, 8 * d);
  OpPerShape(1, -3 * dir1, 8 * d);
  OpPerShape(2, -3 * dir2, 8 * d);
  OpPerShape(3, -3 * dir3, 8 * d);
  
  if (isSixNotWin) {
    OpPerShape(0, 3 * dir0, 8 * d);
    OpPerShape(1, 3 * dir1, 8 * d);
    OpPerShape(2, 3 * dir2, 8 * d);
    OpPerShape(3, 3 * dir3, 8 * d);
  }



#undef OpPerShape
}

VCF::SearchResult VCFsolver::search(int maxNoThree, Loc forceLoc)
{
  //std::cout << nodeNumThisSearch << std::endl;
  //printboard();
  nodeNumThisSearch++;

  //����Ƿ���hash����
  SearchResult hashResult = SearchResult(hashtable.get(boardHash));
  if (hashResult > maxNoThree)//�б����ڵĽ��׼ȷ�Ľ�
  {
    return hashResult;
  }


  int ptNumOld = ptNum;
  SearchResult result = SR_Lose;
  for (int i = 0; i < ptNumOld; i++)
  {
    PT pt = pts[i];
    if (!shape_isMyThree( shape[pt.shapeDir][pt.shapeLoc]))continue;//����������
    for (int j = 0; j < 2; j++)
    {
      Loc loc1 = pt.loc1, loc2 = pt.loc2;
      if (j)loc1 = pt.loc2, loc2 = pt.loc1;
      if (forceLoc != LOC_NULL && forceLoc != loc1)continue;//���Ĳ���

      //���ӣ��ݹ�
      Loc nextForceLoc;
      ptNum = ptNumOld;
      PlayResult  pr = playTwo(loc1, loc2, nextForceLoc);
      if (pr == PR_Win)//ֱ��return
      {
        undo(loc1);
        undo(loc2);
        PV[movenum] = loc1;
        PV[movenum + 1] = loc2;
        PVlen = movenum + 3;
        ptNum = ptNumOld;
        return SR_Win;
      }
      else if (pr == PR_Lose)//�ⲽ������
      {
        undo(loc1);
        undo(loc2);
        continue;
      }
      else//�������
      {
        int decrease = (forceLoc != LOC_NULL) ? 0 :
          (pr == PR_OneFourWithoutTwo) ? NoTwoDecrease :
          (pr == PR_OneFourWithTwo) ? NoThreeDecrease :
          NormalDecrease;

        int newMaxNoThree = maxNoThree - decrease;
        if (newMaxNoThree < 0)
        {
          result = SearchResult(maxNoThree + 1);//˵�������п����ߵģ�ֻ�Ǽ�֦��
          undo(loc1);
          undo(loc2);
          continue;
        }
        SearchResult sr = search(newMaxNoThree, nextForceLoc);//�ݹ�����
        if(sr==SR_Win)//ֱ��return
        {
          undo(loc1);
          undo(loc2);
          PV[movenum] = loc1;
          PV[movenum + 1] = loc2;
          ptNum = ptNumOld;
          return SR_Win;
        }
        else if(sr!=SR_Lose)//��ʱ�޽�
        {
          result = SearchResult(maxNoThree + 1);
        }
        undo(loc1);
        undo(loc2);
      }
    }

  }

  ptNum = ptNumOld;
  hashtable.set(boardHash, result);
  return result;
}

void VCFsolver::printboard()
{
  using namespace std;
  cout << "   ";
  for (int x = 0; x < W; x++)
  {
    cout << char('A' + x)<<" ";
  }
  cout << endl;
  for (int y = 0; y < H; y++)
  {
    int printy = H - y;
    if (printy < 10)cout << " ";
    cout << printy;
    cout << " ";
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

