#pragma once
#include "Evaluator.h"
#include "VCF/VCFsolver.h"
#include "ExtraStates.h"
const double policyQuant = 50000;
const double policyQuantInv = 1/policyQuant;

inline ValueSum sureResultWR(MCTSsureResult sr)
{
  if (sr == MC_Win)
    return ValueSum(1, 0, 0);
  else if (sr == MC_LOSE)
    return ValueSum(0, 1, 0);
  else if (sr == MC_DRAW)
    return ValueSum(0, 0, 1);
  else
    return ValueSum(-1e100, -1e100, -1e100);
}

inline double MCTSpuctFactor(double totalVisit, double puct, double puctPow, double puctBase)
{
  return  puct * pow((totalVisit+puctBase)/puctBase, puctPow);
}

inline double MCTSselectionValue(double puctFactor,
                                 double value,
                                 double draw,
                                 double   parentdraw,
                                 double   childVisit,
                                 double   childPolicy)
{
  return value - 0.5 * draw * (1-parentdraw) + puctFactor * childPolicy / (childVisit + 1);
}

struct MCTSnode;
class MCTSsearch;

struct MCTSchild
{
  MCTSnode* ptr;
  Loc loc;
  uint16_t policy;//除以policyQuant是原始policy
};

struct MCTSnode
{
  MCTSsureResult sureResult;
  int16_t childrennum;
  int16_t legalChildrennum;//=min(合法招法，MAX_MCTS_CHILDREN)
  MCTSchild children[MAX_MCTS_CHILDREN];
  

  uint64_t visits;//包括自己
  ValueSum WRtotal;//以下一次落子的这一方的视角,1胜-1负
  //平均胜率=WRtotal/visits

  Color nextColor;
  
  MCTSnode(MCTSsearch* search,  Color nextColor,double policyTemp);
  MCTSnode(MCTSsureResult sureResult, Color nextColor);
  ~MCTSnode();
  
};

class MCTSsearch 
{
public:


  static NNUEHashTable hashTable;




  MCTSnode   *rootNode;
  Color       board[MaxBS * MaxBS];
  Hash128 posHash;// only board
  ExtraStates states;

  Evaluator *evaluator;  //在engine里析构这个evaluator，不在这里析构
  VCFsolver vcfSolver[2];


  std::atomic_bool terminate;

  struct Option
  {
    size_t maxNodes=0;
  } option;
  struct Param
  {
    double expandFactor = 0.2;//传统的mcts对应0.0, 也就是每次playout固定新增1个叶子节点
    double puct = 2;
    double puctPow = 0.75;//传统的mcts对应0.5
    double puctBase = 10;
    double fpuReduction = 0.1;
    double policyTemp = 1.1;

  }params;



  MCTSsearch(Evaluator *e);
  float    fullsearch(Color color, double factor, Loc &bestmove);
  void play(Color color, Loc loc);
  void undo( Loc loc);
  void clearBoard(); 

  Loc bestRootMove() const;
  float getRootValue() const;
  int64_t getRootVisit() const;
  void stop() { terminate.store(true, std::memory_order_relaxed); }

  void             setOptions(size_t maxNodes) { option.maxNodes = maxNodes; }
  void loadParamFile(std::string filename);
  ~MCTSsearch() { if(rootNode!=NULL)delete rootNode; }

private:

  Loc        locbuf[MaxBS * MaxBS];
  PolicyType pbuf1[MaxBS * MaxBS], pbuf2[MAX_MCTS_CHILDREN];
  float      pbuf3[MAX_MCTS_CHILDREN];
  float      gfbuf[NNUEV2::globalFeatureNum];

  friend MCTSnode::MCTSnode(MCTSsearch *search,
                            Color       nextColor,
                            double      policyTemp);
  struct SearchResult
  {
    uint64_t newVisits;
    ValueSum WRchange;
  };
  void playForSearch(Color color, Loc loc);
  void undoForSearch(Loc loc);

  SearchResult search(MCTSnode* node, uint64_t remainVisits,bool isRoot);
  MCTSsureResult checkSureResult(Loc nextMove, Color color);//check VCF
  int            selectChildIDToSearch(MCTSnode *node);
  void      getGlobalFeatureInput(Color nextPlayer);
  Hash128            getStateHash(Color nextPlayer);



};

