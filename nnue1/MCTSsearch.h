#pragma once
#include "Search.h"
#include "VCF/VCFsolver.h"
const int    MAX_MCTS_CHILDREN = 50;
const double policyQuant       = 50000;
const double policyQuantInv    = 1 / policyQuant;

enum MCTSsureResult : int16_t { MC_Win = 1, MC_LOSE = -1, MC_DRAW = 2, MC_UNCERTAIN = 0 };
inline double sureResultWR(MCTSsureResult sr)
{
  if (sr == MC_Win)
    return 1;
  else if (sr == MC_LOSE)
    return -1;
  else
    return 0;
}

inline double
MCTSpuctFactor(double totalVisit, double puct, double puctPow, double puctBase)
{
  return puct * pow((totalVisit + puctBase) / puctBase, puctPow);
}

inline double
MCTSselectionValue(double puctFactor, double value, double childVisit, double childPolicy)
{
  return value + puctFactor * childPolicy / (childVisit + 1);
}

struct MCTSnode;

struct int128_t
{
  int64_t data[2];
};

struct MCTSchild
{
  int32_t  offset;
  Loc      loc;
  uint16_t policy;  // 除以policyQuant是原始policy

  MCTSnode *ptr() const
  {
    if (offset)
      return reinterpret_cast<MCTSnode *>(base_ptr() + offset);
    else
      return nullptr;
  }
  void setPtr(MCTSnode *ptr)
  {
    // 对于空指针，offset直接存0
    if (ptr == nullptr) {
      offset = 0;
      return;
    }

    // 由于MCTSnode已经是16byte对齐的，该转换没问题
    int128_t *target = reinterpret_cast<int128_t *>(ptr);
    ptrdiff_t diff   = target - base_ptr();
    offset           = (int32_t)diff;
    // 由于内存是连续分配的，两次分配的距离超过32位offset表示范围:
    // 2^32 * sizeof(int128_t) = 64GB 的情况不可能发生
    // 但最好还是检测一下
    if (offset != diff) {
      std::cerr << "You are very unlucky!!" << std::endl;
      std::terminate();
    }
  }

private:
  // 获得对齐到int64_t的基准指针
  int128_t *base_ptr() const
  {
    intptr_t this_ptr = reinterpret_cast<intptr_t>(this);
    return reinterpret_cast<int128_t *>(this_ptr & (~15));
  }
};

struct alignas(int128_t) MCTSnode
{
  MCTSsureResult sureResult;
  int16_t        childrennum;
  int16_t        legalChildrennum;  //=min(合法招法，MAX_MCTS_CHILDREN)
  Color          nextColor;
  uint64_t       visits;  //包括自己
  double WRtotal;  //以下一次落子的这一方的视角,1胜-1负, 平均胜率=WRtotal/visits
  MCTSchild children[MAX_MCTS_CHILDREN];

  MCTSnode(Evaluator * evaluator,
           Color       nextColor,
           double      policyTemp,
           Loc *       locbuf,
           PolicyType *pbuf1,
           PolicyType *pbuf2,
           float *     pbuf3);
  MCTSnode(MCTSsureResult sureResult, Color nextColor);
  ~MCTSnode();
};

const int a = sizeof(MCTSnode);

class MCTSsearch : public Search
{
public:
  MCTSnode *rootNode;

  struct Option
  {
    size_t maxNodes = 0;
  } option;
  struct Param
  {
    double expandFactor = 0.4;
    double puct         = 0.6;
    double puctPow      = 0.7;  //传统的mcts对应0.5
    double puctBase     = 1.0;
    double fpuReduction = 0.1;
    double policyTemp   = 1.1;

  } params;

  VCFsolver vcfSolver[2];

  Loc        locbuf[BS * BS];
  PolicyType pbuf1[BS * BS], pbuf2[MAX_MCTS_CHILDREN];
  float      pbuf3[MAX_MCTS_CHILDREN];

  MCTSsearch(Evaluator *e);
  virtual float fullsearch(Color color, double factor, Loc &bestmove);
  virtual void  play(Color color, Loc loc);
  virtual void  undo(Loc loc);
  virtual void  clearBoard();

  Loc     bestRootMove() const;
  float   getRootValue() const;
  int64_t getRootVisit() const;

  void setOptions(size_t maxNodes) { option.maxNodes = maxNodes; }
  void loadParamFile(std::string filename);
  ~MCTSsearch()
  {
    if (rootNode != NULL)
      delete rootNode;
  }

private:
  struct SearchResult
  {
    uint64_t newVisits;
    double   WRchange;
  };
  void playForSearch(Color color, Loc loc);
  void undoForSearch(Loc loc);

  SearchResult   search(MCTSnode *node, uint64_t remainVisits, bool isRoot);
  MCTSsureResult checkSureResult(Loc nextMove, Color color);  // check VCF
  int            selectChildIDToSearch(MCTSnode *node);
};
