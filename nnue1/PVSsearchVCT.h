#pragma once
#include "Search.h"
#include "global.h"
#include "VCF/VCFsolver.h"

#include "PVSsearch.h"
#include <array>
#include <cmath>

// -------------------------------------------------

inline int scoreToVCFlayer(float score)
{
  if (score < 0)return 3;
  if (score > 1000)return 10000;
  return 4 + score / 200;

}
inline float scoreToVCFfactor(float score)
{
  if (score < 0)return 2000;
  if (score > 1000)return 1e20;
  return 2000 * exp(score / 200);

}

// -------------------------------------------------

class PVSsearchVCT : public Search
{
public:
  PVSsearchVCT(Evaluator* e);
  virtual float    fullsearch(Color color, double factor, Loc& bestmove);
  virtual void     stop() { terminate.store(true, std::memory_order_relaxed); }
  std::vector<Loc> rootPV() const;
  void             clear();
  void             setOptions(size_t maxNodes) { option.maxNodes = maxNodes; }
  void             setVCTside(Color side) { option.VCTside = side; }
  size_t           nodes, interiorNodes;
  size_t           ttHits, ttCuts;
  int              selDepth;
  struct Option
  {
    size_t maxNodes;
    Color VCTside;
  } option;
  std::atomic_bool terminate;

private:
  template <bool PV>
  float
    search(Color color, int ply, int depth, int alpha, int beta, bool isCut, Loc& bestmove);
  bool isWin(Color color, Loc toplayLoc);
  void normalizePolicy(const PolicyType* rawPolicy, float* normPolicy) const;
  void sortPolicy(const float* policy, Loc* policyRank) const;
  void copyPV(Loc* pvDst, Loc bestMove, Loc* pvSrc) const;

  struct PlyInfo
  {
    Loc   pv[MAX_PLY];
    int   staticEval;
    int   moveCount;
    int   nullMoveCount;
    Loc   currentMove;
    float currentPolicySum;
  } plyInfos[BS * BS + 1];  // plyInfos[ply]

  VCFsolver vcfSolver[2];
};
