#pragma once
#include "Search.h"
#include "global.h"
#include "VCF/VCFsolver.h"

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

constexpr int   MAX_PLY               = 128;
constexpr int   VALUE_NONE            = -30001;
constexpr int   VALUE_MATE            = 30000;
constexpr int   VALUE_MATE_IN_MAX_PLY = 30000 - MAX_PLY;
constexpr int   VALUE_MAX_EVAL        = 10000;
constexpr int   VALUE_DRAW            = 0;
constexpr float WLR_MATE_VALUE        = 0.999;
constexpr float WLR_VALUE_K           = 1 / (2 * 0.0025);
constexpr int   mateValue(int ply) { return VALUE_MATE - ply; }

constexpr int valueFromWLR(float wlr, int ply = 0)
{
  if (wlr > WLR_MATE_VALUE)
    return +VALUE_MAX_EVAL;  // mateValue(ply);
  else if (wlr < -WLR_MATE_VALUE)
    return -VALUE_MAX_EVAL;  //-mateValue(ply);
  else
    return std::clamp(int(WLR_VALUE_K * std::log((1 + wlr) / (1 - wlr))),
                      -VALUE_MAX_EVAL,
                      VALUE_MAX_EVAL);
}

constexpr int valueFromTT(int ttValue, int ply)
{
  return ttValue >= VALUE_MATE_IN_MAX_PLY    ? ttValue - ply
         : ttValue <= -VALUE_MATE_IN_MAX_PLY ? ttValue + ply
                                             : ttValue;
}

constexpr int valueToTT(int value, int ply)
{
  return value >= VALUE_MATE_IN_MAX_PLY    ? value + ply
         : value <= -VALUE_MATE_IN_MAX_PLY ? value - ply
                                           : value;
}

inline std::string valueText(int value)
{
  if (value >= VALUE_MATE_IN_MAX_PLY)
    return "+M" + std::to_string(VALUE_MATE - value);
  else if (value <= -VALUE_MATE_IN_MAX_PLY)
    return "-M" + std::to_string(VALUE_MATE + value);
  else
    return std::to_string(value);
}

// -------------------------------------------------

constexpr int MARGIN_INFINITE       = INT16_MAX;
constexpr int RAZOR_DEPTH_REDUCTION = 2;
constexpr int IID_DEPTH             = 9;
constexpr int IID_REDUCTION         = 7;
constexpr int LMR_DEPTH             = 3;

constexpr int razorMargin(int d)
{
  return d < 6 ? std::max(80 + 60 * d, 0) : MARGIN_INFINITE;
}

constexpr int razorVerifyMargin(int d) { return razorMargin(d - 2.0f); }

constexpr int futilityMargin(int d) { return std::max(60 * d, 0); }

constexpr int nullMoveMargin(int d)
{
  return d >= 8 ? 250 + std::max(20 * (d - 8), 0) : MARGIN_INFINITE;
}

constexpr int nullMoveReduction(int d) { return d / 3 + 2; }

template <bool PV>
constexpr int futilityMoveCount(int d)
{
  return 2 + int(d * d) * 2 / (3 - PV);
}

inline double trivialPolicyResidual(int depth)
{
  return double(0.1 * std::exp(-depth * 0.125));
}

template <bool PV>
constexpr int lateMoveCount(int d)
{
  return 1 + 2 * PV + int(d * 3 / 2);
}

inline const auto Reductions = []() {
  std::array<float, BS * BS + 1> R {0};
  for (size_t i = 1; i < R.size(); i++)
    R[i] = float(std::log(i) * 0.72);  // magic number
  return R;
}();

constexpr int lmrReduction(int d, int moveCount)
{
  return int(0.5f + Reductions[d] * Reductions[moveCount]);
}

// -------------------------------------------------

class PVSsearch : public Search
{
public:
  PVSsearch(Evaluator *e);
  virtual float    fullsearch(Color color, double factor, Loc &bestmove);
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
    Option() :maxNodes(UINT64_MAX), VCTside(DEFAULT_VCT_SIDE) {}
  } option;
  std::atomic_bool terminate;

  VCFsolver vcfSolver[2];
private:
  template <bool PV>
  float
  search(Color color, int ply, int depth, int alpha, int beta, bool isCut, Loc &bestmove);
  bool isWin(Color color, Loc toplayLoc);
  void normalizePolicy(const PolicyType *rawPolicy, float *normPolicy) const;
  void sortPolicy(const float *policy, Loc *policyRank) const;
  void copyPV(Loc *pvDst, Loc bestMove, Loc *pvSrc) const;

  struct PlyInfo
  {
    Loc   pv[MAX_PLY];
    int   staticEval;
    int   moveCount;
    int   nullMoveCount;
    Loc   currentMove;
    float currentPolicySum;
  } plyInfos[BS * BS + 1];  // plyInfos[ply]

};
