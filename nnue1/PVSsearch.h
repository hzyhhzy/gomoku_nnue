#pragma once
#include "Search.h"
#include "global.h"

#include <array>
#include <cmath>

// -------------------------------------------------

constexpr int   MAX_PLY               = 128;
constexpr int   VALUE_MATE            = 30000;
constexpr int   VALUE_MATE_IN_MAX_PLY = 30000 - 256;
constexpr int   VALUE_MAX_EVAL        = 10000;
constexpr int   VALUE_DRAW            = 0;
constexpr float WLR_MATE_VALUE        = 0.99;
constexpr float WLR_VALUE_K           = 1 / (2 * 0.0025);
constexpr int   mateValue(int ply)
{
    return VALUE_MATE - ply;
}

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

inline std::string valueText(int value)
{
    if (value >= VALUE_MATE_IN_MAX_PLY)
        return "+M" + std::to_string(VALUE_MATE - value);
    else if (value <= -VALUE_MATE_IN_MAX_PLY)
        return "-M" + std::to_string(-VALUE_MATE + value);
    else
        return std::to_string(value);
}

// -------------------------------------------------

constexpr int MARGIN_INFINITE       = INT16_MAX;
constexpr int RAZOR_DEPTH_REDUCTION = 2;
constexpr int LMR_DEPTH             = 3;

constexpr int razorMargin(int d)
{
    return d < 6 ? std::max(80 + 60 * d, 0) : MARGIN_INFINITE;
}

constexpr int razorVerifyMargin(int d)
{
    return razorMargin(d - 2.0f);
}

constexpr int futilityMargin(int d)
{
    return std::max(60 * d, 0);
}

template <bool PV>
constexpr int futilityMoveCount(int d)
{
    return 2 + int(d * d) * 2 / (3 - PV);
}

inline double trivialPolicyResidual(int depth)
{
    return double(0.25 * std::exp(-depth * 0.25));
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
    return int(Reductions[d] * Reductions[moveCount]);
}

// -------------------------------------------------

class PVSsearch : public Search
{
public:
    PVSsearch(Evaluator *e);
    virtual float fullsearch(Color color, double factor, Loc &bestmove);
    size_t        nodes, interiorNodes;

private:
    template <bool PV>
    float search(Color color, int ply, int depth, int alpha, int beta, bool isCut, Loc &bestmove);
    bool  isWin(Color color, Loc toplayLoc);
    void  normalizePolicy(const PolicyType *rawPolicy, float *normPolicy);
    void  sortPolicy(const float *policy, Loc *policyRank);

    struct PlyInfo
    {
        int   staticEval;
        int   moveCount;
        Loc   currentMove;
        float currentPolicySum;
    } plyInfos[BS * BS];  // plyInfos[ply]
};
