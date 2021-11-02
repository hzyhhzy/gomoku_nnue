#include "PVSsearch.h"

PVSsearch::PVSsearch(Evaluator *e) : Search(e), nodes(0), interiorNodes(0) {}

float PVSsearch::fullsearch(Color color, double factor, Loc &bestmove)
{
    int depth = (int)factor;
    return search<true>(color, 0, depth, -VALUE_MATE, VALUE_MATE, false, bestmove);
}

template <bool PV>
float PVSsearch::search(Color me,
                        int   ply,
                        int   depth,
                        int   alpha,
                        int   beta,
                        bool  isCut,
                        Loc & bestmove)
{
    const bool  Root  = ply == 0;
    const Color oppo  = 3 - me;
    int         value = valueFromWLR(evaluator->evaluate(me, nullptr).winlossrate(), ply);
    nodes++;

    // 叶子条件: 水平线 或者 估值杀
    if (depth <= 0 || ply >= MAX_PLY || std::abs(value) >= VALUE_MATE_IN_MAX_PLY) {
        return value;
    }

    // TODO: 满盘和棋

    // 剪枝: Mate distance pruning
    alpha = std::max(-mateValue(ply), alpha);
    beta  = std::min(mateValue(ply + 1), beta);
    if (alpha >= beta)
        return alpha;

    // TODO: 置换表查找

    // Static eval
    int eval = plyInfos[ply].staticEval = value;

    // 剪枝: Razoring
    if (!PV && eval + razorMargin(depth) <= alpha) {
        // TODO: do some VCF search to verify!
        int lowAlpha = alpha - razorVerifyMargin(depth);
        value        = search<false>(me,
                              ply + 1,
                              depth - RAZOR_DEPTH_REDUCTION,
                              lowAlpha,
                              lowAlpha + 1,
                              !isCut,
                              bestmove);

        if (value <= alpha)
            return value;
    }

    // 剪枝: Futility pruning
    if (!PV && /*对方无冲四*/ true && eval - futilityMargin(depth) >= beta)
        return eval;

expand_node:
    interiorNodes++;
    PolicyType rawPolicy[BS * BS];
    float      policy[BS * BS];
    Loc        policyRank[BS * BS];
    evaluator->evaluate(me, rawPolicy);

    normalizePolicy(rawPolicy, policy);
    sortPolicy(policy, policyRank);
    float maxPolicy = policy[policyRank[0]];
    float policySum = 0;

    bestmove      = NULL_LOC;
    int moveCount = 0;
    int bestValue = -VALUE_MATE;
    for (int i = 0; i < BS * BS; i++) {
        Loc move = policyRank[i];
        if (boardPointer[move] != C_EMPTY)
            continue;

        plyInfos[ply].currentMove      = move;
        plyInfos[ply].currentPolicySum = (policySum += policy[move]);
        plyInfos[ply].moveCount        = ++moveCount;

        if (isWin(me, move)) {
            value = mateValue(ply);
        }
        else {
            if (!Root && bestValue > -VALUE_MATE_IN_MAX_PLY) {
                // PRUNING: Move count based pruning
                if (moveCount >= futilityMoveCount<PV>(depth))
                    continue;

                // PRUNING: trivial policy pruning
                if (1 - policySum >= trivialPolicyResidual(depth))
                    continue;
            }

            Loc  nextBestMove    = NULL_LOC;
            int  newDepth        = depth - 1;
            bool fullDepthSearch = !PV || moveCount > 1;

            evaluator->play(me, move);

            // Do LMR
            if (depth >= LMR_DEPTH && moveCount > 1
                && (moveCount >= lateMoveCount<PV>(depth) || isCut)) {
                int r = lmrReduction(depth, moveCount) + 2 * isCut;
                r += (plyInfos[ply].currentPolicySum > 0.95)
                     + (plyInfos[ply].currentPolicySum > 0.99);
                r -= (ply > 1 && plyInfos[ply - 1].currentPolicySum < 0.1);

                int d = std::clamp(newDepth - r, 1, newDepth);
                value = -search<false>(oppo, ply + 1, d, -(alpha + 1), -alpha, true, nextBestMove);
                fullDepthSearch = value > alpha && d < newDepth;
            }

            // Do full depth non-pv search
            if (fullDepthSearch)
                value = -search<false>(oppo,
                                       ply + 1,
                                       newDepth,
                                       -(alpha + 1),
                                       -alpha,
                                       !isCut,
                                       nextBestMove);

            // Do full PV search
            if (PV && (moveCount == 1 || value > alpha && (Root || value < beta)))
                value = -search<true>(oppo, ply + 1, newDepth, -beta, -alpha, false, nextBestMove);

            evaluator->undo(move);
        }

        if (value > bestValue) {
            bestValue = value;
            if (value > alpha) {
                bestmove = move;
                if (!PV || value >= beta)
                    break;
                else
                    alpha = value;
            }
        }
    }
    return bestValue;
}

template float PVSsearch::search<true>(Color color,
                                       int   ply,
                                       int   depth,
                                       int   alpha,
                                       int   beta,
                                       bool  isCut,
                                       Loc & bestmove);
template float PVSsearch::search<false>(Color color,
                                        int   ply,
                                        int   depth,
                                        int   alpha,
                                        int   beta,
                                        bool  isCut,
                                        Loc & bestmove);

bool PVSsearch::isWin(Color color, Loc toplayLoc)
{
    const Color *board =
        (color == C_BLACK) ? evaluator->blackEvaluator->board : evaluator->whiteEvaluator->board;
    int x0 = toplayLoc % BS, y0 = toplayLoc / BS;
    int dirX[4] = {1, 0, 1, 1};
    int dirY[4] = {0, 1, 1, -1};
    for (int dir = 0; dir < 4; dir++) {
        int x = x0 + dirX[dir], y = y0 + dirY[dir];
        int len = 1;
        while (x >= 0 && x < BS && y >= 0 && y < BS && board[x + BS * y] == C_MY) {
            len++;
            x = x + dirX[dir], y = y + dirY[dir];
        }
        x = x0 - dirX[dir], y = y0 - dirY[dir];
        while (x >= 0 && x < BS && y >= 0 && y < BS && board[x + BS * y] == C_MY) {
            len++;
            x = x - dirX[dir], y = y - dirY[dir];
        }
        if (len >= 5)
            return true;
    }
    return false;
}

void PVSsearch::normalizePolicy(const PolicyType *rawPolicy, float *normPolicy)
{
    PolicyType maxRawPolicy = *std::max_element(rawPolicy, rawPolicy + BS * BS);
    std::transform(rawPolicy, rawPolicy + BS * BS, normPolicy, [=](auto &p) {
        return (float)std::exp(p - maxRawPolicy);
    });
    float policySum = std::reduce(normPolicy, normPolicy + BS * BS);
    float k         = 1 / policySum;
    std::transform(normPolicy, normPolicy + BS * BS, normPolicy, [=](auto &p) { return p * k; });
}

void PVSsearch::sortPolicy(const float *policy, Loc *policyRank)  // assume policyBuf is ready
{
    std::iota(policyRank, policyRank + BS * BS, 0);
    std::sort(policyRank, policyRank + BS * BS, [&](Loc a, Loc b) {
        return policy[a] > policy[b];
    });
}
