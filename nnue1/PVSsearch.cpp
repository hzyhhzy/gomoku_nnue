#include "PVSsearch.h"

#include "TT.h"

PVSsearch::PVSsearch(Evaluator *e)
    : Search(e)
    , nodes(0)
    , interiorNodes(0)
    , ttHits(0)
    , ttCuts(0)
{}

float PVSsearch::fullsearch(Color color, double factor, Loc &bestmove)
{
  int depth                 = (int)factor;
  selDepth                  = 0;
  plyInfos[0].nullMoveCount = 0;
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
  const Color oppo  = getOpp(me);
  int         value = valueFromWLR(evaluator->evaluateValue(me).winlossrate(), ply);

  plyInfos[ply].pv[0] = bestmove = LOC_NULL;
  nodes++;

  if (ply > selDepth)
    selDepth = ply;

  // 叶子节点估值
  if (depth <= 0 || ply >= MAX_PLY || std::abs(value) >= VALUE_MATE_IN_MAX_PLY) {
    return value;
  }

  // 剪枝: Mate distance pruning
  alpha = std::max(-mateValue(ply), alpha);
  beta  = std::min(mateValue(ply + 1), beta);
  if (alpha >= beta)
    return alpha;

  if (nodes > option.maxNodes)
    terminate.store(true, std::memory_order_relaxed);

  // 置换表查找
  auto [tte, ttHit] = TT.probe(evaluator->key);
  int  ttValue      = ttHit ? valueFromTT(tte->value, ply) : 0;
  Loc  ttMove       = ttHit ? tte->best : LOC_NULL;
  bool ttPv         = ttHit ? tte->pv : false;
  ttHits += ttHit;

  // 剪枝: 置换表剪枝
  if (!PV && ttHit && tte->depth >= depth
      && (ttValue >= beta ? (tte->bound & BOUND_LOWER) : (tte->bound & BOUND_UPPER))) {
    ttCuts++;
    return ttValue;
  }

  // 静态估值
  int eval = plyInfos[ply].staticEval = value;
  int evalDelta = ply >= 2 ? plyInfos[ply].staticEval - plyInfos[ply - 2].staticEval : 0;

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
  if (!PV && true && eval - futilityMargin(depth) >= beta)
    return eval;

  // 剪枝: Null move pruning
  if (!PV && true && plyInfos[ply].nullMoveCount == 0
      && eval - nullMoveMargin(depth) >= beta) {
    int r                           = nullMoveReduction(depth);
    plyInfos[ply].currentMove       = LOC_NULL;
    plyInfos[ply].currentPolicySum  = 0;
    plyInfos[ply].moveCount         = 0;
    plyInfos[ply + 1].nullMoveCount = 1;

    Loc nextBest;
    value = search<false>(oppo, ply + 1, -beta, -(beta - 1), depth - r, !isCut, nextBest);

    if (value >= beta) {
      value = std::min(beta, VALUE_MATE_IN_MAX_PLY);
      if (beta < VALUE_MATE_IN_MAX_PLY)
        return value;
      else {
        value = search<false>(me, ply, beta - 1, beta, depth - r, false, bestmove);
        if (value >= beta)
          return value;
      }
    }
  }

expand_node:
  // IID
  if (depth >= IID_DEPTH && ttMove == LOC_NULL) {
    search<PV>(me, ply, depth - IID_REDUCTION, alpha, beta, isCut, ttMove);
  }

  interiorNodes++;
  plyInfos[ply + 1].nullMoveCount = plyInfos[ply].nullMoveCount;
  PolicyType rawPolicy[BS * BS];
  float      policy[BS * BS];
  Loc        policyRank[BS * BS];
  float      maxPolicy = 0;
  float      policySum = 0;

  auto calcPolicy = [&]() {
    evaluator->evaluatePolicy(me, rawPolicy);

    normalizePolicy(rawPolicy, policy);
    sortPolicy(policy, policyRank);
    maxPolicy = policy[policyRank[0]];
  };

  bestmove      = LOC_NULL;
  int moveCount = 0;
  int bestValue = -VALUE_MATE;
  for (int i = -1; i < BS * BS; i++) {
    Loc move;
    if (i == -1) {
      if (ttMove == LOC_NULL || boardPointer[ttMove] != C_EMPTY)
        continue;
      move = ttMove;
    }
    else {
      if (i == 0)
        calcPolicy();
      move = policyRank[i];
      if (move == ttMove || boardPointer[move] != C_EMPTY)
        continue;
    }

    plyInfos[ply].currentMove      = move;
    plyInfos[ply].currentPolicySum = (policySum += policy[move]);
    plyInfos[ply].moveCount        = ++moveCount;

    if (isWin(me, move)) {
      value                   = mateValue(ply);
      plyInfos[ply + 1].pv[0] = LOC_NULL;
    }
    else {
      if (!Root && bestValue > -VALUE_MATE_IN_MAX_PLY) {
        // 剪枝: Move count based pruning
        if (moveCount >= futilityMoveCount<PV>(depth))
          continue;

        // 剪枝: trivial policy pruning
        if (1 - policySum < trivialPolicyResidual(depth))
          continue;
      }

      Loc  nextBestMove    = LOC_NULL;
      int  newDepth        = depth - 1;
      bool fullDepthSearch = !PV || moveCount > 1;

      // 延伸: 估值显著增加
      newDepth += evalDelta >= 150;

      evaluator->play(me, move);

      // Do LMR
      if (depth >= LMR_DEPTH && moveCount > 1
          && (moveCount >= lateMoveCount<PV>(depth) || isCut)) {
        int r = lmrReduction(depth, moveCount) + 2 * isCut;
        r += (plyInfos[ply].currentPolicySum > 0.95);
        r -= (ply > 1 && policySum < 0.75 && plyInfos[ply - 1].currentPolicySum < 0.1);
        r += (evalDelta < -120) + (evalDelta < -200);

        int d = std::clamp(newDepth - r, 1, newDepth);
        value =
            -search<false>(oppo, ply + 1, d, -(alpha + 1), -alpha, true, nextBestMove);
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
        value =
            -search<true>(oppo, ply + 1, newDepth, -beta, -alpha, false, nextBestMove);

      evaluator->undo(me, move);
    }

    if (terminate.load(std::memory_order_relaxed))
      return VALUE_NONE;

    if (value > bestValue) {
      bestValue = value;
      if (value > alpha) {
        bestmove = move;
        copyPV(plyInfos[ply].pv, move, plyInfos[ply + 1].pv);

        if (!PV || value >= beta)
          break;
        else
          alpha = value;
      }
    }
  }

  tte->save(evaluator->key,
            valueToTT(bestValue, ply),
            bestValue >= beta            ? BOUND_LOWER
            : PV && bestmove != LOC_NULL ? BOUND_EXACT
                                         : BOUND_UPPER,
            depth,
            PV,
            bestmove);

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
  const Color *board = (color == C_BLACK) ? evaluator->blackEvaluator->board
                                          : evaluator->whiteEvaluator->board;
  int          x0 = toplayLoc % BS, y0 = toplayLoc / BS;
  int          dirX[4] = {1, 0, 1, 1};
  int          dirY[4] = {0, 1, 1, -1};
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

void PVSsearch::normalizePolicy(const PolicyType *rawPolicy, float *normPolicy) const
{
  PolicyType maxRawPolicy = *std::max_element(rawPolicy, rawPolicy + BS * BS);
  std::transform(rawPolicy, rawPolicy + BS * BS, normPolicy, [=](auto &p) {
    const double invQ = 1.0 / quantFactor;
    return (float)std::exp((p - maxRawPolicy) * invQ);
    // return (float)std::max(p, PolicyType(0));
  });
  float policySum = std::reduce(normPolicy, normPolicy + BS * BS);
  float k         = 1 / policySum;
  std::transform(normPolicy, normPolicy + BS * BS, normPolicy, [=](auto &p) {
    return p * k;
  });
}

void PVSsearch::sortPolicy(const float *policy,
                           Loc *        policyRank) const  // assume policyBuf is ready
{
  std::iota(policyRank, policyRank + BS * BS, LOC_ZERO);
  std::sort(policyRank, policyRank + BS * BS, [&](Loc a, Loc b) {
    return policy[a] > policy[b];
  });
}

void PVSsearch::copyPV(Loc *pvDst, Loc bestMove, Loc *pvSrc) const
{
  *pvDst++ = bestMove;
  do {
    *pvDst++ = *pvSrc;
  } while (*pvSrc++ != LOC_NULL);
}

std::vector<Loc> PVSsearch::rootPV() const
{
  std::vector<Loc> pv;
  const Loc *      pvPtr = plyInfos[0].pv;
  while (*pvPtr != LOC_NULL)
    pv.push_back(*pvPtr++);
  return pv;
}

void PVSsearch::clear()
{
  terminate = false;
  nodes = interiorNodes = 0;
  ttHits = ttCuts = 0;
}
