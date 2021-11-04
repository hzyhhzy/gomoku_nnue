#include "ABsearch.h"

float ABsearch::fullsearch(Color color, double factor, Loc &bestmove)
{
    int depth = factor;
    return searchRec(color, depth, WIN_VALUE, -WIN_VALUE, bestmove);
}

float ABsearch::searchRec(Color color, int depth, float maxEval, float minEval, Loc &bestmove)
{
    // std::cout << "Debug: depth=" << depth << " maxEval=" << maxEval << " minEval=" << minEval <<
    // std::endl;//debug
    PolicyType *policyPtr     = policyBuf[depth];
    int *       policyRankPtr = policyRankBuf[depth];
    float       value         = evaluator->evaluate(color, policyPtr).winlossrate();
    // std::cout << value<<std::endl;
    // if (color == C_BLACK)evaluator->blackEvaluator->debug_print();//debug
    // if (color == C_WHITE)evaluator->whiteEvaluator->debug_print();//debug
    if (depth == 0)
        return value;
    sortPolicy(policyPtr, policyRankPtr);
    PolicyType minPolicy = policyPtr[policyRankPtr[0]] - explorationFactor(depth);

    bestmove        = NULL_LOC;
    float bestValue = -1e30;
    for (int i = 0; i < BS * BS; i++) {
        Loc loc = policyRankPtr[i];
        if (policyPtr[loc] < minPolicy && i >= minExploreChildren(depth))
            break;
        if (boardPointer[loc] != C_EMPTY) {
            using namespace std;
            std::cout << "This nonempty point has a very big policy " << loc << " " << i << " "
                      << minExploreChildren(depth) << " " << policyPtr[loc] << "\n ";

            for (int y = 0; y < BS; y++) {
                for (int x = 0; x < BS; x++) {
                    Color c = boardPointer[y * BS + x];
                    if (c == C_EMPTY)
                        cout << ".\t";
                    else if (c == C_MY)
                        cout << "x\t";
                    else if (c == C_OPP)
                        cout << "o\t";
                }
                cout << endl;
                for (int x = 0; x < BS; x++) {
                    int p = policyPtr[y * BS + x];
                    cout << p << "\t";
                }
                cout << endl;
            }
            continue;
        }

        float value;
        if (isWin(color, loc)) {
            value = WIN_VALUE;
        }
        else {
            evaluator->play(color, loc);
            Loc nextBestMove;
            value = -searchRec(3 - color,
                               depth - 1,
                               -std::max(bestValue, minEval),
                               -maxEval,
                               nextBestMove);
            evaluator->undo(loc);
        }
        if (depth == 9) {
            // std::cout << " Value=" << value;
        }

        if (value > bestValue) {
            bestmove  = loc;
            bestValue = value;
            if (bestValue >= maxEval)
                break;
        }
    }
    return bestValue;
}

int ABsearch::explorationFactor(int depth)
{
    return 1.0 * depth * quantFactor;
}

int ABsearch::minExploreChildren(int depth)
{
    return depth * 1;
}

bool ABsearch::isWin(Color color, Loc toplayLoc)
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

void ABsearch::sortPolicy(const PolicyType *policy, int *policyRank)  // assume policyBuf is ready
{
    std::iota(policyRank, policyRank + BS * BS, 0);
    std::sort(policyRank, policyRank + BS * BS, [&](int a, int b) {
        return policy[a] > policy[b];
    });
}
