#ifndef NNUE_NNUE_VCF_ABSEARCH_H_
#define NNUE_NNUE_VCF_ABSEARCH_H_

#include "../game/board.h"
#include "../game/boardhistory.h"
#include "NNUEBoardHistory.h"
#include <vector>
#include <utility>

namespace NNUE {

// AB search class for checking consecutive four-threat wins in Connect6 with one player making two moves
class VCF_ABSearch {
private:
  NNUEBoardHistory* boardHistory;
  Player attackPlayer;   // Attacking player
  Player defendPlayer;   // Defending player
  double remainingDepth; // Remaining search depth (equivalent layers)
  
public:
  VCF_ABSearch(NNUEBoardHistory* hist, Player pla);
  
  // Main search function
  double search(double maxDepth);
  
private:
  // AB search core function
  double alphaBeta(double alpha, double beta, double depth);
  
  // Check game state
  int checkGameState();
  
  // Get legal moves and sort by policy
  std::vector<std::pair<Loc, double>> getLegalMovesWithPolicy(const std::vector<Loc>& legalLocs, bool hasLegalLocs);
  
  // Calculate leaf node evaluation
  double evaluateLeafAssumeNotEnd();
};

} // namespace NNUE

#endif // NNUE_NNUE_VCF_ABSEARCH_H_