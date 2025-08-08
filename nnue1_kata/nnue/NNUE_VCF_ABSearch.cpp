#include "NNUE_VCF_ABSearch.h"
#include "../game/gamelogic.h"
#include "Eva_nnuev2.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cassert>

using namespace std;
using namespace NNUE;

// VCF_ABSearch class implementation
VCF_ABSearch::VCF_ABSearch(NNUEBoardHistory* hist, Player pla) 
  : boardHistory(hist), attackPlayer(pla), defendPlayer(getOpp(pla)) {
  // Check initial state
  assert(boardHistory->getBoard().stage == 0);
  assert(boardHistory->getBoard().nextPla == attackPlayer);
}

double VCF_ABSearch::search(double maxDepth) {
  remainingDepth = maxDepth;
  return alphaBeta(-std::numeric_limits<double>::infinity(), 
                   std::numeric_limits<double>::infinity(), 
                   maxDepth);
}

double VCF_ABSearch::alphaBeta(
    double alpha, double beta, double depth) {
  
  // Directly determine search direction based on current player (attacker's perspective)
  
  // Check game state
  Color maybeWinner=C_WALL;
  int gameEndMovenum=0;
  std::vector<Loc> allLegalLocs = GameLogic::getAllVCFAttackOrDefenseLocs(boardHistory->getBoard(), attackPlayer, maybeWinner, gameEndMovenum);

  //int gameState = checkGameState();
  //cout << gameState;
  if (maybeWinner != C_WALL) {

    if (maybeWinner == attackPlayer) { // Attacker wins
      // Attacker wins: board.x_size*board.y_size+100-board.numStones (ensure > 1)
      double winValue = boardHistory->getBoard().x_size * boardHistory->getBoard().y_size + 100 - gameEndMovenum;

      return winValue;
    } else { // Defender wins
      // Defender wins: -2.0
      return -2.0;
    }
  }
  
  // Check if reached leaf node (when defender moves with depth<0 and no clear winner)
  if (depth < 0 && boardHistory->getBoard().nextPla == defendPlayer) {
    return evaluateLeafAssumeNotEnd();
  }
  
  // Get legal moves
  vector<pair<Loc, double>> moves = getLegalMovesWithPolicy(allLegalLocs, true);

  if (moves.empty()) {
    if(boardHistory->getBoard().numStones == boardHistory->getBoard().x_size * boardHistory->getBoard().y_size)
      return -2.0; // Draw,failed to attack
    else
    // play a corner move which cause the second move can only be pass
      if(boardHistory->getBoard().nextPla == attackPlayer)
        return -10000; 
      else
        return 10000;
  }
  
  if (boardHistory->getBoard().nextPla == attackPlayer) { // Attacker maximizes
    double bestValue = -std::numeric_limits<double>::infinity();
    
    // Attacker moves, need to filter move selection
    // Find the move with maximum policy
    double maxPolicy = -1e100;
    for (const auto& move : moves) {
      maxPolicy = max(maxPolicy, move.second);
    }
    int t = 0;
    for (const auto& move : moves) {
      Loc loc = move.first;
      double policy = move.second;
      
      // Check if this move should be excluded
      double policyCost = -policy;
      //if (policy == maxPolicy)policyCost = 0;
      if (policyCost > depth && policy < maxPolicy - 0.01) {
        // Excluded moves are assumed to have value -1.5
        double childValue = -1.5;
        if (childValue > bestValue) {
          bestValue = childValue;
        }
        continue;
      }
      t += 1;
      // Make move
      boardHistory->play(boardHistory->getBoard().nextPla, loc);
      
      // Calculate new depth
      double newDepth = depth - policyCost;
      
      // Recursive search
      double childValue = alphaBeta(alpha, beta, newDepth);
      
      // Undo move
      boardHistory->undo();
      
      if (childValue > bestValue) {
        bestValue = childValue;
      }
      
      // Attacker returns immediately once a winning move is found (value > 1)
      if (bestValue > 1.0) {
        return bestValue;
      }
      
      alpha = max(alpha, bestValue);
      if (beta <= alpha) {
        break; // Beta pruning
      }
    }
    //cout << depth << " " << t << endl;
     return bestValue;
  } else { // Defender minimizes
    double bestValue = std::numeric_limits<double>::infinity();
    for (const auto& move : moves) {
      Loc loc = move.first;
      double policy = move.second;
      
      // Record current player
      Player currentPlayer = boardHistory->getBoard().nextPla;
      
      // Make move
      boardHistory->play(currentPlayer, loc);
      
      // Defender does not deduct depth
      double childValue = alphaBeta(alpha, beta, depth);
      
      // Undo move
      boardHistory->undo();
      
      if (childValue < bestValue) {
        bestValue = childValue;
      }
      
      beta = min(beta, bestValue);
      if (beta <= alpha) {
        break; // Alpha pruning
      }
    }
    return bestValue;
  }
}

int VCF_ABSearch::checkGameState() {
  const Board& board = boardHistory->getBoard();
  //cout << (board.nextPla == defendPlayer ? "w" : "b") << board.stage;
  
  if (board.stage == 0 && board.nextPla == defendPlayer) {
    // Attacker just finished placing two pieces
    int twoFourThreats = GameLogic::checkTwoFourThreats(board, attackPlayer);
    int oppMaxLen = GameLogic::checkMaxConnectLen(board, defendPlayer);

    //Board::printBoard(cout, board, board.firstLoc, NULL);
    //cout.flush();

    if (twoFourThreats == 4) {
      return 1; // Attacker wins
    }
    if (oppMaxLen >= 4) {
      return -1; // Defender wins
    }
    if (twoFourThreats == 0 || twoFourThreats == 1) {
      return -1; // Defender wins (can block)
    }
    if (twoFourThreats == 3) {
      return 1; // Attacker wins (cannot block)
    }
    // twoFourThreats == 2, continue search
  }
  
  if (board.stage == 1 && board.nextPla == defendPlayer) {
    // Defender placed one piece
    int twoFourThreats = GameLogic::checkTwoFourThreats(board, attackPlayer);
    if (twoFourThreats == 1) {
      return 0; // Continue
    }
    if (twoFourThreats == 2 || twoFourThreats == 3) {
      //Board::printBoard(cout, board, board.firstLoc, NULL);
      //cout.flush();
      return 1; // Attacker wins
    }
    if (twoFourThreats == 0) {
      return -1; // Defender wins
    }
  }
  
  if (board.stage == 0 && board.nextPla == attackPlayer) {
    // Attacker's turn to place two pieces
    int oppMaxLen = GameLogic::checkMaxConnectLen(board, defendPlayer);
    if (oppMaxLen >= 6) {
      return -1; // Defender wins
    }
    int plaMaxLen = GameLogic::checkMaxConnectLen(board, attackPlayer);
    if (plaMaxLen >= 4) {
      return 1; // Attacker wins
    }
  }
  
  return 0; // Continue search
}

vector<pair<Loc, double>> VCF_ABSearch::getLegalMovesWithPolicy(const std::vector<Loc>& legalLocs, bool hasLegalLocs) {
  if(hasLegalLocs)
    assert(legalLocs.size() > 0);


  const Board& board = boardHistory->getBoard();
  // Update input buffer
  boardHistory->updateInputBuf(board.nextPla);
  
  // Get policy
  NNUE::PolicyType policy[MaxBS * MaxBS + 1];
  boardHistory->evaluateFull(board.nextPla, policy);
  
  // Collect all legal positions with their policy values
  vector<pair<Loc, double>> locPolicyPairs;
  
  // Find all legal positions and their policy values
  if(hasLegalLocs)
  {
    for (const Loc& loc : legalLocs) {
      assert(board.isLegal(loc, board.nextPla));
      if(board.stage == 1)
        assert(board.getLocationPriority(loc) + Board::PRIOR_EPS >= board.firstLocPriority);
      int nu_loc = Location::getX(loc, board.x_size) + Location::getY(loc, board.x_size) * MaxBS;

      locPolicyPairs.push_back(make_pair(loc, (double)policy[nu_loc]));
    }

  }
  else
  {
    for (int y = 0; y < board.y_size; y++) {
      for (int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if (board.isLegal(loc, board.nextPla) && (board.stage==0 || (board.getLocationPriority(loc) + Board::PRIOR_EPS >= board.firstLocPriority)))
        {
          int nu_loc = x + y * MaxBS;
          locPolicyPairs.push_back(make_pair(loc, (double)policy[nu_loc]));
        }
      }
    }
  }
  
  if (locPolicyPairs.empty()) {
    assert(false);
    return locPolicyPairs;
  }
  
  // Sort by policy value from high to low
  sort(locPolicyPairs.begin(), locPolicyPairs.end(), 
       [](const pair<Loc, double>& a, const pair<Loc, double>& b) {
         return a.second > b.second;
       });
  
  // Apply penalty to sorted positions
  double penalty_const1 = 0.0;
  double penalty_const2 = 5.0;
  double penalty_const3 = 0.3;
  
  for (int i = 0; i < locPolicyPairs.size(); i++) {
    double penalty = penalty_const1 * log(i + penalty_const2);
    locPolicyPairs[i].second -= penalty;
  }
  
  // Find maximum policy value after penalty
  double maxPolicy = -std::numeric_limits<double>::infinity();
  for (const auto& pair : locPolicyPairs) {
    maxPolicy = max(maxPolicy, pair.second);
  }
  
  // Calculate softmax denominator
  double ptemp=1.0;
  double sumExp = 0.0;
  for (const auto& pair : locPolicyPairs) {
    sumExp += exp(ptemp * (pair.second - maxPolicy) / NNUE::policyQuantFactor);
  }
  double logSumExp = log(sumExp) / ptemp;
  
  // Collect legal moves and calculate log_softmax
  for (auto& pair : locPolicyPairs) {
    double logSoftmax = (pair.second - maxPolicy) / NNUE::policyQuantFactor - logSumExp;
    pair.second=logSoftmax;
  }
  
  // Sort by policy from high to low
  //sort(moves.begin(), moves.end(), 
  //     [](const pair<Loc, double>& a, const pair<Loc, double>& b) {
  //       return a.second > b.second;
  //     });
  
  return locPolicyPairs;
}

double VCF_ABSearch::evaluateLeafAssumeNotEnd() {
  // First check if the game outcome is already determined
  //int gameState = checkGameState();
  //cout << gameState;
  //if (gameState == 1) {
    // Attacker wins: board.x_size*board.y_size+100-board.numStones (ensure > 1)
  //  return boardHistory->getBoard().x_size * boardHistory->getBoard().y_size + 100 - boardHistory->getBoard().movenum;
 // }
 // if (gameState == -1) {
    // Defender wins: -2.0
  //  return -2.0;
  //}
  
  // Use NNUE evaluation (neural network returns values in [-1, 1] interval)
  boardHistory->updateInputBuf(boardHistory->getBoard().nextPla);
  NNUE::ValueType value = boardHistory->evaluateFull(boardHistory->getBoard().nextPla, nullptr);

  //Board::printBoard(cout, boardHistory->getBoard(), boardHistory->getBoard().firstLoc, NULL);
  //cout.flush();
  
   
  // Convert to attacker's perspective win rate
  if (boardHistory->getBoard().nextPla == attackPlayer) {
    return value.win-value.loss-value.draw;
  } else {
    return value.loss- value.draw -value.win;
  }
}