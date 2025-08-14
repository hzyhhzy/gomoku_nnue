#pragma once
// VCF Logic header file
#include "../game/board.h"



namespace VCFLogic {
  //Check if player has two four-in-a-row threats
  int checkTwoFourThreats(const Board& board, Player pla);
  
  //Check maximum consecutive length for a player
  int checkMaxConnectLen(const Board& board, Player pla);
  
  //Get all possible VCF attack or defense locations
  //if board.nextPla==attackPla, return all attack locations
  //else return all defense locations
  
  std::vector<Loc> getAllVCFAttackOrDefenseLocs(const Board& board, Player attackPla, Color& winner, int& gameEndMovenum);
  
  //One-layer brute force search for stage==1 and nextPla==attackPla
  //Check if any VCF attack move leads to immediate win
  Loc findImmediateWinInVCFAttack(const Board& board, Player attackPla);
  Loc findImmediateWinInVCFAttackLayer2(const Board& board, Player attackPla);
}
