/*
 * gamelogic.h
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 * 
 * Gomoku as a representive
 */

#ifndef GAME_GAMELOGIC_H_
#define GAME_GAMELOGIC_H_

#include "../game/boardhistory.h"

/*
* Other game logics:
* Board::
*/

namespace GameLogic {

  typedef char MovePriority;
  static const MovePriority MP_NORMAL = 126;
  static const MovePriority MP_SUDDEN_WIN = 1;//win after this move
  static const MovePriority MP_ONLY_NONLOSE_MOVES = 2;//the only non-lose moves
  static const MovePriority MP_WINNING = 3;//sure win, but not this move
  static const MovePriority MP_ILLEGAL = -1;//illegal moves

  //C_EMPTY = draw, C_WALL = not finished 
  Color checkWinnerAfterPlayed(const Board& board, const BoardHistory& hist, Player pla, Loc loc1, Loc loc2);


  //some results calculated before calculating NN
  //part of NN input, and then change policy/value according to this
  struct ResultsBeforeNN {
    bool inited;
    Color winner;
    Loc myOnlyLoc;
    ResultsBeforeNN();
    void init(const Board& board, const BoardHistory& hist, Color nextPlayer);
  };

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




#endif // GAME_RULELOGIC_H_
