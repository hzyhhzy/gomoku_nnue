#include "../program/playutils.h"

#include <sstream>

#include "../core/timer.h"
#include "../core/test.h"

using namespace std;


void PlayUtils::playMoveLocSequence(Board& board, Player& nextPlayer, vector<Loc> locs) {
  nextPlayer = board.nextPla;
  for(int i = 0; i < locs.size(); i++) {
    Loc loc = locs[i];
    if(!board.isLegal(loc, nextPlayer))
      throw StringError("Illegal moves in maybeParseInitialMoveBonus");
    if(loc != Board::NULL_LOC)
      board.playMoveAssumeLegal(loc, nextPlayer);
    nextPlayer = board.nextPla;
  }
}
