#ifndef PROGRAM_PLAY_UTILS_H_
#define PROGRAM_PLAY_UTILS_H_

#include "../core/config_parser.h"
#include "../game/board.h"

//This is a grab-bag of various useful higher-level functions that select moves or evaluate the board in various ways.

namespace PlayUtils {
 



  void playMoveLocSequence(Board& board, Player& nextPlayer, std::vector<Loc> locs); 

}


#endif //PROGRAM_PLAY_UTILS_H_
