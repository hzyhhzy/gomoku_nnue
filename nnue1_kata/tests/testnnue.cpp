#include "../tests/tests.h"

#include "../program/playutils.h"
#include "../main.h"

#include "../nnue/Eva_nnuev2.h"
#include "../nnue/NNUEBoardHistory.h"
#include "../game/gamelogic.h"
#include "../neuralnet/nninputs.h"
//------------------------
#include "../core/using.h"
//------------------------

using namespace std;
using namespace NNUE;
using namespace NNUEV2;

int MainCmds::testnnue() {
  Board::initHash();
  Rand seedRand;

  ConfigParser cfg;
  string nnueModelFile;
  /*
  try {
    KataGoCommandLine cmd("Test NNUE.");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(), "nnue_example.cfg");
    TCLAP::ValueArg<int> boardSizeArg(
      "", "boardsize", "Size of board, default 19", false, 19, "SIZE");
    cmd.add(boardSizeArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    boardSize = boardSizeArg.getValue();
    cmd.getConfig(cfg);

  } catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }*/

  const bool logToStdoutDefault = true;
  const bool logToStderrDefault = false;
  const bool logTimeDefault = false;
  Logger logger(NULL, logToStdoutDefault, logToStderrDefault, logTimeDefault);

  string modelpath = "H:/gomtrain2024/connectsix/export/v2_64_highwd2.txt";

  NNUEV2::ModelWeight* nnueWeight = new NNUEV2::ModelWeight();
  nnueWeight->loadParam(modelpath);

  Eva_nnuev2* eva = new Eva_nnuev2(nnueWeight);

  Board board(19, 19);
  string rootBoardSequence = "j10i8l10";
  vector<Loc> rootBoardLocSeq = Location::parseSequenceGom(rootBoardSequence, board);
  PlayUtils::playMoveLocSequence(board, board.nextPla, rootBoardLocSeq);
  eva->debug_print();
  
  // Test auto-play function
  testAutoPlay(nnueWeight);
  
  delete nnueWeight;
  return 0;
}

// Test function: auto-play from empty 19x19 board using highest policy moves
void testAutoPlay(const ModelWeight* weights) {
  cout << "Starting auto-play test from empty 19x19 board..." << endl;
  
  // Create board and rules
  Board board(19, 19);
  Rules rules;
  string rootBoardSequence = "j10i9i11";
  vector<Loc> rootBoardLocSeq = Location::parseSequenceGom(rootBoardSequence, board);
  PlayUtils::playMoveLocSequence(board, board.nextPla, rootBoardLocSeq);
  
  // Create NNUE input parameters
  MiscNNInputParams nnInputParams;
  
  // Create NNUEBoardHistory
  NNUEBoardHistory nnueHistory(weights, nnInputParams);
  nnueHistory.clear(board, board.nextPla, rules);
  
  
  cout << "Starting game with empty board..." << endl;
  
  while (true) {
    Color pla = board.nextPla;
    // Get policy from NNUE
    PolicyType policy[MaxBS * MaxBS + 1];
    auto value = nnueHistory.evaluateFull(pla, policy);
    bool cons= nnueHistory.checkEvaluatorBoardConsistency();
    if (!cons)
      throw StringError("nnueHistory not consist");
    
    // Find the move with highest policy value among legal moves
    Loc bestMove = Board::NULL_LOC;
    PolicyType bestPolicy = MIN_POLICY;
    
    for (int nu = 0; nu < MaxBS * MaxBS + 1; nu++) {
      Loc loc = nu < MaxBS * MaxBS ? Location::getLoc(nu % MaxBS, nu / MaxBS, board.x_size) : Board::PASS_LOC;
      if(!board.isLegal( loc, pla))
        continue;
      if(board.stage == 1 && loc != Board::PASS_LOC) { // check move priority
        if(board.getLocationPriority(loc) + Board::PRIOR_EPS < board.firstLocPriority)
          continue;
      }

      if (policy[nu] > bestPolicy) {
        bestPolicy = policy[nu];
        bestMove = loc;
      }
        
    }
    
    // If no legal move found, try pass
    if (bestMove == Board::NULL_LOC) {
      ASSERT_UNREACHABLE;
      bestMove = Board::PASS_LOC;
    }
    
    
    // Make the move
    nnueHistory.play(pla, bestMove);
    board.playMoveAssumeLegal(bestMove, pla);
    
    // Print move information
    string moveStr = Location::toString(bestMove, board);
    cout << "Move " << board.movenum << ": " << (pla == C_BLACK ? "Black" : "White") 
         << " plays " << moveStr << " (policy: " << bestPolicy << ")" << endl;
    cout << "Win=" << value.win << " Lose=" << value.loss <<" Draw="<< value.draw << endl;
    
    
    
    // Optional: print board state every 10 moves
    cout << "Board state after " << board.movenum << " moves:" << endl;
    Board::printBoard(cout, board, board.firstLoc, NULL);
    cout << endl;

    // Check for game end conditions
    if (nnueHistory.isGameFinished) {
      cout << "Game finished - winner detected after " << board.movenum << " moves." << endl;
      cout<<"winner="<<(nnueHistory.winner==C_BLACK?"Black":nnueHistory.winner==C_WHITE?"White":"Draw")<<endl;

      break;
    }
  }
  
}
