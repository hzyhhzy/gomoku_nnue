#include "Board.h"

#include <random>
using namespace NNUE;

Board::Board(std::string type, std::string filepath, const Rules& rules) :moveCacheBlength(0), moveCacheWlength(0)
{
  x_size = MaxBS;
  y_size = MaxBS;
  rule = rules;
  if (type == "nnuev2") {
    blackEvaluator = new Eva_nnuev2();
    whiteEvaluator = new Eva_nnuev2();
  }
  else
  {
    throw "Invalid type of engine";
  }

  loadParam(filepath, filepath);//include clear

}



bool Evaluator::loadParam(std::string filepathB, std::string filepathW)
{
  bool suc = blackEvaluator->loadParam(filepathB) && whiteEvaluator->loadParam(filepathW);
  clear();
  return suc;
}

void Evaluator::clear()
{
  moveCacheBlength = 0;
  moveCacheWlength = 0;
  blackEvaluator->clear();
  whiteEvaluator->clear();

  for (int loc = 0; loc < MaxBS * MaxBS; loc++)
    board[loc] = C_EMPTY;
  pos_hash = Hash128();
  noResultUtilityForWhite = 0.0;
  movenum = 0;
  stage = 1;//only 1 stone in the first move
  sumStoneX = 0;
  sumStoneY = 0;
  numStones = 0;
  meanStoneX = 0;
  meanStoneY = 0;
  nextPla = C_BLACK;
  firstLoc = NU_LOC_NULL;
  firstLocPriority = 0.0;
  blackPassNum = 0;
  whitePassNum = 0;

}

double Evaluator::getLocationPriority(int x, int y) const {
  if (numStones == 0)
    return 0.0;
  double dx = x - meanStoneX;
  double dy = y - meanStoneY;
  return dx * dx + dy * dy;
}

double Evaluator::getLocationPriority(NU_Loc loc) const {
  if (loc < 0 || loc >= MaxBS * MaxBS || numStones == 0)
    return 0.0;
  double dx = loc % MaxBS - meanStoneX;
  double dy = loc / MaxBS - meanStoneY;
  return dx * dx + dy * dy;
}

void Evaluator::setStone(Color color, NU_Loc loc)
{
  if (loc < 0 || loc >= MaxBS * MaxBS)
    return;


  Color colorOld = board[loc];
  board[loc] = color;
  pos_hash ^= NNUEHashTable::ZOBRIST_loc[colorOld][loc];
  pos_hash ^= NNUEHashTable::ZOBRIST_loc[color][loc];

  int x = loc % MaxBS;
  int y = loc / MaxBS;
  if (colorOld == C_EMPTY && color != C_EMPTY)//add a stone
  {
    numStones += 1;
    sumStoneX += x;
    sumStoneY += y;
  }
  else if (colorOld != C_EMPTY && color == C_EMPTY)  // remove a stone
  {
    numStones -= 1;
    sumStoneX -= x;
    sumStoneY -= y;
  }
  meanStoneX = numStones == 0 ? 0.0 : double(sumStoneX) / double(numStones);
  meanStoneY = numStones == 0 ? 0.0 : double(sumStoneY) / double(numStones);
}

void Evaluator::play(Color color, NU_Loc loc)
{
  if (color != nextPla)
    throw "wrong next player";
  addCache(false, color, loc);//nnue evaluator


  movenum++;

  if (stage == 0)  //choose
  {
    stage = 1;
    firstLoc = loc;
    firstLocPriority = getLocationPriority(loc);
  }
  else if (stage == 1)  //place
  {
    stage = 0;

    if (loc >= 0 && loc < MaxBS * MaxBS) {
      setStone(color, loc);
    }
    if (firstLoc >= 0 && firstLoc < MaxBS * MaxBS) {
      setStone(color, firstLoc);
    }

    int newPassCount = 0;
    if (loc == NU_LOC_PASS)
      newPassCount++;
    if (firstLoc == NU_LOC_PASS)
      newPassCount++;
    if (firstLoc == NU_LOC_NULL)
      newPassCount = 0; //if black passes at the first move, maybe the user needs some special openings rather than blackPassNum+1
    if (newPassCount > 0) {
      if (color == C_BLACK) {
        blackPassNum += newPassCount;
      }
      else if (color == C_WHITE) {
        whitePassNum += newPassCount;
      }
    }

    firstLoc = NU_LOC_NULL;
    firstLocPriority = 0.0;

    nextPla = getOpp(nextPla);

  }
  else
    throw "stage not 0 or 1";


}

void Evaluator::undo(const UndoRecord& ur, Color color, NU_Loc loc)
{
  addCache(true, color, loc);
  if (stage == 0)//remove the played two stones
  {
    if (loc >= 0 && loc < MaxBS * MaxBS) {
      setStone(C_EMPTY, loc);
    }
    NU_Loc firstLoc1 = ur.firstLoc;
    if (firstLoc1 >= 0 && firstLoc1 < MaxBS * MaxBS) {
      setStone(C_EMPTY, firstLoc1);
    }
  }
  // Restore the state from UndoRecord
  movenum = ur.movenum;
  stage = ur.stage;
  meanStoneX = ur.meanStoneX;
  meanStoneY = ur.meanStoneY;
  nextPla = ur.nextPla;
  firstLoc = ur.firstLoc;
  firstLocPriority = ur.firstLocPriority;
  blackPassNum = ur.blackPassNum;
  whitePassNum = ur.whitePassNum;
}

void Evaluator::clearCache(Color color)
{
  if (color == C_BLACK)
  {
    for (int i = 0; i < moveCacheBlength; i++)
    {
      MoveCache move = moveCacheB[i];
      if (move.isUndo)blackEvaluator->undo(move.loc);
      else blackEvaluator->play(move.color, move.loc);
    }
    moveCacheBlength = 0;
  }
  else if (color == C_WHITE)
  {
    for (int i = 0; i < moveCacheWlength; i++)
    {
      MoveCache move = moveCacheW[i];
      if (move.isUndo)whiteEvaluator->undo(move.loc);
      else whiteEvaluator->play(getOpp(move.color), move.loc);
    }
    moveCacheWlength = 0;
  }
}

void Evaluator::addCache(bool isUndo, Color color, NU_Loc loc)
{
  MoveCache newcache(isUndo, color, loc);

  if (moveCacheBlength == 0 || !isContraryMove(moveCacheB[moveCacheBlength - 1], newcache))
  {
    moveCacheB[moveCacheBlength] = newcache;
    moveCacheBlength++;
  }
  else//可以消除一步
  {
    moveCacheBlength--;
  }

  if (moveCacheWlength == 0 || !isContraryMove(moveCacheW[moveCacheWlength - 1], newcache))
  {
    moveCacheW[moveCacheWlength] = newcache;
    moveCacheWlength++;
  }
  else//可以消除一步
  {
    moveCacheWlength--;
  }
}

bool Evaluator::isContraryMove(MoveCache a, MoveCache b)
{
  if (a.isUndo == b.isUndo)return false;
  else
  {
    if (a.loc != b.loc)std::cout << "Evaluator::isContraryMove strange bugs";
    if (a.color != b.color)std::cout << "Evaluator::isContraryMove strange bugs";
    return true;
  }
}

Evaluator::UndoRecord::UndoRecord(const Evaluator& eva)
{
  movenum = eva.movenum;              // Initialize move number
  stage = eva.stage;                  // Initialize stage
  meanStoneX = eva.meanStoneX;        // Initialize meanStoneX
  meanStoneY = eva.meanStoneY;        // Initialize meanStoneY
  nextPla = eva.nextPla;              // Initialize the next player's color
  firstLoc = eva.firstLoc;            // Initialize the first stone location
  firstLocPriority = eva.firstLocPriority; // Initialize the priority for firstLoc
  blackPassNum = eva.blackPassNum;    // Initialize black's pass count
  whitePassNum = eva.whitePassNum;    // Initialize white's pass count
}
