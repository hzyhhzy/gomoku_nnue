#include "Evaluator.h"
#include "AllEvaluator.h"

#include <random>

Evaluator::Evaluator(std::string type, std::string filepath):moveCacheBlength(0),moveCacheWlength(0)
{
  initZobrist(0x114514AA114514AA);
  if (type == "nnuev2") {
    blackEvaluator = new Eva_nnuev2();
    whiteEvaluator = new Eva_nnuev2();
    loadParam(filepath, filepath);
  }
  else
  {
    throw "Invalid type of engine";
  }

}

void Evaluator::initZobrist(uint64_t seed)
{
  std::mt19937_64 prng {seed};
  prng();
  prng();
  key = prng();
  for (Key &k : zobrist[0])
    k = prng();
  for (Key &k : zobrist[1])
    k = prng();
}


bool Evaluator::loadParam(std::string filepathB, std::string filepathW)
{
  bool suc= blackEvaluator->loadParam(filepathB) && whiteEvaluator->loadParam(filepathW);
  clear();
  return suc;
}

void Evaluator::clear()
{
  moveCacheBlength = 0;
  moveCacheWlength = 0;
  for (int i = 0; i < MaxBS * MaxBS; i++)board[i] = C_EMPTY;
  blackEvaluator->clear();
  whiteEvaluator->clear();
}

void Evaluator::play(Color color, Loc loc)
{
  key ^= zobrist[color - C_BLACK][loc];
  if (board[loc] != C_EMPTY)std::cout << "Evaluator: Illegal Move\n";
  board[loc] = color;
  addCache(false, color, loc);
  //blackEvaluator->play(color, loc); 
  //whiteEvaluator->play(getOpp(color), loc); 
}

void Evaluator::undo(Loc loc)
{
  Color color = board[loc];
  if (color == C_EMPTY)std::cout << "Evaluator: Illegal Undo\n";
  board[loc] = C_EMPTY;
  key ^= zobrist[color - C_BLACK][loc];
  addCache(true, color, loc);
  //blackEvaluator->undo(loc); 
  //whiteEvaluator->undo(loc); 
}

void Evaluator::clearCache(Color color)
{
  if (color == C_BLACK)
  {
    for (int i = 0; i < moveCacheBlength; i++)
    {
      MoveCache move = moveCacheB[i];
      if(move.isUndo)blackEvaluator->undo(move.loc);
      else blackEvaluator->play(move.color,move.loc);
    }
    moveCacheBlength = 0;
  }
  else if (color == C_WHITE)
  {
    for (int i = 0; i < moveCacheWlength; i++)
    {
      MoveCache move = moveCacheW[i];
      if(move.isUndo)whiteEvaluator->undo(move.loc);
      else whiteEvaluator->play(getOpp(move.color),move.loc);
    }
    moveCacheWlength = 0;
  }
}

void Evaluator::addCache(bool isUndo, Color color, Loc loc)
{
  MoveCache newcache(isUndo, color, loc);

  if (moveCacheBlength == 0|| !isContraryMove(moveCacheB[moveCacheBlength-1],newcache))
  {
    moveCacheB[moveCacheBlength] = newcache;
    moveCacheBlength++;
  }
  else//可以消除一步
  {
    moveCacheBlength--;
  }

  if (moveCacheWlength == 0|| !isContraryMove(moveCacheW[moveCacheWlength-1],newcache))
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
  if (a.loc != b.loc)return false;
  if (a.color != b.color)return false;
  if (a.isUndo != b.isUndo)return true;
  else std::cout<<"Evaluator::isContraryMove strange bugs";

}
