#include "Evaluator.h"
#include "AllEvaluator.h"

#include <random>

Evaluator::Evaluator(std::string type, std::string filepathB, std::string filepathW)
{
  initZobrist(0x114514AA114514AA);
  if (type == "sum1")//¼òµ¥ÇóºÍ
  {
    blackEvaluator = new Eva_sum1();
    whiteEvaluator = new Eva_sum1();
    loadParam(filepathB, filepathW);
  }
  else if (type == "mix6")
  {
    blackEvaluator = new Eva_mix6_avx2();
    whiteEvaluator = new Eva_mix6_avx2();
    loadParam(filepathB, filepathW);
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
  return blackEvaluator->loadParam(filepathB) && whiteEvaluator->loadParam(filepathW);
}
