#include "Evaluator.h"
#include "AllEvaluator.h"
Evaluator::Evaluator(std::string type, std::string filepath)
{
  if (type == "sum1")//¼òµ¥ÇóºÍ
  {
    blackEvaluator = new Eva_sum1();
    whiteEvaluator = new Eva_sum1(); 
    loadParam(filepath, filepath);
  }
  else
  {
    throw "Invalid type of engine";
  }
}


bool Evaluator::loadParam(std::string filepathB, std::string filepathW)
{
  std::ifstream weightfileB(filepathB), weightfileW(filepathW);
  return blackEvaluator->loadParam(weightfileB) && whiteEvaluator->loadParam(weightfileW);
}
