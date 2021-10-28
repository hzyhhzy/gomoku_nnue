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
  std::ifstream weightfileB(filepathB,std::ios::in), weightfileW(filepathW, std::ios::in);
 // const int s = 131072;
  //char *buf=new char[s];
  //weightfileB.rdbuf()->pubsetbuf(buf, s);
 // weightfileW.rdbuf()->pubsetbuf(buf, s);
  return blackEvaluator->loadParam(weightfileB) && whiteEvaluator->loadParam(weightfileW);
}
