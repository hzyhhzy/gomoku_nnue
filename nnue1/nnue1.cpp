// nnue1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "AllEvaluator.h"
#include "AllSearch.h"
#include "Engine.h"
#include "VCF/VCFsolver.h"
#include "validation.h"

#include <chrono>
#include <random>
#include <iostream>


using namespace std;

std::string appPath(int argc, const char **argv)
{
  std::string path(argv[0]);
  while (path[path.length() - 1] != '/' && path[path.length() - 1] != '\\'
         && path.length() > 0)
    path.pop_back();
  return path;
}

int maingtp(int argc, const char **argv)
{
  std::string modelPath = appPath(argc, argv) + "model.txt";
  std::string configPath = appPath(argc, argv) + "config.txt";


  //把txt权重转换成二进制文件，不对弈
  bool convertOnly = argc > 1 && std::string(argv[1]) == "convertonly";
  if (convertOnly)
  {
    if (argc > 2) {
      modelPath = argv[2];
    }
    Eva_nnuev2* eva=new Eva_nnuev2();
    eva->loadParam(modelPath);
    return 0;
  }

  // 如果有第二个参数，认为是model path
  if (argc > 1) {
    modelPath = argv[1];
  }
  if (argc > 2) {
    configPath = argv[2];
  }

  Engine engine("nnuev2", modelPath,configPath,false);
  engine.protocolLoop();
  return 0;
}


int main_testMCTS(int argc, const char **argv)
{
  Evaluator  *eva    = new Evaluator("nnuev2", appPath(argc, argv) + "model.txt");
  MCTSsearch *search = new MCTSsearch(eva);
  /*
  const char boardstr[] = ""
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . ";
  */

  const char boardstr[] = ""
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . o . . . . . . . . "
  ". . . . . o o . . . . . . . . "
  ". . . . . x o x . . . . . . . "
  ". . . x . o x x . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . ";
  for (int y = 0; y < MaxBS; y++)
    for (int x = 0; x < MaxBS; x++) {
      char  colorchar = boardstr[2 * (x + y * MaxBS)];
      Color color     = C_EMPTY;
      if (colorchar == 'x')
        color = C_BLACK;
      else if (colorchar == 'o')
        color = C_WHITE;
      if (color != C_EMPTY)
        search->play(color, MakeLoc(x, y));
    }

  Time tic = now();
  for (int depth = 0; depth < 100000; depth++) {
    Loc    loc;
    double value = search->fullsearch(C_BLACK, 10000, loc);
    Time      toc   = now();
    MCTSnode *rootNode = search->rootNode;
    cout << "Depth = " << depth << " Value = " << value
         << " Draw = " << rootNode->WRtotal.draw / rootNode->visits
      << " Nodes = " << rootNode->visits 
      << " Time = " << toc - tic << " Nps = " << rootNode->visits  * 1000.0 / (toc - tic) << " BestLoc = "<<locstr(loc)<< endl;
    for (int i = 0; i < rootNode->childrennum; i++) {
      MCTSnode *c = rootNode->children[i].ptr;
      cout << locstr(rootNode->children[i].loc) << " visits=" << c->visits
           << " wr=" << (c->WRtotal.loss - c->WRtotal.win) / c->visits
           << " draw=" << c->WRtotal.draw / c->visits << endl;
    }
  }
  return 0;
}

int main_testeval()
{
  Eva_nnuev2 *eva = new Eva_nnuev2();
  eva->loadParam("D:/gomtrain/renju100b/export/v2t1.txt");
  /*
  const char boardstr[] = ""
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . "
                          ". . . . . . . . . . . . . . . ";
    */

  const char boardstr[] = ""
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . o x x . . . . . "
    ". . . . . . x o o . . . . . . "
    ". . . . . . . x . x o . . . . "
    ". . . . . . . . o . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . ";
  for (int y = 0; y < MaxBS; y++)
    for (int x = 0; x < MaxBS; x++) {
      char  colorchar = boardstr[2 * (x + y * MaxBS)];
      Color color     = C_EMPTY;
      if (colorchar == 'x')
        color = C_BLACK;
      else if (colorchar == 'o')
        color = C_WHITE;
      if (color != C_EMPTY)
        eva->play(color, MakeLoc(x, y));
    }

  eva->debug_print();
  return 0;
}
int main_benchmark()
{
  Eva_nnuev2 *eva = new Eva_nnuev2();
  eva->loadParam("v2_c16.txt");

  int64_t testnum = 500000;

  std::mt19937_64 prng {uint64_t(now())};
  prng();
  prng();

  PolicyType p[MaxBS * MaxBS];
  int64_t time_start=now();

  float gf[NNUEV2::globalFeatureNum] = {0};

  // 平均每play和undo两次，然后eval一次
  for (int64_t i = 0; i < testnum; i++) {
    for (int j = 0; j < 3; j++) {
      uint64_t rand  = prng();
      Loc      loc   = rand % (MaxBS * MaxBS);
      rand           = rand / (MaxBS * MaxBS);
      Color newcolor = (eva->board[loc] + 1 + rand % 2) % 3;
      if (eva->board[loc] != C_EMPTY)
        eva->undo(loc);
      if (newcolor != C_EMPTY)
        eva->play(newcolor, loc);
    }

    auto v = eva->evaluateFull(gf,p);

  }

  int64_t time_end = now();
  double  time_used = time_end - time_start;
  cout << "NNevals = " << testnum << " Time = " << time_used / 1000 << " s" << endl;
  cout << "Speed = " << testnum/time_used * 1000 << " eval/s" << endl;
  return 0;
}
int main_testvcf()
{
  Color board[MaxBS * MaxBS];
  /*
  const char boardstr[] = ""
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ;
  */

  const char boardstr1[] = ""
  "x . . x . . x . . x . . x . . "
  "x . . x . . x . . x . . x . . "
  "x . . x . . x . . x . . x . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  "o . . o . . o . . o . . o . . "
  "x . . x . . x . . x . . x . . "
  "x . . x . . x . . x . . x . . "
  "x . . x . . x . . x . . x . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  "x . . x . . x . . x . . x . . "
  "x . . x . . x . . x . . x . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
    ;
  const char boardstr[] = ""
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . o . . . . . . . . . . "
  ". . . . . . . . . . o . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . x . . . . . "
  ". . . . . o . . . . o . . . . "
  ". . . . . x . x . . . . . . . "
  ". . . . . . . x . . . . x . o "
  ". . . x . . . . . . . x . . . "
  ". . . . o . . . . o . . . . . "
  ". . . . o . . x . . . . . . . "
  ". . . . o . . . x . . . . . . "
  ". . . . . . . . . . . . . . . "
  ". . . . . . . . . . . . . . . "
    ;
  const char boardstr2[] = ""
    ". . . . . . . . . o . . . . . "
    ". . . . . . x x x . . . . . . "
    ". . . . . . x x x o . . . . . "
    ". . . . . . x o x x . . . . . "
    ". . . . . x o . o . o . . . . "
    ". . . . o o x . x . . . . . . "
    ". . . . x o x o o o . . . . . "
    ". . . . . o x x o x x . . . . "
    ". . . . . . o x o o o . . . . "
    ". . . . . x o o o x o . . . . "
    ". . . . . . o . x . . x . . . "
    ". . . . . x . o . . . . . . . "
    ". . . . . . x . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ;
  for (int y = 0; y < MaxBS; y++)
    for (int x = 0; x < MaxBS; x++) {
      char  colorchar = boardstr[2 * (x + y * MaxBS)];
      Color color     = C_EMPTY;
      if (colorchar == 'x')
        color = C_BLACK;
      else if (colorchar == 'o')
        color = C_WHITE;
      board[x + y * MaxBS] = color;
    }
  VCFsolver v(MaxBS,MaxBS,DEFAULT_RULE, C_BLACK);
  v.setBoard(board, false,true);
  Loc bestloc;
  int result;
  result = v.fullSearch(1e38,0, bestloc,false);
  cout << "Initial Board:" << endl;
  v.printboard();
  cout <<"Result="<< result << " " << locstr(bestloc) << endl;
  cout << "PV:" << v.getPVlen() << " " << v.getPVreduced();


  return 0;
}
int main(int argc, const char **argv)
{
  //main_testvcf();
  //main_testMCTS(argc, argv);
  return maingtp(argc, argv);
   //return main_testeval();
   //main_testsearch();
  //main_testsearchvct();
   //return main_benchmark();
  //main_validation("D:/gomtrain/export/gomf1.txt", "D:/gomtrain/data/gomf1/vdata.npz");
  return 0;
}