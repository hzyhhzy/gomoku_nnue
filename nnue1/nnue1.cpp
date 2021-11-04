// nnue1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "Engine.h"
#include "AllSearch.h"
#include "AllEvaluator.h"
using namespace std;
int main()
{
  Eva_mix6_avx2* eva = new Eva_mix6_avx2();
  eva->loadParam("weights/t5.txt");
  eva->debug_print();

  const char boardstr[] = ""
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . x . . . . . . "
    ". . . . . . . x . . . . . . . "
    ". . . . . . o . o . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . "
    ". . . . . . . . . . . . . . . ";
  for (int y = 0; y < BS; y++)
    for (int x = 0; x < BS; x++)
    {
      char colorchar = boardstr[2 * (x + y * BS)];
      Color color = C_EMPTY;
      if (colorchar == 'x')color = C_BLACK;
      else if (colorchar == 'o')color = C_WHITE;
      if (color != C_EMPTY)eva->play(color, x + y * BS);
    }


  PolicyType policy[BS * BS];
  ValueType value=eva->evaluate(policy);

  cout << "WR " << (value.win + 0.5 * value.draw) << endl;
  cout << "Win " << value.win  << endl;
  cout << "Loss " << value.loss  << endl;
  cout << "Draw " << value.draw  << endl;

  for (int y = 0; y < BS; y++)
  {
    for (int x = 0; x < BS; x++)
    {
      Color c = eva->board[y * BS + x];
      if (c == C_EMPTY)cout << ".\t";
      else if (c == C_MY)cout << "x\t";
      else if (c == C_OPP)cout << "o\t";
    }
    cout << endl;
    for (int x = 0; x < BS; x++)
    {
      int p = policy[y * BS + x];
      cout << p << "\t";
    }
    cout << endl;
  }

  eva->debug_print();

}
int test1()
{
  Evaluator* eva = new Evaluator("mix6", "weights/t5.txt");
  ABsearch* search = new ABsearch(eva);
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
    ". . . . . x o x . . . . . . . "
    ". . . . x o o . x . . . . . . "
    ". . . . . o x . . . . . . . . "
    ". . . . o . . . . . . . . . . "
    ". . . . . . . x . . . . . . . "
    ". . . . . . . . . . . . . . . ";
  for (int y = 0; y < BS; y++)
    for (int x = 0; x < BS; x++)
    {
      char colorchar = boardstr[2 * (x + y * BS)];
      Color color = C_EMPTY;
      if (colorchar == 'x')color = C_BLACK;
      else if (colorchar == 'o')color = C_WHITE;
      if(color!=C_EMPTY)eva->play(color,x+y*BS);
    }




  for (int depth = 0; depth < 10; depth++)
  {
    Loc loc;
    double value = search->fullsearch(C_WHITE, depth, loc);
    //search->evaluator->recalculate();
    cout << "Depth = " <<depth<< " Value = " << value << " Bestloc = " << char('A'+loc % BS) << 15-loc / BS << endl;
  }


  return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
