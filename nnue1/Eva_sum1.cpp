#include "Eva_sum1.h"

bool Eva_sum1::loadParam(std::string filepath)
{
  using namespace std;
  FILE* fp = fopen(filepath.c_str(), "r");
  if (fp == NULL)cout << "Bad file:" << filepath << endl;


  Total_Eval_Num = 0;
  for (int i = 0; i < shapeNum; i++)
  {
    for (int j = 0; j < 4; j++)weight[j][i] = Weight_No_Implement;
  }
  int id;
  float buf;
  //cout << "1";
  while (fscanf(fp, "%d", &id) != EOF)
  {
    //cout << id<<" ";
    //std::cout << "a";
    for (int j = 0; j < 4; j++) {
      fscanf(fp, "%f", &buf);
      //cout << buf << " ";
      weight[j][id] = buf * quantFactor;
    }
    //std::cout << id << " "<< weight[0][id]<<std::endl;
  }
  //已经有子的地方，policy设为MIN_POLICY
  for (int i = 0; i < shapeNum; i++)//
  {
    if (((i / 243) % 3 != 0) && weight[0][i] != Weight_No_Implement)//有子且合法的棋形
      weight[0][i] = MIN_POLICY;

  }
  clear();
  return true;
}

void Eva_sum1::clear()
{
  for (int i = 0; i < BS * BS; i++)board[i] = C_EMPTY;
  initShapeTable();

  winSum = 0, lossSum = 0, drawSum = 0;
  for (int i = 0; i < BS * BS; i++)policyBuf[i] = 0;
  for (int dir = 0; dir < 4; dir++)
  {
    for (int loc = 0; loc < BS * BS; loc++)
    {
      //std::cout << winSum << " " << lossSum << std::endl;
      int shape = shapeTable[dir][loc];
      policyBuf[loc] += weight[0][shape];
      winSum += weight[1][shape];
      lossSum += weight[2][shape];
      drawSum += weight[3][shape];
    }
  }
}

void Eva_sum1::recalculate()
{
  Color boardCopy[BS * BS];
  memcpy(boardCopy, board, BS * BS * sizeof(Color));
  clear();
  for (Loc i = ZERO_LOC; i < BS * BS; ++i)
  {
    if (boardCopy[i] != C_EMPTY)play(boardCopy[i], i);
  }
}

void Eva_sum1::play(Color color, Loc loc)
{
  board[loc] = color;
  auto affectOnOnePoint = [&](int loc, int dir, int dist)//dist大于0是正方向，小于0是负方向
  {
    int oldShape = shapeTable[dir][loc];
    int newShape = oldShape + color * pow3[dist + 5];
    shapeTable[dir][loc] = newShape;
    if (weight[0][newShape] == Weight_No_Implement)std::cout << "Weight No implement "<<newShape<<std::endl;//debug
    policyBuf[loc] = policyBuf[loc] - weight[0][oldShape] + weight[0][newShape];
    winSum = winSum - weight[1][oldShape] + weight[1][newShape];
    lossSum = lossSum - weight[2][oldShape] + weight[2][newShape];
    drawSum = drawSum - weight[3][oldShape] + weight[3][newShape];
  };
  int x0 = loc % BS;
  int y0 = loc / BS;

  int dxs[4] = { 1,0,1,1 };
  int dys[4] = { 0,1,1,-1 };

  for (int dir = 0; dir < 4; dir++)
  {
    for (int dist = -5; dist <= 5; dist++)
    {
      int x = x0 - dist * dxs[dir];
      int y = y0 - dist * dys[dir];
      if (x < 0 || x >= BS || y < 0 || y >= BS)continue;
      affectOnOnePoint(x + BS * y, dir, dist);
    }
  }
}

ValueType Eva_sum1::evaluate(PolicyType* policy)
{
  Total_Eval_Num++;
  //if (Total_Eval_Num % 100000 == 0)std::cout << "TotalEval " << Total_Eval_Num << std::endl;
  if (policy != nullptr)
  {
    memcpy(policy, policyBuf, sizeof(policyBuf));
  }
  float factor = BS * BS * quantFactor;
  return ValueType(float(winSum)/factor, float(lossSum)/factor, float(drawSum)/factor);
}

void Eva_sum1::undo(Loc loc)
{
  Color color = board[loc];
  board[loc] = C_EMPTY;
  auto affectOnOnePoint = [&](int loc, int dir, int dist)//dist大于0是正方向，小于0是负方向
  {
    int oldShape = shapeTable[dir][loc];
    int newShape = oldShape - color * pow3[dist + 5];
    shapeTable[dir][loc] = newShape;
    policyBuf[loc] = policyBuf[loc] - weight[0][oldShape] + weight[0][newShape];
    winSum = winSum - weight[1][oldShape] + weight[1][newShape];
    lossSum = lossSum - weight[2][oldShape] + weight[2][newShape];
    drawSum = drawSum - weight[3][oldShape] + weight[3][newShape];
  };
  int x0 = loc % BS;
  int y0 = loc / BS;

  int dxs[4] = { 1,0,1,1 };
  int dys[4] = { 0,1,1,-1 };

  for (int dir = 0; dir < 4; dir++)
  {
    for (int dist = -5; dist <= 5; dist++)
    {
      int x = x0 - dist * dxs[dir];
      int y = y0 - dist * dys[dir];
      if (x < 0 || x >= BS || y < 0 || y >= BS)continue;
      affectOnOnePoint(x + BS * y, dir, dist);
    }
  }
}

void Eva_sum1::initShapeTable()
{
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < BS * BS; j++)
      shapeTable[i][j] = 0;

  //以下不可交换次序，因为存在覆盖
  
  //正方向的墙（右墙，下墙，右下墙，右上墙）
  
  for (int thick = 1; thick <= 5; thick++)
  {
    for (int i = 0; i < BS; i++)
    {
      int c = 0;
      for (int j = 0; j < thick; j++)c += pow3[11 - j];
      shapeTable[0][(BS - 6+thick) + i * BS] = c;//右墙
      shapeTable[1][i + (BS - 6 + thick) * BS] = c;//下墙
      shapeTable[2][(BS - 6 + thick) + i * BS] = c;//右下墙靠右
      shapeTable[2][i + (BS - 6 + thick) * BS] = c;//右下墙靠下
      shapeTable[3][(BS - 6 + thick) + i * BS] = c;//右上墙靠右
      shapeTable[3][i + (6-1-thick) * BS] = c;//右下墙靠上
    }
  }

  //负方向的墙（左墙，上墙，左上墙，左下墙）

  //厚度1 
  for (int thick = 1; thick <= 5; thick++)
  {
    for (int i = 0; i < BS; i++)
    {
      int c = 2*pow3[11];//3进制2000000000
      for (int j = 0; j < thick-1; j++)c += pow3[j];
      shapeTable[0][(6-1-thick) + i * BS] = c;//左墙
      shapeTable[1][i + (6 - 1 - thick) * BS] = c;//上墙
      shapeTable[2][(6 - 1 - thick) + i * BS] = c;//左上墙靠左
      shapeTable[2][i + (6 - 1 - thick) * BS] = c;//左上墙靠上
      shapeTable[3][(6 - 1 - thick) + i * BS] = c;//左下墙靠左
      shapeTable[3][i + (BS - 6 + thick) * BS] = c;//左下墙靠下
    }
  }

  //两边都有墙

  for (int a = 1; a <= 5; a++)//正方向墙厚
    for (int b = 1; b <= 5; b++)//负方向墙厚
    {
      int c = 3 * pow3[11];
      for (int i = 0; i < a - 1; i++)c += pow3[10 - i];
      for (int i = 0; i < b - 1; i++)c += pow3[i];
      shapeTable[2][(BS - 6 + a) + (5 - b) * BS] = c;//右上角
      shapeTable[2][(BS - 6 + a) * BS + (5 - b)] = c;//左下角
      shapeTable[3][(5 - b) + (5 - a) * BS] = c;//左上角
      shapeTable[3][(BS-6+ a) + (BS - 6 + b) * BS] = c;//右下角

    }
}

double Eva_sum1::getWinlossRate()
{
  //softmax
  double maxValue = winSum > lossSum ? winSum : lossSum;
  maxValue = maxValue > drawSum ? maxValue : drawSum;
  double factor = BS * BS * quantFactor;
  double win = exp((winSum - maxValue) / factor);
  double loss = exp((lossSum - maxValue) / factor);
  double draw = exp((drawSum - maxValue) / factor);
  return (win-loss)/(win+loss+draw);
}
using namespace std;
void Eva_sum1::debug_print()
{
  for (int y = 0; y < BS; y++)
  {
    for (int x = 0; x < BS; x++)
    {
      Color c = board[y * BS + x];
      if (c == C_EMPTY)cout << ". ";
      else if (c == C_MY)cout << "x ";
      else if (c == C_OPP)cout << "o ";
    }
    cout << endl;
  }



  double maxValue = winSum > lossSum ? winSum : lossSum;
  maxValue = maxValue > drawSum ? maxValue : drawSum;
  double factor = BS * BS * quantFactor;
  double win = exp((winSum - maxValue) / factor);
  double loss = exp((lossSum - maxValue) / factor);
  double draw = exp((drawSum - maxValue) / factor);
  double sum = win + loss + draw;
  cout << "WR " << (win+0.5*draw) / sum << endl;
  cout << "Win " << win / sum << endl;
  cout << "Loss " << loss/sum << endl;
  cout << "Draw " << draw/sum<< endl;

  for (int y = 0; y < BS; y++)
  {
    for (int x = 0; x < BS; x++)
    {
      Color c = board[y * BS + x];
      if (c == C_EMPTY)cout << ".\t";
      else if (c == C_MY)cout << "x\t";
      else if (c == C_OPP)cout << "o\t";
    }
    cout << endl;
    for (int x = 0; x < BS; x++)
    {
      int p = policyBuf[y * BS + x];
      cout << p << "\t";
    }
    cout << endl;
  }




  /*
  auto print3 = [&](int a)//10位三进制
  {
    int buf[10];
    for (int i = 0; i < 9; i++)
    {
      buf[9 - i] = a % 3;
      a = a / 3;
    }
    buf[0] = a;
    for (int i = 0; i < 10; i++)std::cout << buf[i];
    std::cout << " ";
  };
  
  for (int dir = 0; dir < 4; dir++)
  {
    std::cout << "dir=" << dir << std::endl;
    for (int y = 0; y < BS; y++)
    {
      for (int x = 0; x < BS; x++)
      {
        print3(shapeTable[dir][x + y * BS]);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  */
}
