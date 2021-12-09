// SGFprocess.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>

using namespace std;
const unsigned char GAME_START = 226;
const unsigned char GAME_END = 227;
const int VCTside = 1;

uniform_real_distribution<double> randu(0.0, 1.0);
mt19937 randi(114514);	 // 大随机数

string loc2str(unsigned char c)
{
  int x = c % 15, y = c / 15;
  return string(1, x + 'a') + to_string(y + 1);

}

//一局一共n步，取n-randomMovenumReduce步
int randomMovenumReduce()
{
  double c = 15 * (randu(randi) - log(randu(randi) + 1e-8)) + 5;
  int k = c;
  //k = k - ((k +1)% 2);
  return k;
}


bool addOneSgf(string str, ofstream& ofs);//只支持katago的训练谱
void addSgfs(string loadpath, string savepath)
{
  ifstream ifs(loadpath);
  if (!ifs.good())
  {
    cout << "addSgfs :File Not Exist" << endl;
    return;
  }
  ofstream ofs(savepath, ios::binary);
  if (!ofs.good())
  {
    cout << "addSgfs :File Not Able to Save" << endl;
  }
  string oneSgf;
  int count = 0;
  while (getline(ifs, oneSgf, ')'))
  {
    if (oneSgf.size() > 10)
    {
      if (addOneSgf(oneSgf, ofs))
      {
        count++;
        if (count % 10000 == 0)cout << "successfully load sgf " << count << endl;
      }
      else
      {
        count++;
        cout << "failed to load sgf " << count << endl;
      }
    }
  }


}

bool addOneSgf(string str, ofstream& ofs)//只支持katago的训练谱
{
  //格式：
  //GAME_START RESULT AB_SIZE AW_SIZE AB AW 每一步 GAME_END
  //RESULT:0和棋，1黑胜，2白胜
  if (str.size() < 10)return false;
  stringstream ss(str);
  string temp;
  getline(ss, temp, ';');
  if (temp[temp.size() - 1] != '(')return false;

  //读取sgf属性
  char result;
  getline(ss, temp, ';');
  if (temp.size() < 10)return false;
  stringstream infoss(temp);
  string item, value;
  int itemcount = 0;//保证SZ KM RE的顺序

  string ABstr, AWstr, movestr;


  while (getline(infoss, item, '[') && getline(infoss, value, ']'))
  {
    if (item == "SZ")
    {
      itemcount = 1;
      int sgfbs;
      stringstream(value) >> sgfbs;
      if (sgfbs != 15)return false;

    }
    else if (item == "RE")
    {
      if (itemcount != 1)return false;
      itemcount = 2;
      if (value[0] == 'B')
      {
        result = 1;
      }
      else if (value[0] == 'W')
      {
        result = 2;
      }
      else if (value[0] == 'V' || value[0] == '0')
      {
        result = 0;
      }
    }
    else if (item == "AB")
    {
      if (itemcount != 2)return false;
      itemcount = 3;
      if (value.size() != 2)return false;
      int x = value[0] - 'a';
      int y = value[1] - 'a';
      unsigned char c = x + y * 15;
      ABstr.push_back(c);
    }
    else if (item == "AW")
    {
      if (itemcount != 3)return false;
      itemcount = 4;
      if (value.size() != 2)return false;
      int x = value[0] - 'a';
      int y = value[1] - 'a';
      unsigned char c = x + y * 15;
      AWstr.push_back(c);
    }
    else if (item == "")
    {
      if (itemcount == 3)
      {
        if (value.size() != 2)return false;
        int x = value[0] - 'a';
        int y = value[1] - 'a';
        unsigned char c = x + y * 15;
        ABstr.push_back(c);

      }
      else if (itemcount == 4)
      {
        if (value.size() != 2)return false;
        int x = value[0] - 'a';
        int y = value[1] - 'a';
        unsigned char c = x + y * 15;
        AWstr.push_back(c);

      }
      else return false;
    }
  }

  int movenum = ABstr.size() + AWstr.size();
  char nextColor = movenum % 2 ? 'W' : 'B';
  //读棋谱内容
  while (getline(ss, temp, ';'))
  {
    //	s.print();
    if (temp.empty())
    {
      break;
    }
    unsigned char color = temp[0];
    if (color != nextColor)return false;
    movenum++;

    nextColor = movenum % 2 ? 'W' : 'B';
    if (temp[2] == ']')
    {
      return false;
    }
    unsigned char x, y;
    x = temp[2] - 'a';
    y = temp[3] - 'a';
    char c = x + y * 15;
    movestr.push_back(c);


  }

  //和棋不要
  if (result == 0)return true;

  int totalMovenumIncludeOpening = ABstr.size() + AWstr.size() + movestr.size();
  int totalMovenumNoOpening = movestr.size();

  int reduce = randomMovenumReduce();
  if (totalMovenumIncludeOpening % 2 == 0)//最后一手白棋，需要去掉偶数步
    reduce = reduce - reduce % 2;
  else//最后一手黑棋，需要去掉奇数步
    reduce = reduce - (reduce + 1) % 2;

  //随机去掉一部分短的棋谱
  if (reduce > totalMovenumNoOpening)return true;

  int remainMove = totalMovenumNoOpening - reduce;

  string out;

  if ((AWstr.size() != ABstr.size()) && (AWstr.size() != ABstr.size() - 1))
  {
    cout << "AW length not correct" << endl;
    return false;
  }

  for (int i = 0; i < AWstr.size(); i++)
  {
    out.append(loc2str(ABstr[i]));
    out.append(loc2str(AWstr[i]));
  }
  if(AWstr.size() == ABstr.size() - 1)
    out.append(loc2str(ABstr[AWstr.size()]));

  for (int i = 0; i < remainMove; i++)
    out.append(loc2str(movestr[i]));


  /*
  string out;
  out.push_back(GAME_START);
  out.push_back(result);
  out.push_back(char(ABstr.size()));
  out.push_back(char(AWstr.size()));
  out.push_back(char(movestr.size()));
  out = out + ABstr + AWstr + movestr;
  out.push_back(GAME_END);
  ofs << out;
  */
  ofs << out <<"\r\n";
  return true;
}

int main()
{
  addSgfs("all.sgfs", "vctOpeningBlack.txt");

}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
