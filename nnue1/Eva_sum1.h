/*
* 最简单的
* 棋形映射求和
* 长度9
*/


#pragma once
#include "EvaluatorOneSide.h"
const int32_t pow3[] = { 1,3,9,27,81,243,729,2187,6561,19683,59049,177147,531441 };
class Eva_sum1 :
    public EvaluatorOneSide
{
public:
  uint64_t Total_Eval_Num;
  uint32_t shapeTable[4][BS * BS];//4个方向，BS*BS个位置
  int64_t winSum, lossSum, drawSum;
  PolicyType policyBuf[BS * BS];


  /*
  * 棋形编码规则
  * x表示己方，o表示对方，-表示空白，#表示墙，*表示任意
  * 三进制表示，己方1，对方2，空白0
  * 为了方便，以下的数字用三进制表示
  * 编号小的方向（上或者左或者上左或者上右）为低位，例如，x---------- = 00000000001, ----------x = 10000000000
  * 对于右墙，最高位加上几个1，例如，----------# = 100000000000，---------x# = 101000000000，---------## = 110000000000，--------x## = 110100000000
  * 对于左墙，最高位加上2,如果厚度为k，在最低位加上k-1个1，例如，#---------- = 200000000000，#x--------- = 200000000010，##--------- = 200000000001，##x-------- = 200000000101，###x------- = 200000001021
  * 对于两边都有墙，最高位加上3，左边右边都加上“厚度-1”个1，例如，#---------# = 300000000000，##--------# = 300000000001，###------## = 310000000011
  * 以上编码规则可以确保可以通过三进制位运算来实现增量运算。
  * 最大编码为4*3^11=708588
  */
  static const int shapeNum = 708588;
  int32_t weight[4][shapeNum];//4分别是（policy，胜，负，和）
  static const uint32_t Weight_No_Implement = 114514;
  virtual bool loadParam(std::ifstream& fs);
  virtual void clear();
  virtual void recalculate();//根据board完全重新计算棋形表

  //计算拆分为两部分，第一部分是可增量计算的，放在play函数里。第二部分是不易增量计算的，放在evaluate里。
  virtual void play(Color color, Loc loc);
  virtual ValueType evaluate(PolicyType* policy);//policy通过函数参数返回

  virtual void undo(Loc loc);//play的逆过程

  virtual void debug_print();


private:
  void initShapeTable();
  double getWinlossRate();
};

