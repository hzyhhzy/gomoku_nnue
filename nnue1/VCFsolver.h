#pragma once
#include "global.h"
namespace VCF {

struct alignas(32) PT
{
  int16_t loc1, loc2;
};
inline Loc xytoshapeloc(int x, int y) { return Loc((BS + 3) * y + x + 3); }

enum PlayResult : int16_t {
  PR_WIN,                  //黑：双四或者连五  白：双四或者连五
  PR_OneFourWithThree,     //黑：单个冲四同时生成眠三 白：无意义
  PR_OneFourWithoutThree,  //黑：单个冲四同时不生成眠三 白：无意义
  PR_NORMAL,               //黑：无意义 白：正常的防守
  PR_LOSE                  //黑：不合法(被抓禁)或者没有生成冲四 白：不合法(被抓禁)
};

const Loc dirs[4] = { Loc(1), Loc(BS + 6), Loc(BS + 6 + 1), Loc(-BS - 6 + 1) };//+x +y +x+y +x-y

}  // namespace VCF

class VCFsolver
{
public:
  const int H, W;

  Color board[(BS + 6) * (BS + 6)];  //预留3圈
  // shape=1*己方棋子+6*长连+256*对方棋子-16384*出界
  // 眠三当且仅当shape=3，冲四当且仅当shape=4，对手冲四当且仅当shape==256*4
  int16_t shape[4][(BS + 6) * (BS + 6)];  //预留3圈

  int     ptNum;             // pts里面前多少个有效
  VCF::PT pts[8 * BS * BS];  //眠三


  VCFsolver();

  //两种board
  //b是外部的棋盘，pla是进攻方
  //katagoType是否是katago的棋盘，false对应loc=x+y*BS，true对应loc=x+1+y*(BS+1)
  void setBoard(Color* b,Color pla,bool katagoType);
  void setBoard2(Color* b);


  //为了方便，假设进攻方是黑，防守方是白。
  VCF::PlayResult playB(Loc loc, Loc &nextForceLoc);
  VCF::PlayResult playW(Loc loc, Loc &nextForceLoc);
  
};
