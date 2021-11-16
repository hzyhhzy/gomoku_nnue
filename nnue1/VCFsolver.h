#pragma once
#include "global.h"
namespace VCF {

  struct alignas(32) PT
  {
    Loc loc1, loc2;
    PT():loc1(LOC_NULL),loc2(LOC_NULL){}
    PT(Loc loc1,Loc loc2):loc1(loc1),loc2(loc2){}
  };
  inline Loc xytoshapeloc(int x, int y) { return Loc((BS + 3) * (y + 3) + x + 3); }

  enum PlayResult : int16_t {
    PR_Win,                  //双四或者连五或者抓禁
    PR_OneFourWithThree,     //单个冲四同时生成眠三
    PR_OneFourWithoutThree,  //单个冲四同时不生成眠三
    PR_Lose                  //不合法(被抓禁)或者没有生成冲四
  };
  enum SearchResult : int16_t {
    SR_Win = 0,//有解
    SR_Lose = 32767//确定无解
    //其他值代表暂时无解，若SearchResult=n，说明“允许n-1次无新眠三的冲四无解”
  };


  static const Loc dirs[4] = { 1, BS + 6, BS + 6 + 1, -BS - 6 + 1 };//+x +y +x+y +x-y
  //为了方便常量调用，分开再写一遍
  static const Loc dir0 = 1;
  static const Loc dir1 =  BS + 6;
  static const Loc dir2 =  BS + 6 + 1;
  static const Loc dir3 = -BS - 6 + 1;
  static const int ArrSize = (BS + 6) * (BS + 6);//考虑额外3圈后的棋盘格点数


}  // namespace VCF

class VCFsolver
{
public:
  const int H, W;
  const bool isWhite;//如果进攻方是黑棋，则false，进攻方是白棋则true。若true，color全是反向

  Color board[(BS + 6) * (BS + 6)];  //预留3圈
  // shape=1*己方棋子+8*长连+64*对方棋子+512*对手长连+4096*出界
  int16_t shape[4][(BS + 6) * (BS + 6)];  //预留3圈

  int     ptNum;             // pts里面前多少个有效
  VCF::PT pts[8 * BS * BS];  //眠三

  int nodeNumThisSearch;//记录这次搜索已经搜索了多少节点了，用来提前终止，每次fullSearch开始时清零

  VCFsolver(Color pla) :VCFsolver(BS, BS,pla) {}
  VCFsolver(int h, int w,Color pla);
  void reset();

  //两种board
  //b是外部的棋盘，pla是进攻方
  //katagoType是否是katago的棋盘，false对应loc=x+y*BS，true对应loc=x+1+(y+1)*(BS+1)
  void setBoard(Color* b, bool katagoType);


  VCF::SearchResult fullSearch(float factor, Loc& bestmove);//factor是搜索因数，保证factor正比于节点数，具体如何实现还未想好。
  void playOutside(Loc loc, Color color, int locType);//用于外部调用，更新棋盘。保证shape正确，不保证pts正确。
  //color是外界color（C_BLACK/WHITE），不是C_MY/OPP
  //loctype=0是vcf求解器的loc，=1是y*BS+x，=2是katago

private:

  void resetPts();//重置pts，完整搜索前一定调用这个

  //找两个空点
  VCF::PT findEmptyPoint2(Loc loc,Loc bias);
  //找一个空点
  Loc findEmptyPoint1(Loc loc,Loc bias);

  //走一个回合，先冲四再防守
  //loc1是攻，loc2是守，nextForceLoc是白棋冲四
  VCF::PlayResult playTwo(Loc loc1,Loc loc2, Loc& nextForceLoc);

  VCF::SearchResult search(int maxNoThree, Loc forceLoc, Loc& winLoc);//最多maxNoThree步没有新眠三，挡对手反四的冲四不算

  // shape=1*己方棋子+8*长连+64*对方棋子+512*对手长连+4096*出界
  inline bool shape_isMyFour(int16_t s) { return (s & 0170777) == 4; }
  inline bool shape_isMyThree(int16_t s) { return (s & 0170777) == 3; }
  inline bool shape_isOppFour(int16_t s) { return (s & 0177707) == 4*64; }
};
