#pragma once
#include "NNUEglobal.h"
class ExtraStates //including rules,passes,movenums...
{
public:
  //rules
  int H;
  int W;
  int   basicRule;
  int initialVCNRule;
  int  maxMoves;
  bool firstPassWin;

  float drawBlackWinlossrate;
  float pda;

  //stages
  int movenum;
  int blackPassNum;
  int whitePassNum;

  
  ExtraStates();
  void getGlobalFeatureInput_States(float* gf, Color nextPlayer);  // 3�����̴�С���gf[0:3]+katago��gf��13��37 gf[8:33]����ʣ��VCF��gf[3:8](katago��gf��8��12)

};
