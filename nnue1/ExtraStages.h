#pragma once
#include "global.h"
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
  std::vector<float> getGlobalFeatureInput_States(Color nextPlayer); // 3�����̴�С���+katago��gf��13��37

};
