#include "ExtraStages.h"

ExtraStates::ExtraStates() { 
	H = MaxBS; 
	W = MaxBS;
  basicRule = DEFAULT_RULE;
  initialVCNRule   = Rules::VCNRULE_NOVC;
  maxMoves  = 0;
  firstPassWin = false;

  drawBlackWinlossrate = 0;
  pda                  = 0;


  movenum      = 0;
  blackPassNum = 0;
  whitePassNum = 0;
}

std::vector<float> ExtraStates::getGlobalFeatureInput_States(Color nextPlayer)
{
  std::vector<float> gf(28,0);
  //=katago-10

  double boardArea = H * W;
  gf[0]            = boardArea / 225.0 - 1;
  gf[1]            = sqrt(boardArea / 225.0) - 1;
  gf[2]            = (H - W) * (H - W) / boardArea;


  int myPassNum = nextPlayer == C_WHITE ? whitePassNum : blackPassNum;
  int oppPassNum = nextPlayer == C_BLACK ? whitePassNum : blackPassNum;
  if (initialVCNRule == Rules::VCNRULE_NOVC && !firstPassWin) {
    gf[3] = nextPlayer == C_BLACK  ? drawBlackWinlossrate : -drawBlackWinlossrate;
    gf[4] = oppPassNum>0;
  }

  gf[5] = pda != 0;
  gf[6] = nextPlayer == C_BLACK ? 0.5*pda : -0.5*pda;

  if (firstPassWin) {
    gf[7] = 1.0;
    gf[8] = myPassNum > 0;
    gf[9] = oppPassNum > 0;
  }

  if (initialVCNRule != Rules::VCNRULE_NOVC) {
    static_assert(Rules::VCNRULE_VC4_W == 14);
    Color VCside = 1 + initialVCNRule / 10;
    int VCLevel  = ((VCside == nextPlayer) ? oppPassNum:myPassNum) + initialVCNRule % 10;
    if (VCLevel >= 1 && VCLevel <= 5) {
      if (VCside == nextPlayer)
        gf[9 + VCLevel] = 1.0;
      else
        gf[14 + VCLevel] = 1.0;
    }
    else 
      std::cout << "illegal VCN rule in nninput:" << VCLevel << " " << initialVCNRule
                << std::endl;
  }

  
  if (maxMoves != 0) {
    gf[20]    = 1.0;
    double boardArea = H * W;
    double movenum   = movenum;
    gf[21]    = maxMoves / boardArea;
    gf[22]    = movenum / boardArea;
    gf[23]    = exp(-(maxMoves - movenum) / 50.0);
    gf[24]    = exp(-(maxMoves - movenum) / 15.0);
    gf[25]    = exp(-(maxMoves - movenum) / 5.0);
    gf[26]    = exp(-(maxMoves - movenum) / 1.5);
    gf[27]    = 2 * ((int(maxMoves - movenum)) % 2) - 1;
  }
}
