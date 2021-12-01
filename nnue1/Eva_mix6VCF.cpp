#include "Eva_mix6VCF.h"

bool Eva_mix6VCF::loadParam(std::string filepath)
{
  vcfResult = VCF::SR_Uncertain;
  return Eva_mix6_avx2::loadParam(filepath);
}

void Eva_mix6VCF::clear()
{
  vcfResult = VCF::SR_Uncertain;
  Eva_mix6_avx2::clear();
  vcfsolver.reset();
}

void Eva_mix6VCF::recalculate()
{
  vcfResult = VCF::SR_Uncertain;
  Eva_mix6_avx2::recalculate();
  vcfsolver.setBoard(board, false, false);
}

void Eva_mix6VCF::play(Color color, Loc loc)
{
  vcfResult = VCF::SR_Uncertain;
  Eva_mix6_avx2::play(color, loc);
  vcfsolver.playOutside(loc, color, 1, false);

}

ValueType Eva_mix6VCF::evaluateFull(PolicyType* policy)
{
  if (policy != nullptr) {
    evaluatePolicy(policy);
  }
  return evaluateValue();
}

void Eva_mix6VCF::evaluatePolicy(PolicyType* policy)
{
  if (vcfResult == VCF::SR_Uncertain) {
    vcfResult = vcfsolver.fullSearch(1000, 0,vcfWinLoc, false);
  }
  if (vcfResult == VCF::SR_Win)
  {
    for (Loc loc = 0; loc < BS * BS; ++loc)policy[loc] = 0;
    policy[vcfWinLoc] = VCF_POLICY;
  }
  else Eva_mix6_avx2::evaluatePolicy(policy);
}

ValueType Eva_mix6VCF::evaluateValue()
{
  if (vcfResult == VCF::SR_Uncertain) {
    vcfResult = vcfsolver.fullSearch(1000,0, vcfWinLoc, false);
  }
  if (vcfResult == VCF::SR_Win)return ValueType(0, -100, -100);
  else return Eva_mix6_avx2::evaluateValue();
}

void Eva_mix6VCF::undo(Loc loc)
{
  vcfResult = VCF::SR_Uncertain;
  Eva_mix6_avx2::undo( loc);
  vcfsolver.undoOutside(loc,1);
}
