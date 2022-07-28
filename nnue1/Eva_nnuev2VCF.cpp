#include "Eva_nnuev2VCF.h"

bool Eva_nnuev2VCF::loadParam(std::string filepath)
{
  vcfResult = VCF::SR_Uncertain;
  return Eva_nnuev2::loadParam(filepath);
}

void Eva_nnuev2VCF::clear()
{
  vcfResult = VCF::SR_Uncertain;
  Eva_nnuev2::clear();
  vcfsolver.reset();
}

void Eva_nnuev2VCF::recalculate()
{
  vcfResult = VCF::SR_Uncertain;
  Eva_nnuev2::recalculate();
  vcfsolver.setBoard(board, false, false);
}

void Eva_nnuev2VCF::play(Color color, Loc loc)
{
  vcfResult = VCF::SR_Uncertain;
  Eva_nnuev2::play(color, loc);
  vcfsolver.playOutside(loc, color, 1, false);
}

ValueType Eva_nnuev2VCF::evaluateFull(PolicyType *policy)
{
  if (policy != nullptr) {
    evaluatePolicy(policy);
  }
  return evaluateValue();
}

void Eva_nnuev2VCF::evaluatePolicy(PolicyType *policy)
{
  if (vcfResult == VCF::SR_Uncertain) {
    vcfResult = vcfsolver.fullSearch(1000, 0, vcfWinLoc, false);
  }
  if (vcfResult == VCF::SR_Win) {
    for (Loc loc = 0; loc < BS * BS; ++loc)
      policy[loc] = 0;
    policy[vcfWinLoc] = VCF_POLICY;
  }
  else
    Eva_nnuev2::evaluatePolicy(policy);
}

ValueType Eva_nnuev2VCF::evaluateValue()
{
  if (vcfResult == VCF::SR_Uncertain) {
    vcfResult = vcfsolver.fullSearch(1000, 0, vcfWinLoc, false);
  }
  if (vcfResult == VCF::SR_Win)
    return ValueType(0, -100, -100);
  else
    return Eva_nnuev2::evaluateValue();
}

void Eva_nnuev2VCF::undo(Loc loc)
{
  vcfResult = VCF::SR_Uncertain;
  Eva_nnuev2::undo(loc);
  vcfsolver.undoOutside(loc, 1);
}
