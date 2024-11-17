
#include "validation.h"


#include "external/cnpy/cnpy.h"
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <cmath>

#include "Eva_nnuev2.h"

using namespace NNUE;
using namespace std;
void loss_oneSample(Eva_nnuev2 *eva,
                    double      *totalVloss,
                    double      *totalPloss,
                    const float *bf,
                    const float *gf,
                    const float *pt,
                    const float *vt)
{
  //set board
  for (NU_Loc loc = 0; loc < MaxBS * MaxBS; loc++) {
    float bf0 = bf[loc + MaxBS * MaxBS];
    float bf1 = bf[loc + 2 * MaxBS*MaxBS];
    Color c   = C_WALL;
    if (bf0 == 0 && bf1 == 0) {
      c = C_EMPTY;
    }
    else if (bf0 == 1 && bf1 == 0) {
      c = C_MY;
    }
    else if (bf0 == 0 && bf1 == 1) {
      c = C_OPP;
    }
    else {
      cout << "ERROR bf0=" << bf0 << " bf1=" << bf1 << endl;
    }
    Color oldc = eva->board[loc];
    if (oldc != c) {
      if (oldc != C_EMPTY)
        eva->undo(loc);
      if (c != C_EMPTY)
        eva->play(c,loc);
    }
  }

  //illegal map
  bool illegalMap[MaxBS * MaxBS];
  for (int i = 0; i < MaxBS * MaxBS; i++)
    illegalMap[i] = false;
  if (gf[3] == 1)
  {
    for (int i = 0; i < MaxBS * MaxBS; i++)
    {
      float legal = bf[4 * MaxBS * MaxBS + i];
      if (legal == 0)
        illegalMap[i] = true;

    }

  }


  PolicyType policy_int[MaxBS * MaxBS + 1];
  ValueType  value = eva->evaluateFull(gf, illegalMap ,policy_int);
  double      policy[MaxBS * MaxBS + 1];
  for (NU_Loc loc = 0; loc < MaxBS * MaxBS + 1; loc++) {
    policy[loc] = policy_int[loc] / policyQuantFactor;
  }

  //ploss---------------------------------------------------------------------------------------------

  //output logsoftmax
  double policyMax = -1e30;
  for (NU_Loc loc = 0; loc < MaxBS * MaxBS + 1; loc++) {
    if (policy[loc] > policyMax)
      policyMax = policy[loc];
  }
  double policyTotal=0;
  for (NU_Loc loc = 0; loc < MaxBS * MaxBS + 1; loc++) {
    policyTotal+=exp(policy[loc]-policyMax);
  }
  policyTotal = log(policyTotal);
  for (NU_Loc loc = 0; loc < MaxBS * MaxBS + 1; loc++) {
    policy[loc] -= (policyTotal+policyMax);
  }

  //pt sum=1
  double ptTotal = 0;
  for (NU_Loc loc = 0; loc < MaxBS * MaxBS + 1; loc++) {
    ptTotal += double(pt[loc]);
  }
  //cout << ptTotal << endl;
  double pt_n[MaxBS * MaxBS + 1];
  for (NU_Loc loc = 0; loc < MaxBS * MaxBS + 1; loc++) {
    pt_n[loc] = pt[loc]/ptTotal;
  }

  //if (pt_n[MaxBS * MaxBS] > 0.3 || policy[MaxBS * MaxBS] > -1)
  //  cout << pt_n[MaxBS * MaxBS] << " " << policy[MaxBS * MaxBS] << endl;

  //cross entropy loss
  double ploss = 0;
  for (NU_Loc loc = 0; loc < MaxBS * MaxBS + 1; loc++) {
    double a = policy[loc];
    double b = pt_n[loc];
    ploss += (-a * b + b * log(b + 1e-30));
  }

  totalPloss[0] += ploss;

  // vloss---------------------------------------------------------------------------------------------
  double value_n[3] = {value.win, value.loss, value.draw};
  double vt_n[3]    = {vt[0], vt[1], vt[2]};
  double vloss      = 0;
  for (int i = 0; i < 3; i++) {
    double a = value_n[i];
    double b = vt_n[i];
    a        = log(a + 1e-30);
    vloss += (-a * b + b * log(b + 1e-30));
  }
  totalVloss[0] += vloss;

  //cout << ploss << " " << vloss << endl;
}

void  main_validation(string modelpath, string datapath)
{
  Eva_nnuev2 *eva = new Eva_nnuev2();
  eva->loadParam(modelpath);

  cnpy::npz_t npz = cnpy::npz_load(datapath);

  // check that the loaded myVar1 matches myVar1
  cnpy::NpyArray bf_npy = npz["bf"];
  cnpy::NpyArray gf_npy = npz["gf"];
  cnpy::NpyArray pt_npy = npz["pt"];
  cnpy::NpyArray vt_npy = npz["vt"];
  assert(bf_npy.shape.size() == 4);
  int            N      = bf_npy.shape[0];
  assert(bf_npy.shape[1] == 5);
  assert(bf_npy.shape[2] == MaxBS);
  assert(bf_npy.shape[3] == MaxBS);
  assert(gf_npy.shape.size() == 2);
  assert(gf_npy.shape[0] == N);
  assert(gf_npy.shape[1] == 39);
  assert(pt_npy.shape.size() == 2);
  assert(pt_npy.shape[0] == N);
  assert(pt_npy.shape[1] == MaxBS * MaxBS + 1);
  assert(vt_npy.shape.size() == 2);
  assert(vt_npy.shape[0] == N);
  assert(vt_npy.shape[1] == 3);
  cout << "num=" << N << endl;
  float *bf = bf_npy.data<float>();
  float *gf = gf_npy.data<float>();
  float *pt = pt_npy.data<float>();
  float *vt = vt_npy.data<float>();

  int bfsize = 5 * MaxBS * MaxBS;
  int gfsize = 39;
  int ptsize = MaxBS * MaxBS + 1;
  int vtsize = 3;

  double totalVloss = 0;
  double totalPloss = 0;
  int validNum = 0;
  for (int i = 0; i < N; i++) {
    /*
    float bf1[2 * MaxBS * MaxBS];
    for (int d1 = 0; d1 < 2; d1++) {
      for (int d2 = 0; d2 < MaxBS; d2++) {
        for (int d3 = 0; d3 < MaxBS; d3++) {
          bf1[d1 * MaxBS * MaxBS + d2 * MaxBS + d3] =
              // bf[d3 * MaxBS * 2 * N + d2 * 2 * N + d1 * N + i];
              bf[d1 * MaxBS * MaxBS + d2 * MaxBS + d3 + i * MaxBS * MaxBS * 2];
        }
      }
    }
    float pt1[MaxBS * MaxBS + 1];
    for (int d1 = 0; d1 < MaxBS*MaxBS+1; d1++) {
      pt1[d1] = pt[d1 * N + i];
    }*/

    float gf1[NNUEV2::globalFeatureNum];
    gf1[0] = MaxBS * MaxBS / 225.0 - 1;
    gf1[1] = sqrt(MaxBS * MaxBS / 225.0) - 1;
    gf1[2] = 0;
    for (int d1 = 0; d1 < 39; d1++) {
      gf1[d1 + 3] = gf[d1 * N + i];
    }

    bool not19x19 = false;
    for (int j = 0; j < MaxBS * MaxBS; j++) {
      if (bf[i * bfsize + j] != 1)//not 19x19
      {
        not19x19 = true;
      }
    }
    if (not19x19)continue;


    validNum += 1;
    float vt1[3];
    for (int d1 = 0; d1 < 3; d1++) {
      vt1[d1] = vt[d1 * N + i];
    }
    loss_oneSample(eva,
                   &totalVloss,
                   &totalPloss, 
                   bf + i * bfsize,
                   gf1,
                   pt + i * ptsize,
                   vt1);
  }
  cout << "validNum=" << validNum << endl;
  cout << "ploss=" << totalPloss / validNum << endl;
  cout << "vloss=" << totalVloss / validNum << endl;
}
