#pragma once
#include "NNUEglobal.h"
#include <vector>

namespace NNUEV2 {
    const int shapeNum     = 708588;
    const int globalFeatureNum           = 33;
    const int groupSize    = 64;
    const int groupBatch   = groupSize / 16;
    const int featureNum   = groupSize*2;
    const int featureBatch = featureNum / 16;
    const int trunkconv1GroupSize = 4;

    const int mlpChannel = 64;
    const int mlpBatch32 = mlpChannel / 8;

    /*
    �������̣�
    1.��������id��������feature����

    */
    struct ModelWeight
    {
      // 1  mapf = self.mapping(x), shape=H*W*4*2g
      static const int16_t IllegalShapeFeature = 11454;  //>6000
      int16_t mapping[ shapeNum][ featureNum];

      // 2 
      //  g1=mapf[:,:,:self.groupc,:,:]#��һ��ͨ��
      //  g2=mapf[:,:,self.groupc:,:,:]#�ڶ���ͨ��
      //  gfvector=mlp(gf)#gf��ʾ����
      //  �����gfvector��python��4������Ϊ����4��ƽ���ĳ���4�����
      float gfmlp_w1[globalFeatureNum][groupSize];  // shape=(inc��outc)����ͬ��inc��ӦȨ������
      float gfmlp_b1[groupSize];
      float gfmlp_w2[groupSize][groupSize];
      float gfmlp_b2[groupSize];

      // 3  h1 = self.g1lr(g1.mean(1)+gfvector) #��������ټӹ���������leakyrelu
      int16_t g1lr_w[ groupSize];

      // 4  h1 = torch.stack(self.h1conv(h1), dim = 1) #����һ���߾��

      int16_t h1conv_w[(11 + 1) / 2][ groupSize]; //������ǶԳƵģ����Գ�2
      int16_t h1conv_b[ groupSize];

      // 5  h2 = self.h1lr2(self.h1lr1(h1, dim = 2) + g2, dim = 2)
      int16_t h1lr1_w[ groupSize];
      int16_t h1lr2_w[ groupSize];

      // 6  h3 = h2.mean(1) #������������������

      // 7  trunk = self.h3lr(h3) 
      int16_t h3lr_w[ groupSize];
      int16_t h3lr_b[ groupSize]; 

      // 8  trunk = self.trunkconv1(trunk) 
      int16_t trunkconv1_w[ trunkconv1GroupSize][ groupSize];
      int16_t trunkconv1_b[ groupSize];

      // 9  trunk = self.trunklr1(trunk) 
      int16_t trunklr1_w[ groupSize];

      // 10 trunk = self.trunkconv2(trunk)
      int16_t trunkconv2_w[3][ groupSize];//�ԳƵ�3x3���

      // 11 p = self.trunklr2p(trunk) 
      //    v = self.trunklr2v(trunk)
      int16_t trunklr2p_w[ groupSize];
      int16_t trunklr2p_b[ groupSize];
      int16_t trunklr2v_w[ groupSize];
      int16_t trunklr2v_b[ groupSize]; 

      // 12 p = self.policy_linear(p)
      int16_t policy_linear_w[groupSize];
      float   scale_policyInv;

      // 13  v=v.mean((2,3))
      float scale_beforemlpInv;
      float valuelr_w[ groupSize];
      float valuelr_b[ groupSize]; 

      // 14  mlp
      float mlp_w1[ groupSize]
                  [ mlpChannel];  // shape=(inc��outc)����ͬ��inc��ӦȨ������
      float mlp_b1[ mlpChannel];
      float mlp_w2[ mlpChannel][ mlpChannel];
      float mlp_b2[ mlpChannel];
      float mlp_w3[ mlpChannel][ mlpChannel];
      float mlp_b3[ mlpChannel];
      float mlpfinal_w[ mlpChannel][4];
      float mlpfinal_w_for_safety[4];  // mlp_w3��read��ʱ��һ��read
                                   // 8������read�������ڴ�mlp_w3[mix6::valueNum-1][3]+4��
      float mlpfinal_b[4];
      float mlpfinal_b_for_safety[4];  // mlp_b3��read��ʱ��һ��read
                                   // 8������read�������ڴ�mlp_b3[3]+4��

      bool loadParam(std::string filename);
    };
    struct OnePointChange
    {
      NU_Loc      loc;
      int16_t  dir;
      uint32_t oldshape;
      uint32_t newshape;
    };
    struct ModelBuf
    {
      // 1 convert board to shape
      uint32_t shapeTable[MaxBS * MaxBS][4];  // 4������MaxBS*MaxBS��λ��

      // 2  shape��vector  g1������ȡ��ֻ����g2
      int16_t g2[MaxBS * MaxBS][4][groupSize];

      // 3  g1sum=g1.sum(1), shape=H*W*g
      int16_t g1sum[MaxBS * MaxBS][groupSize];

      // 4  h1=self.g1lr(g1sum), shape=HWc
      //int16_t h1[MaxBS * MaxBS][groupSize];
      int16_t h1m[(MaxBS + 10) * (MaxBS + 10) * 6 * 16];//ֻ�ǿ���һ��ռ䣬����Ƶ��new/delete��ÿ���õ�ʱ������

      // ����Ĳ��ּ���û����������


      // valueͷ��policyͷ����trunk������Ҳ���ڻ�����
      bool    trunkUpToDate;
      int16_t trunk[MaxBS * MaxBS][groupSize];  

      void update(Color oldcolor, Color newcolor, NU_Loc loc, const ModelWeight &weights);

      void emptyboard(const ModelWeight &weights);  // init
    };

}  // namespace NNUEV2
class Eva_nnuev2
{
public:
  Color mySide;  //�����ֵ�����ıߵġ�������һ����˭�ߵģ���������һ��ΪmySide�Ĺ�ֵ
  Color board[MaxBS * MaxBS];

  uint64_t         TotalEvalNum;
  NNUEV2::ModelWeight weights;
  NNUEV2::ModelBuf buf;

  bool loadParam(std::string filepath);
  void clear();
  void recalculate();  //����board��ȫ���¼������α�

  //������Ϊ�����֣���һ�����ǿ���������ģ�����play������ڶ������ǲ�����������ģ�����evaluate�
  void      play(Color color, NU_Loc loc);
  NNUE::ValueType evaluateFull(const float *gf, NNUE::PolicyType *policy);  // policyͨ��������������
  void evaluatePolicy(const float *gf, NNUE::PolicyType *policy);  // policyͨ��������������
  NNUE::ValueType evaluateValue(const float *gf);                //

  void undo(NU_Loc loc);  // play�������

  void debug_print();

private:
  void calculateTrunk(const float *gf);
};
