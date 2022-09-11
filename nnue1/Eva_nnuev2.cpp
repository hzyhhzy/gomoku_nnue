#include "Eva_nnuev2.h"

#include "external/simde_avx2.h"
#include "external/simde_fma.h"

#include <filesystem>
using namespace NNUEV2;

void ModelBuf::update(Color                   oldcolor,
                           Color                   newcolor,
                           Loc                     loc,
                           const ModelWeight &weights)
{
  trunkUpToDate = false;

  // update shapeTable
  std::vector<OnePointChange> changeTable(44);
  int                         changenum = 0;

  {
    int x0 = loc % MaxBS;
    int y0 = loc / MaxBS;

    int dxs[4] = {1, 0, 1, 1};
    int dys[4] = {0, 1, 1, -1};

    for (int dir = 0; dir < 4; dir++) {
      for (int dist = -5; dist <= 5; dist++) {
        int x = x0 - dist * dxs[dir];
        int y = y0 - dist * dys[dir];
        if (x < 0 || x >= MaxBS || y < 0 || y >= MaxBS)
          continue;
        OnePointChange c;
        c.dir = dir, c.loc = MakeLoc(x, y);
        c.oldshape             = shapeTable[c.loc][dir];
        c.newshape             = c.oldshape + (newcolor - oldcolor) * pow3[dist + 5];
        shapeTable[c.loc][dir] = c.newshape;
        changeTable[changenum] = c;
        changenum++;
      }
    }
  }

  for (int p = 0; p < changenum; p++) {
    OnePointChange c = changeTable[p];

    int y0 = c.loc / MaxBS, x0 = c.loc % MaxBS;

    for (int i = 0; i < groupBatch; i++) {

      // g2 update
      auto neww = simde_mm256_loadu_si256(weights.mapping[c.newshape] + i * 16 + groupSize);
      simde_mm256_storeu_si256(g2[c.loc][c.dir] + i * 16, neww);

      // g1 update
      auto  oldw = simde_mm256_loadu_si256(weights.mapping[c.oldshape] + i * 16);
      neww = simde_mm256_loadu_si256(weights.mapping[c.newshape] + i * 16);
      void *wp   = g1sum[c.loc] + i * 16;
      auto  sumw = simde_mm256_loadu_si256(wp);
      sumw       = simde_mm256_sub_epi16(sumw, oldw);
      sumw       = simde_mm256_add_epi16(sumw, neww);
      simde_mm256_storeu_si256(wp, sumw);



    }
  }
}

//����������pythonѵ��������ͬ���Ķ�ʱ������python�������
void Eva_nnuev2::calculateTrunk()
{
  for (int batch = 0; batch < groupBatch; batch++) {  //һֱ��trunk������ϣ���ͬbatch֮�䶼û�н���,���Է��������
    int addrBias = batch * 16;

    //�������̫�󣬾Ͳ�ֱ��int16_t[(MaxBS + 10) * (MaxBS + 10)][6][16]��
    int16_t *h1m = new int16_t[(MaxBS + 10) * (MaxBS + 10)*6*16];  //�����ľ�����ȳ�����ӣ��˴�����˵���û��ӡ�h1m��һ������ӵõ�h1c������5��padding�����������
    memset(h1m, 0, sizeof(int16_t) * (MaxBS + 10) * (MaxBS + 10) * 6 * 16);
    //-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // g1 prelu��h1conv�ĳ˷�����
    auto g1lr_w   = simde_mm256_loadu_si256(weights.g1lr_w + addrBias);
    auto h1conv_w0 = simde_mm256_loadu_si256(weights.h1conv_w[0] + addrBias);
    auto h1conv_w1 = simde_mm256_loadu_si256(weights.h1conv_w[1] + addrBias);
    auto h1conv_w2 = simde_mm256_loadu_si256(weights.h1conv_w[2] + addrBias);
    auto h1conv_w3 = simde_mm256_loadu_si256(weights.h1conv_w[3] + addrBias);
    auto h1conv_w4 = simde_mm256_loadu_si256(weights.h1conv_w[4] + addrBias);
    auto h1conv_w5 = simde_mm256_loadu_si256(weights.h1conv_w[5] + addrBias);
    for (Loc locY = 0; locY < MaxBS; locY++) {
      for (Loc locX = 0; locX < MaxBS; locX++) {
        Loc loc1 = locY * MaxBS + locX;             //ԭʼloc
        Loc loc2 = (locY + 5)  * (MaxBS + 10) + locX + 5;  // padding���loc
        int16_t *h1mbias = h1m + loc2 * 6 * 16;

        auto g1sum = simde_mm256_loadu_si256(buf.g1sum[loc1] + addrBias);
        auto h1 = simde_mm256_max_epi16(g1sum, simde_mm256_mulhrs_epi16(g1sum, g1lr_w));
        simde_mm256_storeu_si256(h1mbias + 0 * 16,
                                 simde_mm256_mulhrs_epi16(h1, h1conv_w0));
        simde_mm256_storeu_si256(h1mbias + 1 * 16,
                                 simde_mm256_mulhrs_epi16(h1, h1conv_w1));
        simde_mm256_storeu_si256(h1mbias + 2 * 16,
                                 simde_mm256_mulhrs_epi16(h1, h1conv_w2));
        simde_mm256_storeu_si256(h1mbias + 3 * 16,
                                 simde_mm256_mulhrs_epi16(h1, h1conv_w3));
        simde_mm256_storeu_si256(h1mbias + 4 * 16,
                                 simde_mm256_mulhrs_epi16(h1, h1conv_w4));
        simde_mm256_storeu_si256(h1mbias + 5 * 16,
                                 simde_mm256_mulhrs_epi16(h1, h1conv_w5));
      }
    }

    //-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    int16_t h3[MaxBS*MaxBS][16]; //�Ѿ�����h3lr_b

    auto h1lr1_w = simde_mm256_loadu_si256(weights.h1lr1_w + addrBias);
    auto h1lr2_w = simde_mm256_loadu_si256(weights.h1lr2_w + addrBias);
    auto h1conv_b = simde_mm256_loadu_si256(weights.h1conv_b + addrBias);
    auto h3lr_b    = simde_mm256_loadu_si256(weights.h3lr_b + addrBias);

    for (Loc locY = 0; locY < MaxBS; locY++) {
      for (Loc locX = 0; locX < MaxBS; locX++) {
        Loc loc1 = locY * MaxBS + locX;             //ԭʼloc
        Loc      loc2    = (locY + 5) * (MaxBS + 10) + locX + 5;  // padding���loc
        int16_t *h1mbias = h1m + loc2 * 6 * 16;

        auto h2sum = h3lr_b;

        const int dloc2s[4] = {1, MaxBS + 10, MaxBS + 10 + 1, -MaxBS - 10 + 1};
        for (int dir=0;dir<4;dir++)
        {
          const int dloc2 = dloc2s[dir];  

          //��������Ҫ��ȫ��load����
          auto      g2    = simde_mm256_loadu_si256(buf.g2[loc1][dir] + addrBias);
          auto h1cm5  = simde_mm256_loadu_si256(h1mbias - 5 * 16 * (6 * dloc2 - 1));
          auto h1cm4  = simde_mm256_loadu_si256(h1mbias - 4 * 16 * (6 * dloc2 - 1));
          auto h1cm3  = simde_mm256_loadu_si256(h1mbias - 3 * 16 * (6 * dloc2 - 1));
          auto h1cm2  = simde_mm256_loadu_si256(h1mbias - 2 * 16 * (6 * dloc2 - 1));
          auto h1cm1  = simde_mm256_loadu_si256(h1mbias - 1 * 16 * (6 * dloc2 - 1));
          auto h1c0   = simde_mm256_loadu_si256(h1mbias);
          auto h1c1   = simde_mm256_loadu_si256(h1mbias + 1 * 16 * (6 * dloc2 + 1));
          auto h1c2   = simde_mm256_loadu_si256(h1mbias + 2 * 16 * (6 * dloc2 + 1));
          auto h1c3   = simde_mm256_loadu_si256(h1mbias + 3 * 16 * (6 * dloc2 + 1));
          auto h1c4   = simde_mm256_loadu_si256(h1mbias + 4 * 16 * (6 * dloc2 + 1));
          auto h1c5   = simde_mm256_loadu_si256(h1mbias + 5 * 16 * (6 * dloc2 + 1));

          //11��h1c��h1conv_bȫ����ӣ�ʹ�á���������ʽ�ӷ�
          h1cm5 = simde_mm256_adds_epi16(h1cm5, h1conv_b);
          h1cm3 = simde_mm256_adds_epi16(h1cm3, h1cm4);
          h1cm1 = simde_mm256_adds_epi16(h1cm1, h1cm2);
          h1c1  = simde_mm256_adds_epi16(h1c1, h1c0);
          h1c3  = simde_mm256_adds_epi16(h1c3, h1c2);
          h1c5 = simde_mm256_adds_epi16(h1c5, h1c4);

          h1cm5 = simde_mm256_adds_epi16(h1cm5, h1cm3);
          h1cm1 = simde_mm256_adds_epi16(h1cm1, h1c1);
          h1c3 = simde_mm256_adds_epi16(h1c3, h1c5);

          auto h2 = simde_mm256_adds_epi16(h1cm1, h1c3);
          h2      = simde_mm256_adds_epi16(h1cm5, h2);

          //h1lr1
          h2 = simde_mm256_max_epi16(h2, simde_mm256_mulhrs_epi16(h2, h1lr1_w));
          //+g2
          h2 = simde_mm256_adds_epi16(h2, g2);
          //h1lr2
          h2 = simde_mm256_max_epi16(h2, simde_mm256_mulhrs_epi16(h2, h1lr2_w));

          
          h2sum = simde_mm256_adds_epi16(h2sum, simde_mm256_srai_epi16(h2,2)); //h2sum=mean(h2)=(h2+h2+h2+h2)/4
        }
        //save h3
        simde_mm256_storeu_si256(h3[loc1],h2sum);
      }
    }

    delete[] h1m;
    
    //-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    int16_t trunk1[(MaxBS+2) * (MaxBS+2)][16];//trunkconv2ǰ��trunk��padding=1
    memset(trunk1, 0, sizeof(int16_t) * (MaxBS + 2) * (MaxBS + 2) * 16);
    //��Ҫ�õ���Ȩ��
    auto h3lr_w  = simde_mm256_loadu_si256(weights.h3lr_w + addrBias);
    static_assert(trunkconv1GroupSize == 4, "���ڵĴ���ֻ֧��trunkconv1GroupSize == 4");
    auto trunkconv1_b = simde_mm256_loadu_si256(weights.trunkconv1_b + addrBias);
    auto trunkconv1_w0 = simde_mm256_loadu_si256(weights.trunkconv1_w[0] + addrBias);
    auto trunkconv1_w1 = simde_mm256_loadu_si256(weights.trunkconv1_w[1] + addrBias);
    auto trunkconv1_w2 = simde_mm256_loadu_si256(weights.trunkconv1_w[2] + addrBias);
    auto trunkconv1_w3 = simde_mm256_loadu_si256(weights.trunkconv1_w[3] + addrBias);
    auto trunklr1_w = simde_mm256_loadu_si256(weights.trunklr1_w + addrBias);
    h3lr_b   = simde_mm256_loadu_si256(weights.h3lr_b + addrBias);

    for (Loc locY = 0; locY < MaxBS; locY++) {
      for (Loc locX = 0; locX < MaxBS; locX++) {
        Loc loc1 = locY * MaxBS + locX;             //ԭʼloc
        Loc  loc2  = (locY + 1) * (MaxBS + 2) + locX + 1;  // padding���loc
        auto trunk  = simde_mm256_loadu_si256(h3[loc1]);
        // h3lr
        trunk = simde_mm256_max_epi16(trunk, simde_mm256_mulhrs_epi16(trunk, h3lr_w));
        //trunkconv1
        trunk = simde_mm256_adds_epi16(
            trunkconv1_b,
            simde_mm256_adds_epi16(
                simde_mm256_adds_epi16(
                    simde_mm256_mulhrs_epi16(
                        simde_mm256_permute4x64_epi64(trunk, 0b00000000),
                        trunkconv1_w0),
                    simde_mm256_mulhrs_epi16(
                        simde_mm256_permute4x64_epi64(trunk, 0b01010101),
                        trunkconv1_w1)),
                simde_mm256_adds_epi16(
                    simde_mm256_mulhrs_epi16(
                        simde_mm256_permute4x64_epi64(trunk, 0b10101010),
                        trunkconv1_w2),
                    simde_mm256_mulhrs_epi16(
                        simde_mm256_permute4x64_epi64(trunk, 0b11111111),
                        trunkconv1_w3))));

        // trunklr1
        trunk = simde_mm256_max_epi16(trunk, simde_mm256_mulhrs_epi16(trunk, trunklr1_w));

        //save
        simde_mm256_storeu_si256(trunk1[loc2], trunk);
      }
    }

    //-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    //��Ҫ�õ���Ȩ��
    auto trunkconv2_w0 = simde_mm256_loadu_si256(weights.trunkconv2_w[0] + addrBias);
    auto trunkconv2_w1 = simde_mm256_loadu_si256(weights.trunkconv2_w[1] + addrBias);
    auto trunkconv2_w2 = simde_mm256_loadu_si256(weights.trunkconv2_w[2] + addrBias);

    for (Loc locY = 0; locY < MaxBS; locY++) {
      for (Loc locX = 0; locX < MaxBS; locX++) {
        Loc  loc1  = locY * MaxBS + locX;            //ԭʼloc
        Loc  loc2   = (locY + 1) * (MaxBS + 2) + locX + 1;  // padding���loc
        auto trunka = simde_mm256_adds_epi16(
            simde_mm256_adds_epi16(simde_mm256_loadu_si256(trunk1[loc2 - (MaxBS + 2)]),
                                   simde_mm256_loadu_si256(trunk1[loc2 + (MaxBS + 2)])),
            simde_mm256_adds_epi16(simde_mm256_loadu_si256(trunk1[loc2 - 1]),
                                   simde_mm256_loadu_si256(trunk1[loc2 + 1])));
        auto trunkb = simde_mm256_adds_epi16(
            simde_mm256_adds_epi16(simde_mm256_loadu_si256(trunk1[loc2 - (MaxBS + 2) - 1]),
                                   simde_mm256_loadu_si256(trunk1[loc2 - (MaxBS + 2) + 1])),
            simde_mm256_adds_epi16(simde_mm256_loadu_si256(trunk1[loc2 + (MaxBS + 2) - 1]),
                                   simde_mm256_loadu_si256(trunk1[loc2 + (MaxBS + 2) + 1])));
        auto trunk = simde_mm256_loadu_si256(trunk1[loc2]);

        trunk = simde_mm256_mulhrs_epi16(trunk, trunkconv2_w0);
        trunk = simde_mm256_adds_epi16(simde_mm256_mulhrs_epi16(trunka, trunkconv2_w1), trunk);
        trunk = simde_mm256_adds_epi16(simde_mm256_mulhrs_epi16(trunkb, trunkconv2_w2), trunk);

        // save
        simde_mm256_storeu_si256(buf.trunk[loc1]+addrBias, trunk);
      }
    }

  }
    
  buf.trunkUpToDate = true;
  return;

}


void ModelBuf::emptyboard(const ModelWeight &weights)
{
  trunkUpToDate = false;

  // shape table
  {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < MaxBS * MaxBS; j++)
        shapeTable[j][i] = 0;

    //���²��ɽ���������Ϊ���ڸ���

    //�������ǽ����ǽ����ǽ������ǽ������ǽ��

    for (int thick = 1; thick <= 5; thick++) {
      for (int i = 0; i < MaxBS; i++) {
        int c = 0;
        for (int j = 0; j < thick; j++)
          c += pow3[11 - j];
        shapeTable[(MaxBS - 6 + thick) + i * MaxBS][0] = c;  //��ǽ
        shapeTable[i + (MaxBS - 6 + thick) * MaxBS][1] = c;  //��ǽ
        shapeTable[(MaxBS - 6 + thick) + i * MaxBS][2] = c;  //����ǽ����
        shapeTable[i + (MaxBS - 6 + thick) * MaxBS][2] = c;  //����ǽ����
        shapeTable[(MaxBS - 6 + thick) + i * MaxBS][3] = c;  //����ǽ����
        shapeTable[i + (6 - 1 - thick) * MaxBS][3]  = c;  //����ǽ����
      }
    }

    //�������ǽ����ǽ����ǽ������ǽ������ǽ��

    //���1
    for (int thick = 1; thick <= 5; thick++) {
      for (int i = 0; i < MaxBS; i++) {
        int c = 2 * pow3[11];  // 3����2000000000
        for (int j = 0; j < thick - 1; j++)
          c += pow3[j];
        shapeTable[(6 - 1 - thick) + i * MaxBS][0]  = c;  //��ǽ
        shapeTable[i + (6 - 1 - thick) * MaxBS][1]  = c;  //��ǽ
        shapeTable[(6 - 1 - thick) + i * MaxBS][2]  = c;  //����ǽ����
        shapeTable[i + (6 - 1 - thick) * MaxBS][2]  = c;  //����ǽ����
        shapeTable[(6 - 1 - thick) + i * MaxBS][3]  = c;  //����ǽ����
        shapeTable[i + (MaxBS - 6 + thick) * MaxBS][3] = c;  //����ǽ����
      }
    }

    //���߶���ǽ

    for (int a = 1; a <= 5; a++)    //������ǽ��
      for (int b = 1; b <= 5; b++)  //������ǽ��
      {
        int c = 3 * pow3[11];
        for (int i = 0; i < a - 1; i++)
          c += pow3[10 - i];
        for (int i = 0; i < b - 1; i++)
          c += pow3[i];
        shapeTable[(MaxBS - 6 + a) + (5 - b) * MaxBS][2]      = c;  //���Ͻ�
        shapeTable[(MaxBS - 6 + a) * MaxBS + (5 - b)][2]      = c;  //���½�
        shapeTable[(5 - b) + (5 - a) * MaxBS][3]           = c;  //���Ͻ�
        shapeTable[(MaxBS - 6 + a) + (MaxBS - 6 + b) * MaxBS][3] = c;  //���½�
      }
  }

  //g1 and g2
  for (Loc loc = 0; loc < MaxBS*MaxBS; loc++) {

    for (int i = 0; i < groupBatch; i++) {
      auto g1sum_ = simde_mm256_setzero_si256();
      for (int dir = 0; dir < 4; dir++) {
        // g2 update
        auto neww = simde_mm256_loadu_si256(weights.mapping[shapeTable[loc][dir]] + i * 16 + groupSize);
        simde_mm256_storeu_si256(g2[loc][dir] + i * 16, neww);
        // g1 update
        auto g1 = simde_mm256_loadu_si256(weights.mapping[shapeTable[loc][dir]] + i * 16);
        g1sum_  = simde_mm256_add_epi16(g1sum_, g1);
      }

      simde_mm256_storeu_si256(g1sum[loc] + i * 16, g1sum_);
    }
  }
}

bool Eva_nnuev2::loadParam(std::string filepath)
{
  using namespace std::filesystem;
  path ext = path(filepath).extension();
  if (ext.string() == ".bin") {
    std::ifstream cacheStream(path(filepath), std::ios::binary);
    cacheStream.read(reinterpret_cast<char *>(&weights), sizeof(weights));
    if (cacheStream.good()) {
      buf.emptyboard(weights);
      return true;
    }
    else
      return false;
  }

  path cachePath = path(filepath).replace_extension("bin");
  // Read parameter cache if exists
  if (exists(cachePath)) {
    std::ifstream cacheStream(cachePath, std::ios::binary);
    cacheStream.read(reinterpret_cast<char *>(&weights), sizeof(weights));
    if (cacheStream.good()) {
      buf.emptyboard(weights);
      return true;
    }
  }

  bool suc = weights.loadParam(filepath);
  if (suc) {
    buf.emptyboard(weights);
    // Save parameters cache
    std::ofstream cacheStream(cachePath, std::ios::binary);
    cacheStream.write(reinterpret_cast<char *>(&weights), sizeof(weights));
  }
  return suc;
}

void Eva_nnuev2::clear()
{
  for (int i = 0; i < MaxBS * MaxBS; i++)
    board[i] = C_EMPTY;
  buf.emptyboard(weights);
}

void Eva_nnuev2::recalculate()
{
  Color boardCopy[MaxBS * MaxBS];
  memcpy(boardCopy, board, MaxBS * MaxBS * sizeof(Color));
  clear();
  for (Loc i = LOC_ZERO; i < MaxBS * MaxBS; ++i) {
    if (boardCopy[i] != C_EMPTY)
      play(boardCopy[i], i);
  }
}

void Eva_nnuev2::play(Color color, Loc loc)
{
  board[loc] = color;
  buf.update(C_EMPTY, color, loc, weights);
}

ValueType Eva_nnuev2::evaluateFull(PolicyType *policy)
{
  if (policy != nullptr) {
    evaluatePolicy(policy);
  }
  return evaluateValue();
}

void Eva_nnuev2::evaluatePolicy(PolicyType *policy)
{
  if (policy == NULL)
    return;
  if (!buf.trunkUpToDate)
    calculateTrunk();
  
  for (Loc loc = 0; loc < MaxBS * MaxBS; loc++) {
    auto psum = simde_mm256_setzero_si256();//int32
    for (int batch = 0; batch < groupBatch; batch++) {

      //load
      auto t = simde_mm256_loadu_si256(buf.trunk[loc] + batch * 16);
      auto trunklr2p_b = simde_mm256_loadu_si256(weights.trunklr2p_b + batch * 16);
      auto trunklr2p_w = simde_mm256_loadu_si256(weights.trunklr2p_w + batch * 16);
      auto policy_linear_w = simde_mm256_loadu_si256(weights.policy_linear_w + batch * 16);

      //trunklr2p
      t = simde_mm256_adds_epi16(t, trunklr2p_b);
      t = simde_mm256_max_epi16(t, simde_mm256_mulhrs_epi16(t, trunklr2p_w));

      //policy linear
      t          = simde_mm256_madd_epi16(t, policy_linear_w); 
      psum       = simde_mm256_add_epi32(t, psum);

    }

    psum = simde_mm256_hadd_epi32(psum, psum);
    psum = simde_mm256_hadd_epi32(psum, psum);

    auto  p = simde_mm256_extract_epi32(psum, 0) + simde_mm256_extract_epi32(psum, 4);
    policy[loc] = p * weights.scale_policyInv * quantFactor/32768;
  }
}

ValueType Eva_nnuev2::evaluateValue()
{
  if (!buf.trunkUpToDate)
    calculateTrunk();


  //trunklr2v, sum board
  float vsum[groupSize];
  for (int batch16 = 0; batch16 < groupBatch; batch16++) {
    auto vsum0 = simde_mm256_setzero_si256();
    auto vsum1 = simde_mm256_setzero_si256();

    auto trunklr2v_b = simde_mm256_loadu_si256(weights.trunklr2v_b + batch16 * 16);
    auto trunklr2v_w = simde_mm256_loadu_si256(weights.trunklr2v_w + batch16 * 16);
    for (Loc loc = 0; loc < MaxBS * MaxBS; loc++) {
      auto t = simde_mm256_loadu_si256(buf.trunk[loc] + batch16 * 16);
      // trunklr2p
      t = simde_mm256_adds_epi16(t, trunklr2v_b);
      t     = simde_mm256_max_epi16(t, simde_mm256_mulhrs_epi16(t, trunklr2v_w));
      vsum0 = simde_mm256_add_epi32(
          vsum0,
          simde_mm256_cvtepi16_epi32(simde_mm256_extractf128_si256(t, 0)));
      vsum1 = simde_mm256_add_epi32(
          vsum1,
          simde_mm256_cvtepi16_epi32(simde_mm256_extractf128_si256(t, 1)));
    }
    simde_mm256_storeu_ps(vsum + batch16 * 16, simde_mm256_cvtepi32_ps(vsum0));
    simde_mm256_storeu_ps(vsum + batch16 * 16 + 8, simde_mm256_cvtepi32_ps(vsum1));
  }

  //scale, valuelr
  auto scale       = simde_mm256_set1_ps(weights.scale_beforemlpInv / MaxBS / MaxBS);
  for (int batch32 = 0; batch32 < groupBatch * 2; batch32++) {
    auto valuelr_b = simde_mm256_loadu_ps(weights.valuelr_b + batch32 * 8);
    auto valuelr_w = simde_mm256_loadu_ps(weights.valuelr_w + batch32 * 8);
    auto v         = simde_mm256_loadu_ps(vsum + batch32 * 8);
    v              = simde_mm256_mul_ps(v, scale);
    v              = simde_mm256_add_ps(v, valuelr_b);
    v              = simde_mm256_max_ps(v, simde_mm256_mul_ps(v, valuelr_w));
    simde_mm256_storeu_ps(vsum + batch32 * 8, v);
  }

  // linear 1
  float layer1[mlpChannel];
  for (int i = 0; i < mlpBatch32; i++) {
    auto sum = simde_mm256_loadu_ps(weights.mlp_b1 + i * 8);
    for (int j = 0; j < groupSize; j++) {
      auto x = simde_mm256_set1_ps(vsum[j]);
      auto w = simde_mm256_loadu_ps(weights.mlp_w1[j] + i * 8);
      sum    = simde_mm256_fmadd_ps(w, x, sum);
    }
    sum = simde_mm256_max_ps(simde_mm256_setzero_ps(), sum);  // relu
    simde_mm256_storeu_ps(layer1 + i * 8, sum);
  }

  // linear 2
  float layer2[mlpChannel];
  for (int i = 0; i < mlpBatch32; i++) {
    auto sum = simde_mm256_loadu_ps(weights.mlp_b2 + i * 8);
    for (int j = 0; j < mlpChannel; j++) {
      auto x = simde_mm256_set1_ps(layer1[j]);
      auto w = simde_mm256_loadu_ps(weights.mlp_w2[j] + i * 8);
      sum    = simde_mm256_fmadd_ps(w, x, sum);
    }
    sum = simde_mm256_max_ps(simde_mm256_setzero_ps(), sum);  // relu
    simde_mm256_storeu_ps(layer2 + i * 8, sum);
  }

  // linear 3
  float layer3[mlpChannel];
  for (int i = 0; i < mlpBatch32; i++) {
    auto sum = simde_mm256_loadu_ps(weights.mlp_b3 + i * 8);
    for (int j = 0; j < mlpChannel; j++) {
      auto x = simde_mm256_set1_ps(layer2[j]);
      auto w = simde_mm256_loadu_ps(weights.mlp_w3[j] + i * 8);
      sum    = simde_mm256_fmadd_ps(w, x, sum);
    }
    sum = simde_mm256_max_ps(simde_mm256_setzero_ps(), sum);  // relu
    simde_mm256_storeu_ps(layer3 + i * 8, sum);
  }

  // final linear

  auto v = simde_mm256_loadu_ps(weights.mlpfinal_b);
  for (int inc = 0; inc < mlpChannel; inc++) {
    auto x = simde_mm256_set1_ps(layer3[inc]);
    auto w = simde_mm256_loadu_ps(weights.mlpfinal_w[inc]);
    v      = simde_mm256_fmadd_ps(w, x, v);
  }
  float value[8];
  simde_mm256_storeu_ps(value, v);
  return ValueType(value[0], value[1], value[2]);
}

void Eva_nnuev2::undo(Loc loc)
{
  buf.update(board[loc], C_EMPTY, loc, weights);
  board[loc] = C_EMPTY;
}

void Eva_nnuev2::debug_print()
{
  using namespace std;
  Loc loc = MakeLoc(0, 0);
  PolicyType p[MaxBS * MaxBS];
  auto       v = evaluateFull(p);
  cout << "value: win=" << v.win << " loss=" << v.loss << " draw=" << v.draw << endl;
  //for (int i = 48; i < groupSize; i++)
  //  cout << buf.g1sum[loc][i] << "|" << buf.g2[loc][3][i] << "|" << buf.trunk[loc][i] << " ";


  cout << "policy: " <<  endl;
  for (int y = 0; y < MaxBS; y++) {
    for (int x = 0; x < MaxBS; x++)
      cout << p[y*MaxBS+x] << "\t";
    cout << endl;
  }
  cout << endl;
  /*
  cout << "mapsum";
  for (int i = 0; i < gr; i++)
    cout << buf.mapsum[loc][i] << " ";
  cout << endl;
  cout << "mapafterlr";
  for (int i = 0; i < mix6::featureNum; i++)
    cout << buf.mapAfterLR[loc][i] << " ";
  cout << endl;
  cout << "policyafterconv";
  for (int i = 0; i < mix6::policyNum; i++)
    cout << buf.policyAfterConv[loc][i] << " ";
  cout << endl;
  cout << "valueavg";
  for (int i = 0; i < mix6::valueNum; i++)
    cout << float(buf.valueSumBoard[i]) / (MaxBS * MaxBS) << " ";
  cout << endl;*/
}

bool ModelWeight::loadParam(std::string filename)
{
  using namespace std;
  ifstream fs(filename);

  // clear map
  for (int i = 0; i < shapeNum; i++) {
    for (int j = 0; j < featureNum; j++)
      mapping[i][j] = IllegalShapeFeature;
  }

  string modelname;
  fs >> modelname;
  if (modelname != "v2") {
    cout << "Wrong model type:" << modelname << endl;
    return false;
  }

  int param;
  fs >> param;
  if (param != groupSize) {
    cout << "Wrong group size:" << param << endl;
    return false;
  }
  fs >> param;
  if (param != mlpChannel) {
    cout << "Wrong mlp channel:" << param << endl;
    return false;
  }



  string varname;

  // mapping
  fs >> varname;
  if (varname != "mapping") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  int shapeN;
  fs >> shapeN;
  for (int i = 0; i < shapeN; i++) {
    int shapeID;
    fs >> shapeID;
    for (int j = 0; j < featureNum; j++)
      fs >> mapping[shapeID][j];
  }

  // g1lr_w
  fs >> varname;
  if (varname != "g1lr_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> g1lr_w[i];

  // h1conv_w
  fs >> varname;
  if (varname != "h1conv_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < 6; j++)
    for (int i = 0; i < groupSize; i++)
      fs >> h1conv_w[j][i];

  // h1conv_b
  fs >> varname;
  if (varname != "h1conv_b") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> h1conv_b[i];

  // h1lr1_w
  fs >> varname;
  if (varname != "h1lr1_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> h1lr1_w[i];

  // h1lr2_w
  fs >> varname;
  if (varname != "h1lr2_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> h1lr2_w[i];

  // h3lr_w
  fs >> varname;
  if (varname != "h3lr_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> h3lr_w[i];

  // h3lr_b
  fs >> varname;
  if (varname != "h3lr_b") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> h3lr_b[i];

  // trunkconv1_w
  fs >> varname;
  if (varname != "trunkconv1_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < trunkconv1GroupSize; j++)
    for (int i = 0; i < groupSize; i++)
        fs >> trunkconv1_w[j][i];

  // trunkconv1_b
  fs >> varname;
  if (varname != "trunkconv1_b") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> trunkconv1_b[i];

  // trunklr1_w
  fs >> varname;
  if (varname != "trunklr1_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> trunklr1_w[i];

  // trunkconv2_w
  fs >> varname;
  if (varname != "trunkconv2_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < groupSize; i++)
        fs >> trunkconv2_w[j][i];

  // trunklr2p_w
  fs >> varname;
  if (varname != "trunklr2p_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> trunklr2p_w[i];

  // trunklr2p_b
  fs >> varname;
  if (varname != "trunklr2p_b") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> trunklr2p_b[i];

  // trunklr2v_w
  fs >> varname;
  if (varname != "trunklr2v_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> trunklr2v_w[i];

  // trunklr2v_b
  fs >> varname;
  if (varname != "trunklr2v_b") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> trunklr2v_b[i];

  // policy_linear_w
  fs >> varname;
  if (varname != "policy_linear_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> policy_linear_w[i];

  // scale_policyInv
  fs >> varname;
  if (varname != "scale_policyInv") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  fs >> scale_policyInv;

  // scale_beforemlpInv
  fs >> varname;
  if (varname != "scale_beforemlpInv") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  fs >> scale_beforemlpInv;

  // valuelr_w
  fs >> varname;
  if (varname != "valuelr_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> valuelr_w[i];

  // valuelr_b
  fs >> varname;
  if (varname != "valuelr_b") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < groupSize; i++)
    fs >> valuelr_b[i];

  // mlp_w1
  fs >> varname;
  if (varname != "mlp_w1") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < groupSize; j++)
    for (int i = 0; i < mlpChannel; i++)
        fs >> mlp_w1[j][i];

  // mlp_b1
  fs >> varname;
  if (varname != "mlp_b1") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mlpChannel; i++)
    fs >> mlp_b1[i];

  // mlp_w2
  fs >> varname;
  if (varname != "mlp_w2") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < mlpChannel; j++)
    for (int i = 0; i < mlpChannel; i++)
      fs >> mlp_w2[j][i];

  // mlp_b2
  fs >> varname;
  if (varname != "mlp_b2") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mlpChannel; i++)
    fs >> mlp_b2[i];

  // mlp_w3
  fs >> varname;
  if (varname != "mlp_w3") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < mlpChannel; j++)
    for (int i = 0; i < mlpChannel; i++)
      fs >> mlp_w3[j][i];

  // mlp_b3
  fs >> varname;
  if (varname != "mlp_b3") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mlpChannel; i++)
    fs >> mlp_b3[i];

  // mlpfinal_w
  fs >> varname;
  if (varname != "mlpfinal_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < mlpChannel; j++)
    for (int i = 0; i < 3; i++)
      fs >> mlpfinal_w[j][i];

  // mlpfinal_b
  fs >> varname;
  if (varname != "mlpfinal_b") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < 3; i++)
    fs >> mlpfinal_b[i];

  for (int i = 0; i < 5; i++) {
    mlpfinal_w_for_safety[i] = 0;
    mlpfinal_b_for_safety[i] = 0;
  }

  return true;
}