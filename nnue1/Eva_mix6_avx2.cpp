#include "Eva_mix6_avx2.h"

#include "external/simde_avx2.h"
#include "external/simde_fma.h"

#include <filesystem>

void Mix6buf_int16::update(Color                   oldcolor,
                           Color                   newcolor,
                           Loc                     loc,
                           const Mix6weight_int16 &weights)
{
  // update shapeTable
  std::vector<OnePointChange> changeTable(44);
  int                         changenum = 0;

  {
    int x0 = loc % BS;
    int y0 = loc / BS;

    int dxs[4] = {1, 0, 1, 1};
    int dys[4] = {0, 1, 1, -1};

    for (int dir = 0; dir < 4; dir++) {
      for (int dist = -5; dist <= 5; dist++) {
        int x = x0 - dist * dxs[dir];
        int y = y0 - dist * dys[dir];
        if (x < 0 || x >= BS || y < 0 || y >= BS)
          continue;
        OnePointChange c;
        c.dir = dir, c.loc = MakeLoc(x, y);
        c.oldshape             = shapeTable[c.loc][dir];
        c.newshape             = c.oldshape + (newcolor - oldcolor) * pow3[dist + 5];
        shapeTable[c.loc][dir] = c.newshape;
        if (c.loc == 106) {
          Color o = oldcolor, n = newcolor;
          int   a = 0;
        }
        changeTable[changenum] = c;
        changenum++;
      }
    }
  }

  for (int p = 0; p < changenum; p++) {
    OnePointChange c = changeTable[p];

    int y0 = c.loc / BS, x0 = c.loc % BS;

    for (int i = 0; i < mix6::featureBatch; i++) {
      // mapsum
      auto  oldw = simde_mm256_loadu_si256(weights.map[c.oldshape] + i * 16);
      auto  neww = simde_mm256_loadu_si256(weights.map[c.newshape] + i * 16);
      void *wp   = mapsum[c.loc] + i * 16;
      auto  sumw = simde_mm256_loadu_si256(wp);
      sumw       = simde_mm256_sub_epi16(sumw, oldw);
      sumw       = simde_mm256_add_epi16(sumw, neww);
      simde_mm256_storeu_si256(wp, sumw);

      // leaky relu
      auto lrw = simde_mm256_loadu_si256(weights.map_lr_slope_sub1div8 + i * 16);
      auto lrb = simde_mm256_loadu_si256(weights.map_lr_bias + i * 16);
      neww     = simde_mm256_add_epi16(
          simde_mm256_add_epi16(                // 0.25leakyrelu(x)=
              simde_mm256_srai_epi16(sumw, 2),  // 0.25x
              simde_mm256_slli_epi16(
                  simde_mm256_mulhrs_epi16(
                      lrw,
                      simde_mm256_min_epi16(simde_mm256_setzero_si256(), sumw)),
                  1)),  //+2*slopeSub1Div8*-relu(-x)
          lrb);         //+bias
      wp   = mapAfterLR[c.loc] + i * 16;
      oldw = simde_mm256_loadu_si256(wp);
      simde_mm256_storeu_si256(wp, neww);

      // policy conv
      if (i < mix6::policyBatch) {
        for (int dy = -1; dy <= 1; dy++) {
          int y = y0 + dy;
          if (y < 0 || y >= BS)
            continue;
          for (int dx = -1; dx <= 1; dx++) {
            int x = x0 + dx;
            if (x < 0 || x >= BS)
              continue;
            Loc  thisloc = MakeLoc(x, y);
            auto convw = simde_mm256_loadu_si256(weights.policyConvWeight[4 - dy * 3 - dx]
                                                 + i * 16);
            wp         = policyAfterConv[thisloc] + i * 16;
            sumw       = simde_mm256_loadu_si256(wp);
            sumw = simde_mm256_sub_epi16(sumw, simde_mm256_mulhrs_epi16(oldw, convw));
            sumw = simde_mm256_add_epi16(sumw, simde_mm256_mulhrs_epi16(neww, convw));
            simde_mm256_storeu_si256(wp, sumw);
          }
        }
      }
      else  // value sum
      {
        // lower
        int valueBatchID = 2 * (i - mix6::policyBatch);
        wp               = valueSumBoard + valueBatchID * 8;
        auto oldw32 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(oldw, 0));
        auto neww32 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(neww, 0));
        sumw        = simde_mm256_loadu_si256(wp);
        sumw        = simde_mm256_sub_epi32(sumw, oldw32);
        sumw        = simde_mm256_add_epi32(sumw, neww32);
        simde_mm256_storeu_si256(wp, sumw);

        // upper
        valueBatchID++;
        wp     = valueSumBoard + valueBatchID * 8;
        oldw32 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(oldw, 1));
        neww32 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(neww, 1));
        sumw   = simde_mm256_loadu_si256(wp);
        sumw   = simde_mm256_sub_epi32(sumw, oldw32);
        sumw   = simde_mm256_add_epi32(sumw, neww32);
        simde_mm256_storeu_si256(wp, sumw);
      }
    }
  }
}

ValueType Mix6buf_int16::calculateValue(const Mix6weight_int16 &weights)
{
  // layer 0 leakyrelu
  float layer0[mix6::valueNum];
  for (int i = 0; i < mix6::valueBatch32; i++) {
    auto x = simde_mm256_loadu_si256(valueSumBoard + i * 8);  // load
    auto y = simde_mm256_cvtepi32_ps(x);
    y = simde_mm256_mul_ps(y, simde_mm256_set1_ps(weights.scale_beforemlp));  // scale
    auto w = simde_mm256_loadu_ps(weights.value_lr_slope_sub1 + i * 8);       // load
    y      = simde_mm256_fmadd_ps(w,
                             simde_mm256_min_ps(simde_mm256_setzero_ps(), y),
                             y);  // leaky relu
    simde_mm256_storeu_ps(layer0 + i * 8, y);
  }

  // linear 1
  float layer1[mix6::valueNum];
  for (int i = 0; i < mix6::valueBatch32; i++) {
    auto sum = simde_mm256_loadu_ps(weights.mlp_b1 + i * 8);
    for (int inc = 0; inc < mix6::valueNum; inc++) {
      auto x = simde_mm256_set1_ps(layer0[inc]);
      auto w = simde_mm256_loadu_ps(weights.mlp_w1[inc] + i * 8);
      sum    = simde_mm256_fmadd_ps(w, x, sum);
    }
    sum = simde_mm256_max_ps(simde_mm256_setzero_ps(), sum);  // relu
    simde_mm256_storeu_ps(layer1 + i * 8, sum);
  }

  // linear 2
  float layer2[mix6::valueNum];
  for (int i = 0; i < mix6::valueBatch32; i++) {
    auto sum = simde_mm256_loadu_ps(weights.mlp_b2 + i * 8);
    for (int inc = 0; inc < mix6::valueNum; inc++) {
      auto x = simde_mm256_set1_ps(layer1[inc]);
      auto w = simde_mm256_loadu_ps(weights.mlp_w2[inc] + i * 8);
      sum    = simde_mm256_fmadd_ps(w, x, sum);
    }
    sum      = simde_mm256_max_ps(simde_mm256_setzero_ps(), sum);  // relu
    auto res = simde_mm256_loadu_ps(layer0 + i * 8);               // resnet
    sum      = simde_mm256_add_ps(sum, res);
    simde_mm256_storeu_ps(layer2 + i * 8, sum);
  }

  // final linear

  auto v = simde_mm256_loadu_ps(weights.mlp_b3);
  for (int inc = 0; inc < mix6::valueNum; inc++) {
    auto x = simde_mm256_set1_ps(layer2[inc]);
    auto w = simde_mm256_loadu_ps(weights.mlp_w3[inc]);
    v      = simde_mm256_fmadd_ps(w, x, v);
  }
  float value[8];
  simde_mm256_storeu_ps(value, v);
  return ValueType(value[0], value[1], value[2]);
}

void Mix6buf_int16::calculatePolicy(PolicyType *policy, const Mix6weight_int16 &weights)
{
  static_assert(
      mix6::policyBatch == 1,
      "Assume there's only one policy batch,or we need to calculate policy by batch");  //
  if (policy == NULL)
    return;
  for (Loc loc = LOC_ZERO; loc < BS * BS; ++loc) {
    void *wp   = policyAfterConv[loc];
    auto  t    = simde_mm256_loadu_si256(wp);
    t          = simde_mm256_max_epi16(simde_mm256_setzero_si256(), t);  // relu
    auto convw = simde_mm256_loadu_si256(weights.policyFinalConv);
    t          = simde_mm256_mulhrs_epi16(t, convw);

    t = simde_mm256_hadds_epi16(t, t);
    t = simde_mm256_hadds_epi16(t, t);
    t = simde_mm256_hadds_epi16(t, t);

    auto  p1 = _mm_adds_epi16(simde_mm256_extractf128_si256(t, 0),
                             simde_mm256_extractf128_si256(t, 1));
    float p  = (int16_t)_mm_extract_epi16(p1, 0);

    if (p < 0)
      p = p * weights.policy_neg_slope;
    else
      p = p * weights.policy_pos_slope;
    policy[loc] = p * quantFactor;
  }
}

void Mix6buf_int16::emptyboard(const Mix6weight_int16 &weights)
{
  // shape table
  {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < BS * BS; j++)
        shapeTable[j][i] = 0;

    //���²��ɽ���������Ϊ���ڸ���

    //�������ǽ����ǽ����ǽ������ǽ������ǽ��

    for (int thick = 1; thick <= 5; thick++) {
      for (int i = 0; i < BS; i++) {
        int c = 0;
        for (int j = 0; j < thick; j++)
          c += pow3[11 - j];
        shapeTable[(BS - 6 + thick) + i * BS][0] = c;  //��ǽ
        shapeTable[i + (BS - 6 + thick) * BS][1] = c;  //��ǽ
        shapeTable[(BS - 6 + thick) + i * BS][2] = c;  //����ǽ����
        shapeTable[i + (BS - 6 + thick) * BS][2] = c;  //����ǽ����
        shapeTable[(BS - 6 + thick) + i * BS][3] = c;  //����ǽ����
        shapeTable[i + (6 - 1 - thick) * BS][3]  = c;  //����ǽ����
      }
    }

    //�������ǽ����ǽ����ǽ������ǽ������ǽ��

    //���1
    for (int thick = 1; thick <= 5; thick++) {
      for (int i = 0; i < BS; i++) {
        int c = 2 * pow3[11];  // 3����2000000000
        for (int j = 0; j < thick - 1; j++)
          c += pow3[j];
        shapeTable[(6 - 1 - thick) + i * BS][0]  = c;  //��ǽ
        shapeTable[i + (6 - 1 - thick) * BS][1]  = c;  //��ǽ
        shapeTable[(6 - 1 - thick) + i * BS][2]  = c;  //����ǽ����
        shapeTable[i + (6 - 1 - thick) * BS][2]  = c;  //����ǽ����
        shapeTable[(6 - 1 - thick) + i * BS][3]  = c;  //����ǽ����
        shapeTable[i + (BS - 6 + thick) * BS][3] = c;  //����ǽ����
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
        shapeTable[(BS - 6 + a) + (5 - b) * BS][2]      = c;  //���Ͻ�
        shapeTable[(BS - 6 + a) * BS + (5 - b)][2]      = c;  //���½�
        shapeTable[(5 - b) + (5 - a) * BS][3]           = c;  //���Ͻ�
        shapeTable[(BS - 6 + a) + (BS - 6 + b) * BS][3] = c;  //���½�
      }
  }

  // clear policyAfterConv and valueSumBoard
  for (int i = 0; i < mix6::policyBatch; i++) {
    auto bias = simde_mm256_loadu_si256(weights.policyConvBias + i * 16);
    for (Loc loc = LOC_ZERO; loc < BS * BS; ++loc) {
      simde_mm256_storeu_si256(policyAfterConv[loc] + i * 16, bias);
    }
  }

  for (int i = 0; i < mix6::valueBatch32; i++) {
    simde_mm256_storeu_si256(valueSumBoard + i * 8, simde_mm256_setzero_si256());
  }

  // mapsum,mapAfterLR,policyAfterConv,valueSumBoard

  for (Loc loc = LOC_ZERO; loc < BS * BS; ++loc) {
    int y0 = loc / BS, x0 = loc % BS;

    for (int i = 0; i < mix6::featureBatch; i++) {
      // mapsum
      auto sumw = simde_mm256_setzero_si256();
      for (int dir = 0; dir < 4; dir++) {
        auto dw = simde_mm256_loadu_si256(weights.map[shapeTable[loc][dir]] + i * 16);
        sumw    = simde_mm256_add_epi16(sumw, dw);
        // std::cout << simde_mm256_extract_epi16(sumw, 0) << " " << std::endl;
      }
      // std::cout << simde_mm256_extract_epi16(sumw, 0) <<" " << std::endl;
      void *wp = mapsum[loc] + i * 16;
      simde_mm256_storeu_si256(wp, sumw);

      // leaky relu
      // leaky relu
      auto lrw = simde_mm256_loadu_si256(weights.map_lr_slope_sub1div8 + i * 16);
      auto lrb = simde_mm256_loadu_si256(weights.map_lr_bias + i * 16);
      sumw     = simde_mm256_add_epi16(
          simde_mm256_add_epi16(                // 0.25leakyrelu(x)=
              simde_mm256_srai_epi16(sumw, 2),  // 0.25x
              simde_mm256_slli_epi16(
                  simde_mm256_mulhrs_epi16(
                      lrw,
                      simde_mm256_min_epi16(simde_mm256_setzero_si256(), sumw)),
                  1)),  //+2*slopeSub1Div8*-relu(-x)
          lrb);         //+bias

      wp = mapAfterLR[loc] + i * 16;
      simde_mm256_storeu_si256(wp, sumw);

      // policy conv
      if (i < mix6::policyBatch) {
        for (int dy = -1; dy <= 1; dy++) {
          int y = y0 + dy;
          if (y < 0 || y >= BS)
            continue;
          for (int dx = -1; dx <= 1; dx++) {
            int x = x0 + dx;
            if (x < 0 || x >= BS)
              continue;
            Loc  thisloc = MakeLoc(x, y);
            auto convw = simde_mm256_loadu_si256(weights.policyConvWeight[4 - dy * 3 - dx]
                                                 + i * 16);
            wp         = policyAfterConv[thisloc] + i * 16;
            auto oldw  = simde_mm256_loadu_si256(wp);
            oldw = simde_mm256_add_epi16(oldw, simde_mm256_mulhrs_epi16(sumw, convw));
            simde_mm256_storeu_si256(wp, oldw);
          }
        }
      }
      else  // value sum
      {
        // lower
        int valueBatchID = 2 * (i - mix6::policyBatch);
        wp               = valueSumBoard + valueBatchID * 8;
        auto neww32 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(sumw, 0));
        auto oldw   = simde_mm256_loadu_si256(wp);
        oldw        = simde_mm256_add_epi32(oldw, neww32);
        simde_mm256_storeu_si256(wp, oldw);

        // upper
        valueBatchID++;
        wp     = valueSumBoard + valueBatchID * 8;
        neww32 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(sumw, 1));
        oldw   = simde_mm256_loadu_si256(wp);
        oldw   = simde_mm256_add_epi32(oldw, neww32);
        simde_mm256_storeu_si256(wp, oldw);
      }
    }
  }
}

bool Eva_mix6_avx2::loadParam(std::string filepath)
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

void Eva_mix6_avx2::clear()
{
  for (int i = 0; i < BS * BS; i++)
    board[i] = C_EMPTY;
  buf.emptyboard(weights);
}

void Eva_mix6_avx2::recalculate()
{
  Color boardCopy[BS * BS];
  memcpy(boardCopy, board, BS * BS * sizeof(Color));
  clear();
  for (Loc i = LOC_ZERO; i < BS * BS; ++i) {
    if (boardCopy[i] != C_EMPTY)
      play(boardCopy[i], i);
  }
}

void Eva_mix6_avx2::play(Color color, Loc loc)
{
  board[loc] = color;
  buf.update(C_EMPTY, color, loc, weights);
}

ValueType Eva_mix6_avx2::evaluateFull(PolicyType *policy)
{
  if (policy != nullptr) {
    evaluatePolicy(policy);
  }
  return evaluateValue();
}

void Eva_mix6_avx2::evaluatePolicy(PolicyType *policy)
{
  buf.calculatePolicy(policy, weights);
}

ValueType Eva_mix6_avx2::evaluateValue() { return buf.calculateValue(weights); }

void Eva_mix6_avx2::undo(Loc loc)
{
  buf.update(board[loc], C_EMPTY, loc, weights);
  board[loc] = C_EMPTY;
}

void Eva_mix6_avx2::debug_print()
{
  using namespace std;
  Loc loc = MakeLoc(7, 7);
  cout << "mapsum";
  for (int i = 0; i < mix6::featureNum; i++)
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
    cout << float(buf.valueSumBoard[i]) / (BS * BS) << " ";
  cout << endl;
}

bool Mix6weight_int16::loadParam(std::string filename)
{
  using namespace std;
  ifstream fs(filename);

  // clear map
  for (int i = 0; i < mix6::shapeNum; i++) {
    for (int j = 0; j < mix6::featureNum; j++)
      map[i][j] = IllegalShapeFeature;
  }

  string varname;

  // map
  fs >> varname;
  if (varname != "featuremap") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }

  int shapeN;
  fs >> shapeN;
  for (int i = 0; i < shapeN; i++) {
    int shapeID;
    fs >> shapeID;
    for (int j = 0; j < mix6::featureNum; j++)
      fs >> map[shapeID][j];
  }

  // map_lr_slope_sub1div8
  fs >> varname;
  if (varname != "map_lr_slope_sub1div8") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::featureNum; i++)
    fs >> map_lr_slope_sub1div8[i];

  // map_lr_bias
  fs >> varname;
  if (varname != "map_lr_bias") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::featureNum; i++)
    fs >> map_lr_bias[i];

  // policyConvWeight
  fs >> varname;
  if (varname != "policyConvWeight") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < 9; j++)
    for (int i = 0; i < mix6::policyNum; i++)
      fs >> policyConvWeight[j][i];

  // policyConvBias
  fs >> varname;
  if (varname != "policyConvBias") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::policyNum; i++)
    fs >> policyConvBias[i];

  // policyFinalConv
  fs >> varname;
  if (varname != "policyFinalConv") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::policyNum; i++)
    fs >> policyFinalConv[i];

  // policy_neg_slope
  fs >> varname;
  if (varname != "policy_neg_slope") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  fs >> policy_neg_slope;

  // policy_pos_slope
  fs >> varname;
  if (varname != "policy_pos_slope") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  fs >> policy_pos_slope;

  // scale_beforemlp
  fs >> varname;
  if (varname != "scale_beforemlp") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  fs >> scale_beforemlp;

  // value_lr_slope
  fs >> varname;
  if (varname != "value_lr_slope_sub1") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::valueNum; i++)
    fs >> value_lr_slope_sub1[i];

  // mlp_w1
  fs >> varname;
  if (varname != "mlp_w1") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::valueNum; i++)
    for (int j = 0; j < mix6::valueNum; j++)
      fs >> mlp_w1[i][j];

  // mlp_b1
  fs >> varname;
  if (varname != "mlp_b1") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::valueNum; i++)
    fs >> mlp_b1[i];

  // mlp_w2
  fs >> varname;
  if (varname != "mlp_w2") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::valueNum; i++)
    for (int j = 0; j < mix6::valueNum; j++)
      fs >> mlp_w2[i][j];

  // mlp_b2
  fs >> varname;
  if (varname != "mlp_b2") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::valueNum; i++)
    fs >> mlp_b2[i];

  // mlp_w3
  fs >> varname;
  if (varname != "mlp_w3") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mix6::valueNum; i++)
    for (int j = 0; j < 3; j++)
      fs >> mlp_w3[i][j];

  // mlp_b3
  fs >> varname;
  if (varname != "mlp_b3") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < 3; i++)
    fs >> mlp_b3[i];

  return true;
}
