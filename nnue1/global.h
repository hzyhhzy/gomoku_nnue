#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

const int BS = 15;

typedef int8_t         BasicRule;
const BasicRule BASICRULE_FREESTYLE = 0;
const BasicRule BASICRULE_STANDARD  = 1;
const BasicRule BASICRULE_RENJU     = 2;

const BasicRule DEFAULT_RULE = BASICRULE_STANDARD;

typedef int8_t VCNRule;
const VCNRule VCNRULE_NOVC  = 0;
const VCNRule VCNRULE_VC1_B = 1;
const VCNRule VCNRULE_VC2_B = 2;
const VCNRule VCNRULE_VC3_B = 3;
const VCNRule VCNRULE_VC4_B = 4;
const VCNRule VCNRULE_VC1_W = 11;
const VCNRule VCNRULE_VC2_W = 12;
const VCNRule VCNRULE_VC3_W = 13;
const VCNRule VCNRULE_VC4_W = 14;
const VCNRule VCNRULE_VC1_M = 1;
const VCNRule VCNRULE_VC2_M = 2;
const VCNRule VCNRULE_VC3_M = 3;
const VCNRule VCNRULE_VC4_M = 4;
const VCNRule VCNRULE_VC1_O = 11;
const VCNRule VCNRULE_VC2_O = 12;
const VCNRule VCNRULE_VC3_O = 13;
const VCNRule VCNRULE_VC4_O = 14;



typedef uint64_t Key;

typedef int8_t         Color;
static constexpr Color C_EMPTY = 0;
static constexpr Color C_BLACK = 1;
static constexpr Color C_WHITE = 2;
static constexpr Color C_WALL  = 3;
static constexpr Color C_MY    = 1;
static constexpr Color C_OPP   = 2;

static inline Color getOpp(Color c) { return c ^ 3; }

typedef int16_t      Loc;
static constexpr Loc LOC_ZERO = 0;
static constexpr Loc LOC_NULL = -1;
static constexpr Loc LOC_PASS = BS * BS;
inline Loc           MakeLoc(int x, int y) { return Loc(x + y * BS); }
inline std::string   locstr(Loc loc)
{
  return std::string(1, char('A' + loc % BS)) + std::to_string(int(BS - loc / BS));
}
inline std::ostream &operator<<(std::ostream &os, std::vector<Loc> pv)
{
  for (size_t i = 0; i < pv.size(); i++)
    os << " " + !i << locstr(pv[i]);
  return os;
}

typedef int32_t  PolicyType;
const PolicyType MIN_POLICY     = -5e8;
const PolicyType MYFIVE_POLICY  = 1e8;
const PolicyType OPPFOUR_POLICY = 1e7;
const PolicyType MYFOUR_POLICY  = 1e6;
const PolicyType VCF_POLICY     = 1e5;

const double WIN_VALUE  = 1;
const double LOSE_VALUE = -1;

const float quantFactor = 32;

struct ValueType
{
  float win, loss, draw;
  ValueType(float win, float loss, float draw) : win(win), loss(loss), draw(draw)
  {
    self_softmax();
  }
  void self_softmax()
  {
    // std::cout<<win<<" "<<loss<<" "<<draw << std::endl;
    float maxvalue = std::max(std::max(win, loss), draw);
    win            = exp(win - maxvalue);
    loss           = exp(loss - maxvalue);
    draw           = exp(draw - maxvalue);
    float total    = win + loss + draw;
    win            = win / total;
    loss           = loss / total;
    draw           = draw / total;
  }
  float winlossrate() { return win - loss; }
};

struct ValueSum
{
  double win, loss, draw;
  ValueSum();
  ValueSum(double win, double loss, double draw);
  ValueSum(ValueType vt);
  ValueSum inverse();
};
ValueSum operator+(ValueSum a, ValueSum b);
ValueSum operator-(ValueSum a, ValueSum b);
ValueSum operator*(ValueSum a, double b);
ValueSum operator*(double b, ValueSum a);


inline std::string dbg_board(const Color *board)
{
  std::ostringstream os;
  for (int i = 0; i < BS; i++) {
    for (int j = 0; j < BS; j++) {
      switch (board[j + i * BS]) {
      case C_BLACK: os << "X "; break;
      case C_WHITE: os << "O "; break;
      case C_EMPTY: os << ". "; break;
      }
    }
    os << '\n';
  }

  return os.str();
}

typedef int64_t Time;  // value in milliseconds
inline Time     now()
{
  static_assert(sizeof(Time) == sizeof(std::chrono::milliseconds::rep),
                "Time should be 64 bits");

  auto dur = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
}

const int32_t pow3[] =
    {1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441};

namespace strOp {
using namespace std;
string trim(const string &s);

vector<string> split(const string &s);
vector<string> split(const string &s, char delim);
bool           tryStringToInt(const string &str, int &x);
}  // namespace strOp