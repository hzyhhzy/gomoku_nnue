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

const int MaxBS = 15;

const int MAX_MCTS_CHILDREN = 32;

const float policyQuantFactor = 32;

namespace Rules {

const int BASICRULE_FREESTYLE = 0;
const int BASICRULE_STANDARD  = 1;
const int BASICRULE_RENJU     = 2;

const int VCNRULE_NOVC  = 0;
const int VCNRULE_VC1_B = 1;
const int VCNRULE_VC2_B = 2;
const int VCNRULE_VC3_B = 3;
const int VCNRULE_VC4_B = 4;
const int VCNRULE_VC1_W = 11;
const int VCNRULE_VC2_W = 12;
const int VCNRULE_VC3_W = 13;
const int VCNRULE_VC4_W = 14;
}  // namespace Rules

const int DEFAULT_RULE = Rules::BASICRULE_FREESTYLE;  


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
static constexpr Loc LOC_PASS = MaxBS * MaxBS;

enum MCTSsureResult : int16_t { MC_Win = 1, MC_LOSE = -1, MC_DRAW = 2, MC_UNCERTAIN = 0 };
inline Loc           MakeLoc(int x, int y) { return Loc(x + y * MaxBS); }
inline std::string   locstr(Loc loc)
{
  return std::string(1, char('A' + loc % MaxBS)) + std::to_string(int(MaxBS - loc / MaxBS));
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
  for (int i = 0; i < MaxBS; i++) {
    for (int j = 0; j < MaxBS; j++) {
      switch (board[j + i * MaxBS]) {
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