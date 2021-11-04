#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

const int BS = 15;

typedef uint64_t Key;

enum Color : int8_t { C_EMPTY = 0, C_BLACK = 1, C_WHITE = 2, C_MY = 1, C_OPP = 2 };
constexpr Color operator~(Color c) { return Color(3 - c); }

enum Loc : int16_t { ZERO_LOC = 0, NULL_LOC = BS * BS + 1, PASS_LOC = BS * BS };
inline Loc           MakeLoc(int x, int y) { return Loc(x + y * BS); }
inline Loc &         operator++(Loc &loc) { return loc = Loc(loc + 1); }
inline std::ostream &operator<<(std::ostream &os, Loc loc)
{
  return os << char('A' + loc % BS) << int(1 + loc / BS);
}
inline std::ostream &operator<<(std::ostream &os, std::vector<Loc> pv)
{
  for (size_t i = 0; i < pv.size(); i++)
    os << " " + !i << pv[i];
  return os;
}

typedef int32_t  PolicyType;
const PolicyType MIN_POLICY     = -5e8;
const PolicyType MYFIVE_POLICY  = 1e8;
const PolicyType OPPFOUR_POLICY = 1e7;
const PolicyType MYFOUR_POLICY  = 1e6;

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

const int32_t pow3[] =
    {1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441};