#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <numeric>
#include <algorithm>
const int BS = 15;
typedef int8_t Color;
const Color C_EMPTY = 0;
const Color C_BLACK = 1;
const Color C_WHITE = 2;
const Color C_MY = 1;
const Color C_OPP = 2;

typedef int16_t Loc;
const Loc NULL_LOC = BS * BS + 1;
const Loc PASS_LOC = BS * BS ;

typedef int32_t PolicyType;
const PolicyType MIN_POLICY = -5e8;
const PolicyType MYFIVE_POLICY = 1e8;
const PolicyType OPPFOUR_POLICY = 1e7;
const PolicyType MYFOUR_POLICY = 1e6;


const double WIN_VALUE = 1;
const double LOSE_VALUE = -1;

const float quantFactor = 32;

struct ValueType
{
  float win, loss, draw;
  ValueType(float win, float loss, float draw) :win(win), loss(loss), draw(draw) { self_softmax(); }
  void self_softmax()
  {
    //std::cout<<win<<" "<<loss<<" "<<draw << std::endl;
    float maxvalue = std::max(std::max(win, loss), draw);
    win = exp(win - maxvalue);
    loss = exp(loss - maxvalue);
    draw = exp(draw - maxvalue);
    float total = win + loss + draw;
    win = win / total;
    loss = loss / total;
    draw = draw / total;
  }
  float winlossrate()
  {
    return win -loss;
  }
};







const int32_t pow3[] = { 1,3,9,27,81,243,729,2187,6561,19683,59049,177147,531441 };