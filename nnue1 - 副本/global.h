#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <numeric>
#include <algorithm>
const int BS = 15;
typedef uint8_t Color;
const Color C_EMPTY = 0;
const Color C_BLACK = 1;
const Color C_WHITE = 2;
const Color C_MY = 1;
const Color C_OPP = 2;

typedef uint16_t Loc;
const Loc NULL_LOC = BS * BS + 1;
const Loc PASS_LOC = BS * BS ;

typedef int32_t PolicyType;
const PolicyType MIN_POLICY = -5e8;
const PolicyType MYFIVE_POLICY = 1e8;
const PolicyType OPPFOUR_POLICY = 1e7;
const PolicyType MYFOUR_POLICY = 1e6;

const double WIN_VALUE = 1;
const double LOSE_VALUE = -1;

const double quantFactor = 32;