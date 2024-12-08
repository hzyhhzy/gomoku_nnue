#pragma once
#include "NNUEglobal.h"
#include "rules.h"
#include "HashTable/NNUEHashTable.h"
class Board
{
public:

  //record stats for undo
  struct UndoRecord
  {
    int movenum; //how many moves
    //which stage. Normally 0 = choosing piece. 1 = where to place
    int stage;
    //sum of x and y coordinates of all stones
    //to calculate the mean coordinates (the gravity center of all stones)
    //the second location should be further to the gravity center than the first location
    double meanStoneX;//pre-calculate, avoid high frequency division calculation
    double meanStoneY;

    //who plays the next move
    Color nextPla;

    //location of the first stones of the two stones in one move
    NU_Loc firstLoc;
    //square of the distance of firstLoc and gravity center. 0 if numStones==0 or the last move is null or pass (means all next moves are legal)
    //the second loc should be further
    double firstLocPriority;

    int blackPassNum;  // pass count of black/white, used for VCT/VC2
    int whitePassNum;

    UndoRecord(const Board& eva);
  };




  int x_size;                  //Horizontal size of board
  int y_size;                  //Vertical size of board

  Color board[MaxBS * MaxBS];
  Hash128 pos_hash;
  //Rules rule;
  //double noResultUtilityForWhite;

  int movenum; //how many moves
  //which stage. Normally 0 = choosing piece. 1 = where to place
  int stage;
  //sum of x and y coordinates of all stones
  //to calculate the mean coordinates (the gravity center of all stones)
  //the second location should be further to the gravity center than the first location
  uint32_t sumStoneX;//use int, avoid float accuracy loss
  uint32_t sumStoneY;
  uint32_t numStones;
  double meanStoneX;//pre-calculate, avoid high frequency division calculation
  double meanStoneY;

  //who plays the next move
  Color nextPla;

  //location of the first stones of the two stones in one move
  NU_Loc firstLoc;
  //square of the distance of firstLoc and gravity center. 0 if numStones==0 or the last move is null or pass (means all next moves are legal)
  //the second loc should be further
  double firstLocPriority;

  int blackPassNum;  // pass count of black/white, used for VCT/VC2
  int whitePassNum;




  Board(int x_size, int y_size);

  bool loadParam(std::string filepathB, std::string filepathW);
  void clear();

  void setStone(Color color, NU_Loc loc);
  void play(Color color, NU_Loc loc);
  void undo(const UndoRecord& ur, Color color, NU_Loc loc);

  void updateGfInput(float* gf, Color nextPlayer);

  bool isLegal(NU_Loc loc, Color pla) const;
  void checkConsistency() const;
  Hash128 getHash() const;


  //square of the distance of loc and gravity center. 0 if null_loc or pass_loc (means all next moves are legal)
  double getLocationPriority(int x, int y) const;
  double getLocationPriority(NU_Loc loc) const;

  NNUE::ValueType evaluateFull(const float* gf, Color color, NNUE::PolicyType* policy)
  {
    clearCache(color);
    if (color == C_BLACK)
      return blackEvaluator->evaluateFull(gf, NULL, policy);
    else
      return whiteEvaluator->evaluateFull(gf, NULL, policy);
  }
  void evaluatePolicy(const float* gf, Color color, NNUE::PolicyType* policy)
  {
    clearCache(color);
    if (color == C_BLACK)
      blackEvaluator->evaluatePolicy(gf, NULL, policy);
    else
      whiteEvaluator->evaluatePolicy(gf, NULL, policy);
  }
  NNUE::ValueType evaluateValue(const float* gf, Color color)
  {
    clearCache(color);
    if (color == C_BLACK)
      return blackEvaluator->evaluateValue(gf, NULL);
    else
      return whiteEvaluator->evaluateValue(gf, NULL);
  }




  //Color* board() const { return blackEvaluator->board; }

private:

  //每次调用play或者undo时，先不在EvaluatorOneSide里面走，因为开销很大。先缓存。
  //MCTS的时候，经常“走回头路”，使用cache可以提速。
  struct MoveCache
  {
    bool isUndo;
    Color color;
    NU_Loc loc;
    MoveCache() :isUndo(false), color(C_EMPTY), loc(NNUE::NU_LOC_NULL) {}
    MoveCache(bool isUndo, Color color, NU_Loc loc) :isUndo(isUndo), color(color), loc(loc) {}
  };

  MoveCache moveCacheB[MaxBS * MaxBS], moveCacheW[MaxBS * MaxBS];
  int moveCacheBlength, moveCacheWlength;

  void clearCache(Color color);//把所有缓存的步数清空，使得evaluatorOneSide的board与这里的board相同
  void addCache(bool isUndo, Color color, NU_Loc loc);
  bool isContraryMove(MoveCache a, MoveCache b);//是不是可以抵消的一对操作
};

