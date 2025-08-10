#include "../game/gamelogic.h"

/*
 * gamelogic.cpp
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 *
 * Gomoku as a representive
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>


using namespace std;

int Board::findFour(Color color, Loc& loc1, Loc& loc2) const {
  int bestConnection = 0;
  loc1 = NULL_LOC;
  loc2 = NULL_LOC;

  auto checkOne = [&](Loc loc0, int16_t adj) -> void {
    Loc emptyloc1 = Board::NULL_LOC, emptyloc2 = Board::NULL_LOC;
    int emptyCount = 0;
    for (int i = 0; i < 6; i++)
    {
      Loc loc = loc0 + i * adj;
      assert(isOnBoard(loc));
      Color c = loc == firstLoc ? nextPla : colors[loc];
      if(c == getOpp(color))
        return;
      else if (c == C_EMPTY)
      {
        emptyCount += 1;
        if(emptyCount == 1) {
          emptyloc1 = loc;
        } else if(emptyCount == 2) {
          emptyloc2 = loc;
        } else
          return;
      }
      else if (c == color)
      {

      } 
      else
        ASSERT_UNREACHABLE;
    }
    //return 6 - emptyCount;
    int conNum = 6 - emptyCount;
    if(conNum > bestConnection) {
      bestConnection = conNum;
      loc1 = emptyloc1;
      loc2 = emptyloc2;
    }
  };

  
  //+x direction
  { 
    int adj = 1;
    for(int y = 0; y < y_size; y++)
      for (int x = 0; x < x_size - 5; x++)
      {
        Loc loc0 = Location::getLoc(x, y, x_size);
        checkOne(loc0, adj);
      }
  }
  //+y direction
  {
    int adj = x_size + 1;
    for(int y = 0; y < y_size - 5; y++)
      for(int x = 0; x < x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, x_size);
        checkOne(loc0, adj);
      }
  }

  //+x+y direction
  {
    int adj = x_size + 1 + 1;
    for(int y = 0; y < y_size - 5; y++)
      for(int x = 0; x < x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, x_size);
        checkOne(loc0, adj);
      }
  }

  //-x+y direction
  {
    int adj = x_size + 1 - 1;
    for(int y = 0; y < y_size - 5; y++)
      for(int x = 5; x < x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, x_size);
        checkOne(loc0, adj);
      }
  }

  return bestConnection;
}

int Board::findFiveConsideringFirstLoc(Color color, Loc& loc1) const {
  int bestConnection = 0;
  loc1 = NULL_LOC;

  auto checkOne = [&](Loc loc0, int16_t adj) -> void {
    Loc emptyloc1 = Board::NULL_LOC;
    int emptyCount = 0;
    for(int i = 0; i < 6; i++) {
      Loc loc = loc0 + i * adj;
      assert(isOnBoard(loc));
      Color c = loc == firstLoc ? nextPla : colors[loc];
      if(c == getOpp(color))
        return;
      else if(c == C_EMPTY) {
        emptyCount += 1;
        if(emptyCount == 1) 
          emptyloc1 = loc;
        else
          return;
      } else if(c == color) {
      } else
        ASSERT_UNREACHABLE;
    }
    // return 6 - emptyCount;
    int conNum = 6 - emptyCount;
    if(conNum > bestConnection) {
      bestConnection = conNum;
      loc1 = emptyloc1;
    }
  };

  //+x direction
  {
    int adj = 1;
    for(int y = 0; y < y_size; y++)
      for(int x = 0; x < x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, x_size);
        checkOne(loc0, adj);
      }
  }
  //+y direction
  {
    int adj = x_size + 1;
    for(int y = 0; y < y_size - 5; y++)
      for(int x = 0; x < x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, x_size);
        checkOne(loc0, adj);
      }
  }

  //+x+y direction
  {
    int adj = x_size + 1 + 1;
    for(int y = 0; y < y_size - 5; y++)
      for(int x = 0; x < x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, x_size);
        checkOne(loc0, adj);
      }
  }

  //-x+y direction
  {
    int adj = x_size + 1 - 1;
    for(int y = 0; y < y_size - 5; y++)
      for(int x = 5; x < x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, x_size);
        checkOne(loc0, adj);
      }
  }

  return bestConnection;
}

bool Board::isSix(Color color, Loc loc) const {
  if(!isOnBoard(loc))
    return false;
  auto checkOne = [&](int x0, int y0, int dx, int dy) -> bool {
    for(int i = 0; i < 6; i++) {
      int x = x0 + dx * i;
      int y = y0 + dy * i;
      if(x < 0 || y < 0 || x >= x_size || y >= y_size)
        return false;
      Loc loc0 = Location::getLoc(x, y, x_size);
      Color c = loc0 == firstLoc ? nextPla : colors[loc0];
      if(c != color) {
        return false;
      }
    }
    return true;
  };
  int xa = Location::getX(loc, x_size);
  int ya = Location::getY(loc, x_size);
  //+x direction
  { 
    int dx = 1, dy = 0;
    for(int i = 0; i < 6; i++) {
      int x = xa - 5 + i;
      int y = ya;
      if(checkOne(x, y, dx, dy))
        return true;
    }
  }
  //+y direction
  {
    int dx = 0, dy = 1;
    for(int i = 0; i < 6; i++) {
      int x = xa;
      int y = ya - 5 + i;
      if(checkOne(x, y, dx, dy))
        return true;
    }
  }

  //+x+y direction
  {
    int dx = 1, dy = 1;
    for(int i = 0; i < 6; i++) {
      int x = xa - 5 + i;
      int y = ya - 5 + i;
      if(checkOne(x, y, dx, dy))
        return true;
    }
  }

  //-x+y direction
  {
    int dx = -1, dy = 1;
    for(int i = 0; i < 6; i++) {
      int x = xa + 5 - i;
      int y = ya - 5 + i;
      if(checkOne(x, y, dx, dy))
        return true;
    }
  }

  return false;
}





Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc1,
  Loc loc2
  ) {

  if(board.stage == 1) //only check winner after a full move
    return C_WALL;
  
  if((hist.rules.maxMoves != 0 || hist.rules.VCNRule != Rules::VCNRULE_NOVC) && hist.rules.firstPassWin) {
    throw StringError("GameLogic::checkWinnerAfterPlayed: firstPassWin should not be with VCN or maxMoves");
  }

  Player opp = getOpp(pla);
// connection judge
  if(board.isSix(pla, loc1)) {
    return pla;
  }
  if(board.isSix(pla, loc2)) {
    return pla;
  }

  // if the player want to pass one of the two moves, the pass should be the second
  if(loc1 == Board::PASS_LOC && board.isOnBoard(loc2))
    return opp;
  
  int myPassNum = pla == C_BLACK ? board.blackPassNum : board.whitePassNum;
  int oppPassNum = pla == C_WHITE ? board.blackPassNum : board.whitePassNum;

  if(loc1 == Board::PASS_LOC || loc2 == Board::PASS_LOC) {
    if(hist.rules.VCNRule == Rules::VCNRULE_NOVC) {
      if(oppPassNum > 0) {
        if(!hist.rules.firstPassWin)  //normal draw
        {
          return C_EMPTY;
        } 
        else  // �Է���pass
        {
          return opp;
        }
      }
    } 
    else {
      Color VCside = hist.rules.vcSide();
      int VClevel = hist.rules.vcLevel();

      if(VCside == pla)  // VCN����������pass
      {
        return opp;
      } 
      else  // pass�����㹻����ʤ
      {
        if(myPassNum >= 7 - VClevel) {
          return pla;
        }
      }
    }
  }



  // maxmoves�ж�
  if(hist.rules.maxMoves != 0 && board.movenum >= hist.rules.maxMoves) {
    if(hist.rules.VCNRule == Rules::VCNRULE_NOVC) {
      return C_EMPTY;
    } else  // �����н�������
    {
      static_assert(Rules::VCNRULE_VC1_W == Rules::VCNRULE_VC1_B + 10, "Ensure VCNRule%10==N, VCNRule/10+1==color");
      Color VCside = hist.rules.vcSide();
      return getOpp(VCside);
    }
  }

  return C_WALL;
}

GameLogic::ResultsBeforeNN::ResultsBeforeNN() {
  inited = false;
  winner = C_WALL;
  myOnlyLoc = Board::NULL_LOC;
}

void GameLogic::ResultsBeforeNN::init(const Board& board, const BoardHistory& hist, Color nextPlayer) {
  if(inited)
    return;
  inited = true;

  if(board.stage == 1 && board.firstLoc == Board::PASS_LOC) // if the first move is pass, the second move must also be pass, no need to calculate
    return;

  //find whether the player can connect 6 in a single move
  if(board.stage == 0)  // find four
  {
    Loc loc1, loc2;
    int conNum = board.findFour(nextPlayer, loc1, loc2);
    if (conNum >= 6)
    {
      throw StringError("should not call ResultsBeforeNN::init after game finished");
    }
    if(conNum == 5) {
      assert(board.isOnBoard(loc1));
      winner = nextPlayer;
      myOnlyLoc = loc1;
    }
    if(conNum == 4) {
      assert(board.isOnBoard(loc1));
      assert(board.isOnBoard(loc2));
      winner = nextPlayer;
      if(board.getLocationPriority(loc1) <= board.getLocationPriority(loc2))
        myOnlyLoc = loc1;
      else
        myOnlyLoc = loc2;
    }
  }
  else if(board.stage == 1)  // find five
  {
    Loc loc1;
    int conNum = board.findFiveConsideringFirstLoc(nextPlayer, loc1);
    if(conNum >= 6) {
      //have win in the first move, just pass
      winner = nextPlayer;
      myOnlyLoc = Board::PASS_LOC;
    }
    if(conNum == 5) {
      assert(board.isOnBoard(loc1));
      winner = nextPlayer;
      myOnlyLoc = loc1;
    }
  }

  //if using VCN rule and opponent cant make a six, pass
  if (hist.rules.vcSide() == getOpp(nextPlayer)) {
    int VClevel = hist.rules.vcLevel();
    int myPassNum = nextPlayer == C_BLACK ? board.blackPassNum : board.whitePassNum;
    int requirePass = 7 - VClevel - myPassNum;
    if(board.stage == 1 && board.firstLoc != Board::PASS_LOC)
      requirePass += 1;
    if(requirePass <= 2)  // this move will win
    {
      winner = nextPlayer;
      myOnlyLoc = Board::PASS_LOC;
    }
    else if(requirePass <= 4)  // if opponent has no four, pass will win
    {
      Loc loc1, loc2;
      int conNum = board.findFour(getOpp(nextPlayer), loc1, loc2);
      if(conNum == 0) {
        winner = nextPlayer;
        myOnlyLoc = Board::PASS_LOC;
      }
    }
  }



  return;
}

int GameLogic::checkTwoFourThreats(const Board& board, Player pla) {
  Player opp = getOpp(pla);
  vector<vector<Loc>> emptyPositionsInTuples; // 记录有4~5个pla棋子且没opp棋子的六元组的空位
  
  // 记录所有六元组的状态
  bool hasPlaWin = false;     // 是否有全pla的六元组
  
  auto checkTuple = [&](Loc loc0, int16_t adj) -> void {
     int plaCount = 0;
     int oppCount = 0;
     
     for (int i = 0; i < 6; i++) {
       Loc loc = loc0 + i * adj;
       if (!board.isOnBoard(loc))
       {
         ASSERT_UNREACHABLE;//越界判定应在此函数外
         return; // 无效六元组，跳过
       }
       
       Color c = board.colors[loc];
       if (board.stage == 1 && loc == board.firstLoc)
         c = board.nextPla;
       if (c == pla) {
         plaCount++;
       } else if (c == opp) {
         oppCount++;
       }
     }
     
     // 记录各种状态
     if (plaCount == 6) {
       hasPlaWin = true;
     }
     
     // 记录有4~5个pla棋子且没opp棋子的六元组的空位
     if (plaCount >= 4 && plaCount <= 5 && oppCount == 0) {
       vector<Loc> emptyLocs;
       for (int i = 0; i < 6; i++) {
         Loc loc = loc0 + i * adj;
         Color c = board.colors[loc];
         if (c == C_EMPTY && board.firstLoc != loc) {
           emptyLocs.push_back(loc);
         }
       }
       emptyPositionsInTuples.push_back(emptyLocs);
     }
   };
  
  // 遍历所有方向的六元组
  // +x direction (横向)
  for (int y = 0; y < board.y_size; y++) {
    for (int x = 0; x < board.x_size - 5; x++) {
      Loc loc0 = Location::getLoc(x, y, board.x_size);
      checkTuple(loc0, 1);
    }
  }
  
  // +y direction (竖向)
  for (int y = 0; y < board.y_size - 5; y++) {
    for (int x = 0; x < board.x_size; x++) {
      Loc loc0 = Location::getLoc(x, y, board.x_size);
      checkTuple(loc0, board.x_size + 1);
    }
  }
  
  // +x+y direction (正斜向)
  for (int y = 0; y < board.y_size - 5; y++) {
    for (int x = 0; x < board.x_size - 5; x++) {
      Loc loc0 = Location::getLoc(x, y, board.x_size);
      checkTuple(loc0, board.x_size + 1 + 1);
    }
  }
  
  // -x+y direction (反斜向)
  for (int y = 0; y < board.y_size - 5; y++) {
    for (int x = 5; x < board.x_size; x++) {
      Loc loc0 = Location::getLoc(x, y, board.x_size);
      checkTuple(loc0, board.x_size + 1 - 1);
    }
  }
  
  // 按优先级返回结果
  if (hasPlaWin) {
    return 4; // 全是pla的棋子
  }
  
  // 如果没有威胁六元组，返回0
  if (emptyPositionsInTuples.empty()) {
    return 0;
  }
  
  // 检查是否可以用1个或2个棋子堵住所有威胁
  // 找到所有空位并去重
  set<Loc> uniqueEmptyPositions;
  for (const auto& emptyLocs : emptyPositionsInTuples) {
    for (Loc loc : emptyLocs) {
      uniqueEmptyPositions.insert(loc);
    }
  }
  vector<Loc> allEmptyPositions(uniqueEmptyPositions.begin(), uniqueEmptyPositions.end());
  
  // 检查是否有一个位置能堵住所有六元组
  for (Loc candidateLoc : allEmptyPositions) {
    bool canBlockAll = true;
    for (const auto& emptyLocs : emptyPositionsInTuples) {
      bool foundInThisTuple = false;
      for (Loc loc : emptyLocs) {
        if (loc == candidateLoc) {
          foundInThisTuple = true;
          break;
        }
      }
      if (!foundInThisTuple) {
        canBlockAll = false;
        break;
      }
    }
    if (canBlockAll) {
      return 1; // 一个棋子可以堵住所有
    }
  }
  
  // 检查是否用2个棋子可以堵住所有六元组
  for (size_t i = 0; i < allEmptyPositions.size(); i++) {
    for (size_t j = i + 1; j < allEmptyPositions.size(); j++) {
      Loc loc1 = allEmptyPositions[i];
      Loc loc2 = allEmptyPositions[j];
      
      bool canBlockAll = true;
      for (const auto& emptyLocs : emptyPositionsInTuples) {
        bool foundInThisTuple = false;
        for (Loc loc : emptyLocs) {
          if (loc == loc1 || loc == loc2) {
            foundInThisTuple = true;
            break;
          }
        }
        if (!foundInThisTuple) {
          canBlockAll = false;
          break;
        }
      }
      if (canBlockAll) {
        return 2; // 两个棋子可以堵住所有
      }
    }
  }
  
  return 3; // 两个棋子堵不住
}

int GameLogic::checkMaxConnectLen(const Board& board, Player pla) {
  if (board.stage != 0)
    ASSERT_UNREACHABLE;
  int maxLen = 0;
  
  auto checkDirection = [&](Loc startLoc, int16_t adj) -> int {
    int len = 0;
    Loc loc = startLoc;
    for (int i = 0; i < 6; i++)
    {
      if (!board.isOnBoard(loc))
        return 0;
      if (board.colors[loc] == getOpp(pla))
        return 0;
      if (board.colors[loc] == pla)
      {
        len++;
      }
      loc += adj;
    }
    return len;
  };
  
  // 检查所有位置的四个方向
  for (int y = 0; y < board.y_size; y++) {
    for (int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      if (board.colors[loc] == pla) {
        // 横向 (+x direction)
        int len = checkDirection(loc, 1);
        maxLen = max(maxLen, len);
        
        // 竖向 (+y direction)
        len = checkDirection(loc, board.x_size + 1);
        maxLen = max(maxLen, len);
        
        // 正斜向 (+x+y direction)
        len = checkDirection(loc, board.x_size + 1 + 1);
        maxLen = max(maxLen, len);
        
        // 反斜向 (-x+y direction)
        len = checkDirection(loc, board.x_size + 1 - 1);
        maxLen = max(maxLen, len);
      }
    }
  }
  
  return maxLen;
}

vector<Loc> GameLogic::getAllVCFAttackOrDefenseLocs(const Board& board, Player attackPla, Color& winner, int& gameEndMovenum) {
  winner = C_WALL;
  gameEndMovenum = 0;
  Player defendPla = getOpp(attackPla);
  vector<Loc> locs;
  
  if (board.nextPla == attackPla) {
    // 进攻方逻辑
    int maybeLegalMap[Board::MAX_ARR_SIZE] = {0};
    int maxAttackCount = 0;
    int maxDefendCount = 0;
    int fourCount = 0;


    
    auto checkTuple = [&](Loc loc0, int16_t adj) -> void {
      int attackCount = 0;
      int defendCount = 0;
      
      for (int i = 0; i < 6; i++) {
        Loc loc = loc0 + i * adj;
        if (!board.isOnBoard(loc)) {
          ASSERT_UNREACHABLE;
          return; // 无效六元组，跳过
        }
        
        Color c = board.colors[loc];
        if (board.stage == 1 && loc == board.firstLoc)
          c = board.nextPla;
        if (c == attackPla) {
          attackCount++;
        } else if (c == defendPla) {
          defendCount++;
        }
      }
      
      // 更新最大计数
      if (defendCount == 6) {
        maxDefendCount = max(maxDefendCount, defendCount);
      }
      
      if (board.stage == 0) {
        if (attackCount >= 4 && attackCount <= 6 && defendCount == 0) {
          maxAttackCount = max(maxAttackCount, attackCount);
        }
        // 记录可能的合法位置 (2~3个进攻方棋子且无防守方棋子)
        if (attackCount >= 2 && attackCount <= 3 && defendCount == 0) {
          for (int i = 0; i < 6; i++) {
            Loc loc = loc0 + i * adj;
            if (board.colors[loc] == C_EMPTY) {
              maybeLegalMap[loc] += 1;
            }
          }
        }
      } else { // stage == 1
        if (attackCount >= 5 && attackCount <= 6 && defendCount == 0) {
          maxAttackCount = max(maxAttackCount, attackCount);
        }
        if(attackCount == 4 && defendCount == 0) {
          fourCount++;
        }
        // 记录可能的合法位置 (3~4个进攻方棋子且无防守方棋子)
        if (attackCount >= 3 && attackCount <= 4 && defendCount == 0) {
          for (int i = 0; i < 6; i++) {
            Loc loc = loc0 + i * adj;
            if (board.colors[loc] == C_EMPTY && board.firstLoc != loc) {
              maybeLegalMap[loc] += 1;
            }
          }
        }
      }
    };
    
    // 遍历所有方向的六元组
    // +x direction (横向)
    for (int y = 0; y < board.y_size; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, 1);
      }
    }
    
    // +y direction (竖向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1);
      }
    }
    
    // +x+y direction (正斜向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 + 1);
      }
    }
    
    // -x+y direction (反斜向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 5; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 - 1);
      }
    }
    
    // 检查胜负情况
    if (maxDefendCount == 6) {
      winner = defendPla;
      gameEndMovenum = board.movenum;
      return locs;
    }
    
    if (board.stage == 0) {
      if (maxAttackCount >= 4) {
        winner = attackPla;
        gameEndMovenum = board.movenum + 6 - maxAttackCount;
        return locs;
      }
    } else { // stage == 1
      if (maxAttackCount >= 5) {
        winner = attackPla;
        gameEndMovenum = board.movenum + 6 - maxAttackCount;
        return locs;
      }
    }

    int maybeLegalThreshold = 1;
    if (board.stage == 0) {
      maybeLegalThreshold = 0;//maybe there are enough fours
    }

    if(board.stage == 1 && fourCount == 0) {
      maybeLegalThreshold = 2;//must create two fours at once
    }

    if (board.stage == 1 && fourCount >= 2) {
      maybeLegalThreshold = 0;//maybe there are enough fours
    }

    // 收集可能的合法位置
    for (int y = 0; y < board.y_size; y++) {
      for (int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if (maybeLegalMap[loc] >= maybeLegalThreshold && board.colors[loc] == C_EMPTY && loc != board.firstLoc) {
          if (board.stage == 0 || (board.getLocationPriority(loc) + Board::PRIOR_EPS >= board.firstLocPriority)) {
            locs.push_back(loc);
          }
        }
      }
    }
    
    // 如果没有可行位置，防守方获胜
    if (locs.empty()) {
      winner = defendPla;
      gameEndMovenum = board.movenum;
    }
    
  } else {
    // 防守方逻辑
    vector<vector<Loc>> emptyPositionsInTuples; // 记录有4~5个attackPla棋子且没defendPla棋子的六元组的空位
    int maxDefendCount = 0;
    int maxAttackCount = 0;
    
    auto checkTuple = [&](Loc loc0, int16_t adj) -> void {
      int attackCount = 0;
      int defendCount = 0;
      
      for (int i = 0; i < 6; i++) {
        Loc loc = loc0 + i * adj;
        if (!board.isOnBoard(loc)) {
          ASSERT_UNREACHABLE;
          return; // 无效六元组，跳过
        }
        
        Color c = board.colors[loc];
        if (board.stage == 1 && loc == board.firstLoc)
          c = board.nextPla;
        if (c == attackPla) {
          attackCount++;
        } else if (c == defendPla) {
          defendCount++;
        }
      }
      
      if (board.stage == 0) {
        // 更新最大计数
        if (attackCount == 6) {
          maxAttackCount = max(maxAttackCount, attackCount);
        }
        
        // 更新防守方最大计数
        if (defendCount >= 4 && defendCount <= 6 && attackCount == 0) {
          maxDefendCount = max(maxDefendCount, defendCount);
        }
        
        // 记录有4~5个attackPla棋子且没defendPla棋子的六元组的空位
        if (attackCount >= 4 && attackCount <= 5 && defendCount == 0) {
          vector<Loc> emptyLocs;
          for (int i = 0; i < 6; i++) {
            Loc loc = loc0 + i * adj;
            if (board.colors[loc] == C_EMPTY) {
              emptyLocs.push_back(loc);
            }
          }
          emptyPositionsInTuples.push_back(emptyLocs);
        }
      } else { // stage == 1
        // 更新最大计数
        if (attackCount == 6) {
          maxAttackCount = max(maxAttackCount, attackCount);
        }
        
        // 更新防守方最大计数
        if (defendCount >= 5 && defendCount <= 6 && attackCount == 0) {
          maxDefendCount = max(maxDefendCount, defendCount);
        }
        
        // 记录有4~5个attackPla棋子且没defendPla棋子的六元组的空位
        if (attackCount >= 4 && attackCount <= 5 && defendCount == 0) {
          vector<Loc> emptyLocs;
          for (int i = 0; i < 6; i++) {
            Loc loc = loc0 + i * adj;
            if (board.colors[loc] == C_EMPTY && board.firstLoc != loc) {
              emptyLocs.push_back(loc);
            }
          }
          emptyPositionsInTuples.push_back(emptyLocs);
        }
      }
    };
    
    // 遍历所有方向的六元组
    // +x direction (横向)
    for (int y = 0; y < board.y_size; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, 1);
      }
    }
    
    // +y direction (竖向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1);
      }
    }
    
    // +x+y direction (正斜向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 + 1);
      }
    }
    
    // -x+y direction (反斜向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 5; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 - 1);
      }
    }
    
    // 检查胜负情况
    if (board.stage == 0) {
      if (maxAttackCount == 6) {
        winner = attackPla;
        gameEndMovenum = board.movenum;
        return locs;
      }
      if (maxDefendCount >= 4) {
        winner = defendPla;
        gameEndMovenum = board.movenum + 6 - maxDefendCount;
        return locs;
      }
    } else { // stage == 1
      if (maxAttackCount == 6) {
        assert(false); // stage==1时不应该有6个进攻方棋子
      }
      if (maxDefendCount >= 5) {
        winner = defendPla;
        gameEndMovenum = board.movenum + 6 - maxDefendCount;
        return locs;
      }
    }
    
    // 如果没有威胁六元组，返回空
    if (emptyPositionsInTuples.empty()) {
      winner = defendPla;
      gameEndMovenum = board.movenum;

      return locs;
    }

    if (emptyPositionsInTuples.size() == 1 && board.stage == 0) {
      winner = defendPla;
      gameEndMovenum = board.movenum + 1;

      return locs;
    }
    
    // 计算能堵住所有四的位置
    set<Loc> uniqueEmptyPositions;
    for (const auto& emptyLocs : emptyPositionsInTuples) {
      for (Loc loc : emptyLocs) {
        uniqueEmptyPositions.insert(loc);
      }
    }
    vector<Loc> allEmptyPositions(uniqueEmptyPositions.begin(), uniqueEmptyPositions.end());
    
    if (board.stage == 0) {
      // 检查是否用2个棋子可以堵住所有六元组
      for (size_t i = 0; i < allEmptyPositions.size(); i++) {
        Loc loc1 = allEmptyPositions[i];
        bool canBlock = false;
        for (size_t j = 0; j < allEmptyPositions.size(); j++) {
          Loc loc2 = allEmptyPositions[j];
          if (loc2 == loc1)
            continue;
          
          // 检查第二步是否满足优先级要求
          if (!(board.getLocationPriority(loc2) + Board::PRIOR_EPS >= board.getLocationPriority(loc1))) {
            continue;
          }
          
          bool canBlockAll = true;
          for (const auto& emptyLocs : emptyPositionsInTuples) {
            bool foundInThisTuple = false;
            for (Loc loc : emptyLocs) {
              if (loc == loc1 || loc == loc2) {
                foundInThisTuple = true;
                break;
              }
            }
            if (!foundInThisTuple) {
              canBlockAll = false;
              break;
            }
          }
          if (canBlockAll) {
            canBlock = true;
            break;
          }
        }
        if (canBlock) {
          locs.push_back(loc1);
        }
      }
      
      // 如果没有找到能两步堵住的位置，进攻方获胜
      if (locs.empty()) {
        winner = attackPla;
        gameEndMovenum = board.movenum + 4;
      }
    } else { // stage == 1
      // 检查是否有一个位置能堵住所有六元组且满足优先级要求
      for (Loc candidateLoc : allEmptyPositions) {
        if (!(board.getLocationPriority(candidateLoc) + Board::PRIOR_EPS >= board.firstLocPriority)) {
          continue;
        }
        
        bool canBlockAll = true;
        for (const auto& emptyLocs : emptyPositionsInTuples) {
          bool foundInThisTuple = false;
          for (Loc loc : emptyLocs) {
            if (loc == candidateLoc) {
              foundInThisTuple = true;
              break;
            }
          }
          if (!foundInThisTuple) {
            canBlockAll = false;
            break;
          }
        }
        if (canBlockAll) {
          locs.push_back(candidateLoc);
        }
      }
      
      // 如果没有找到能一步堵住的位置，进攻方获胜
      if (locs.empty()) {
        winner = attackPla;
        gameEndMovenum = board.movenum + 3;
      }
    }
  }
  if (winner != C_WALL)
    assert(gameEndMovenum >= board.movenum);
  return locs;
}

Loc GameLogic::findImmediateWinInVCFAttack(const Board& board, Player attackPla) {
  // 只对stage==1且nextPla==attackPla的情况进行搜索
  if (board.stage != 1 || board.nextPla != attackPla) {
    ASSERT_UNREACHABLE;
  }
  
  Color winner = C_WALL;
  int gameEndMovenum = 0;
  
  // 获取所有VCF攻击位置
  vector<Loc> attackLocs = getAllVCFAttackOrDefenseLocs(board, attackPla, winner, gameEndMovenum);
  
  // 如果已经有确定的胜负结果，直接返回
  if (winner != C_WALL) {

    if(winner==attackPla)
      ASSERT_UNREACHABLE;
    else
      return Board::NULL_LOC;
  }
  
  // 对每个攻击位置进行一层搜索
  for (Loc attackLoc : attackLocs) {
    // 创建临时棋盘进行试探
    Board tempBoard = board;
    tempBoard.playMoveAssumeLegal(attackLoc, attackPla);
    
    // 检查走完这步后的局面
    Color tempWinner;
    int tempGameEndMovenum;
    vector<Loc> defenseLocs = getAllVCFAttackOrDefenseLocs(tempBoard, attackPla, tempWinner, tempGameEndMovenum);
    
    // 如果攻击方直接获胜，返回这个位置
    if (tempWinner == attackPla) {
      assert(tempGameEndMovenum == board.movenum + 5);
      return attackLoc;
    }
  }
  
  // 没有找到立即获胜的位置
  return Board::NULL_LOC;
}

Loc GameLogic::findImmediateWinInVCFAttackLayer2(const Board& board, Player attackPla) {
  // 只对stage==0且nextPla==attackPla的情况进行搜索
  if (board.stage != 0 || board.nextPla != attackPla) {
    ASSERT_UNREACHABLE;
  }

  Color winner = C_WALL;
  int gameEndMovenum = 0;

  // 获取所有VCF攻击位置
  vector<Loc> attackLocs = getAllVCFAttackOrDefenseLocs(board, attackPla, winner, gameEndMovenum);

  // 如果已经有确定的胜负结果，直接返回
  if (winner != C_WALL) {
    ASSERT_UNREACHABLE;
  }

  // 对每个攻击位置进行一层搜索
  for (Loc attackLoc : attackLocs) {
    // 创建临时棋盘进行试探
    Board tempBoard = board;
    tempBoard.playMoveAssumeLegal(attackLoc, attackPla);

    // 检查走完这步后的局面
    Loc winLoc = GameLogic::findImmediateWinInVCFAttack(tempBoard, attackPla);

    // 如果攻击方直接获胜，返回这个位置
    if (winLoc != Board::NULL_LOC) {
      return attackLoc;
    }
  }

  // 没有找到立即获胜的位置
  return Board::NULL_LOC;
}