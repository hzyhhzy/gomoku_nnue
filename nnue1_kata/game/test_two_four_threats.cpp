#include "gamelogic.h"
#include "board.h"
#include <iostream>
#include <cassert>

using namespace std;

void testCheckTwoFourThreats() {
    cout << "Testing checkTwoFourThreats function..." << endl;
    
    // 创建一个19x19的棋盘
    Board board(19, 19);
    
    // 测试1: 空棋盘应该返回0
    int result = GameLogic::checkTwoFourThreats(board, C_BLACK);
    cout << "Empty board result: " << result << " (expected: 0)" << endl;
    assert(result == 0);
    
    // 测试2: 创建一个有6个连续黑子的情况，应该返回4
    for (int i = 0; i < 6; i++) {
        Loc loc = Location::getLoc(i, 0, 19);
        board.setStone(loc, C_BLACK);
    }
    result = GameLogic::checkTwoFourThreats(board, C_BLACK);
    cout << "Six black stones in a row result: " << result << " (expected: 4)" << endl;
    assert(result == 4);
    
    // 重置棋盘
    board = Board(19, 19);
    
    // 测试3: 创建一个有6个连续白子的情况，应该返回-2
    for (int i = 0; i < 6; i++) {
        Loc loc = Location::getLoc(i, 1, 19);
        board.setStone(loc, C_WHITE);
    }
    result = GameLogic::checkTwoFourThreats(board, C_BLACK);
    cout << "Six white stones in a row result: " << result << " (expected: -2)" << endl;
    assert(result == -2);
    
    // 重置棋盘
    board = Board(19, 19);
    
    // 测试4: 创建一个有4个白子没有黑子的六元组，应该返回-1
    for (int i = 0; i < 4; i++) {
        Loc loc = Location::getLoc(i, 2, 19);
        board.setStone(loc, C_WHITE);
    }
    result = GameLogic::checkTwoFourThreats(board, C_BLACK);
    cout << "Four white stones, no black result: " << result << " (expected: -1)" << endl;
    assert(result == -1);
    
    // 重置棋盘
    board = Board(19, 19);
    
    // 测试5: 创建一个有4个黑子没有白子的六元组，应该记录威胁
    for (int i = 0; i < 4; i++) {
        Loc loc = Location::getLoc(i, 3, 19);
        board.setStone(loc, C_BLACK);
    }
    result = GameLogic::checkTwoFourThreats(board, C_BLACK);
    cout << "Four black stones threat result: " << result << " (expected: 1, 2, or 3)" << endl;
    assert(result >= 1 && result <= 3);
    
    cout << "All tests passed!" << endl;
}

int main() {
    testCheckTwoFourThreats();
    return 0;
}