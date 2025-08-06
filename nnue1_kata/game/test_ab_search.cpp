#include "gamelogic.h"
#include "../nnue/NNUE_VCF_ABSearch.h"
#include <iostream>
#include <cassert>

using namespace std;
using namespace GameLogic;
using namespace NNUE;

int main() {
    cout << "Testing checkMaxConnectLen function..." << endl;
    
    // 创建一个简单的棋盘进行测试
    Board board(15, 15);
    
    // 测试空棋盘
    int result = checkMaxConnectLen(board, C_BLACK);
    assert(result == 0);
    cout << "Empty board test passed: " << result << endl;
    
    // 测试单个棋子
    board.setStone(Location::getLoc(7, 7, 15), C_BLACK);
    result = checkMaxConnectLen(board, C_BLACK);
    assert(result == 1);
    cout << "Single stone test passed: " << result << endl;
    
    // 测试连续三个棋子（横向）
    board.setStone(Location::getLoc(8, 7, 15), C_BLACK);
    board.setStone(Location::getLoc(9, 7, 15), C_BLACK);
    result = checkMaxConnectLen(board, C_BLACK);
    assert(result == 3);
    cout << "Three consecutive stones test passed: " << result << endl;
    
    // 测试连续四个棋子（竖向）
    board.setStone(Location::getLoc(5, 5, 15), C_WHITE);
    board.setStone(Location::getLoc(5, 6, 15), C_WHITE);
    board.setStone(Location::getLoc(5, 7, 15), C_WHITE);
    board.setStone(Location::getLoc(5, 8, 15), C_WHITE);
    result = checkMaxConnectLen(board, C_WHITE);
    assert(result == 4);
    cout << "Four consecutive stones test passed: " << result << endl;
    
    // 测试斜向连接
    Board board2(15, 15);
    board2.setStone(Location::getLoc(3, 3, 15), C_BLACK);
    board2.setStone(Location::getLoc(4, 4, 15), C_BLACK);
    board2.setStone(Location::getLoc(5, 5, 15), C_BLACK);
    board2.setStone(Location::getLoc(6, 6, 15), C_BLACK);
    board2.setStone(Location::getLoc(7, 7, 15), C_BLACK);
    result = checkMaxConnectLen(board2, C_BLACK);
    assert(result == 5);
    cout << "Five diagonal stones test passed: " << result << endl;
    
    cout << "Testing checkTwoFourThreats function (modified version)..." << endl;
    
    // 测试修改后的checkTwoFourThreats函数
    Board board3(15, 15);
    result = checkTwoFourThreats(board3, C_BLACK);
    assert(result == 0);
    cout << "Empty board checkTwoFourThreats test passed: " << result << endl;
    
    // 测试六个连续的黑子
    for (int i = 0; i < 6; i++) {
        board3.setStone(Location::getLoc(i, 0, 15), C_BLACK);
    }
    result = checkTwoFourThreats(board3, C_BLACK);
    assert(result == 4);
    cout << "Six consecutive black stones test passed: " << result << endl;
    
    cout << "All tests passed!" << endl;
    
    // 注意：VCF_ABSearch类需要NNUEBoardHistory和模型权重，
    // 这里只是展示如何使用，实际测试需要完整的NNUE环境
    cout << "VCF_ABSearch class is ready for use with proper NNUE setup." << endl;
    
    return 0;
}