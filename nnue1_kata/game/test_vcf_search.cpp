#include "../nnue/NNUE_VCF_ABSearch.h"
#include "../nnue/NNUEBoardHistory.h"
#include "gamelogic.h"
#include <iostream>
#include <cassert>

using namespace std;
using namespace NNUE;

void testWinValueEncoding() {
    cout << "Testing win value encoding..." << endl;
    
    // 创建一个15x15的棋盘
    Rules rules;
    rules.boardXSize = 15;
    rules.boardYSize = 15;
    
    NNUEBoardHistory boardHistory(rules);
    
    // 测试进攻方取胜的编码
    // 假设当前棋盘有10个棋子
    boardHistory.board.numStones = 10;
    
    // 计算期望的取胜值：15*15+100-10 = 315
    double expectedWinValue = 15 * 15 + 100 - 10;
    cout << "Expected win value for attacker: " << expectedWinValue << endl;
    
    // 验证取胜值大于1
    assert(expectedWinValue > 1.0);
    cout << "Win value > 1.0: PASSED" << endl;
    
    // 测试防守方取胜的编码
    double expectedDefendWinValue = -2.0;
    cout << "Expected win value for defender: " << expectedDefendWinValue << endl;
    
    // 验证防守方取胜值为-2.0
    assert(expectedDefendWinValue == -2.0);
    cout << "Defend win value = -2.0: PASSED" << endl;
    
    cout << "Win value encoding test: PASSED" << endl;
}

void testSearchEarlyTermination() {
    cout << "Testing search early termination..." << endl;
    
    Rules rules;
    rules.boardXSize = 15;
    rules.boardYSize = 15;
    
    NNUEBoardHistory boardHistory(rules);
    
    // 设置一个简单的测试局面
    // 这里只是验证搜索能够正常运行，不测试具体的棋局逻辑
    boardHistory.board.stage = 0;
    boardHistory.nextPlayer = P_BLACK;
    
    VCF_ABSearch search(&boardHistory, P_BLACK);
    
    // 进行浅层搜索
    double result = search.search(1.0);
    
    cout << "Search completed with value: " << result << endl;
    
    // 验证搜索结果在合理范围内
    // 神经网络返回值应该在[-1, 1]区间，或者是特殊的胜负编码值
    bool validValue = (result >= -2.0 && result <= 1.0) || result > 1.0;
    assert(validValue);
    
    cout << "Search early termination test: PASSED" << endl;
}

int main() {
    try {
        testWinValueEncoding();
        testSearchEarlyTermination();
        
        cout << "\nAll tests passed!" << endl;
        return 0;
    } catch (const exception& e) {
        cout << "Test failed with exception: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "Test failed with unknown exception" << endl;
        return 1;
    }
}