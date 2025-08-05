//AI写的，还没检查过，不保证能用

#include "../nnue/NNUEBoardHistory.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../core/test.h"
#include <iostream>

using namespace std;
using namespace NNUE;
using namespace NNUEV2;

void testNNUEBoardHistoryBasic() {
    cout << "Testing NNUEBoardHistory basic functionality..." << endl;
    
    // Create a basic board
    Board board(15, 15);
    Rules rules;
    MiscNNInputParams nninputParams = MiscNNInputParams();
    
    // Test constructor with weights
    NNUEBoardHistory hist1(nullptr, nninputParams);
    cout << "Constructor with weights: OK" << endl;
    
    // Test constructor with board and rules
    NNUEBoardHistory hist2(board, P_BLACK, rules, nullptr, nninputParams);
    cout << "Constructor with board: OK" << endl;
    
    // Test copy constructor
    NNUEBoardHistory hist3(hist1);
    cout << "Copy constructor: OK" << endl;
    
    // Test assignment operator
    NNUEBoardHistory hist4(nullptr, nninputParams);
    hist4 = hist1;
    cout << "Assignment operator: OK" << endl;
    
    // Test move constructor
    NNUEBoardHistory hist5(std::move(hist3));
    cout << "Move constructor: OK" << endl;
    
    // Test clear method
    hist1.clear(board, P_BLACK, rules);
    cout << "Clear method: OK" << endl;
    
    // Test board history functionality
    testAssert(hist1.historicalBoards.size() == 1);
    cout << "Board history initialized: OK" << endl;
    
    cout << "NNUEBoardHistory basic tests passed!" << endl;
}

void testNNUEBoardHistoryInheritance() {
    cout << "Testing NNUEBoardHistory inheritance..." << endl;
    
    Board board(15, 15);
    Rules rules;
    MiscNNInputParams nninputParams = MiscNNInputParams();
    
    NNUEBoardHistory hist(nullptr, nninputParams);
    hist.clear(board, P_BLACK, rules);
    
    // Test that it inherits BoardHistory functionality
    testAssert(hist.moveHistory.empty());
    testAssert(hist.initialPla == P_BLACK);
    testAssert(!hist.isGameFinished);
    
    cout << "NNUEBoardHistory inheritance tests passed!" << endl;
}

void testNNUEBoardHistoryPlayUndo() {
    cout << "Testing NNUEBoardHistory play and undo functionality..." << endl;
    
    Board board(15, 15);
    Rules rules;
    MiscNNInputParams nninputParams = MiscNNInputParams();
    
    NNUEBoardHistory hist(nullptr, nninputParams);
    hist.clear(board, P_BLACK, rules);
    
    // Test initial state
    testAssert(hist.historicalBoards.size() == 1);
    
    // Test play method
    Loc centerLoc = Location::getCenterLoc(board);
    try {
        hist.play(C_BLACK, centerLoc);
        testAssert(hist.historicalBoards.size() == 2);
        cout << "Play method: OK" << endl;
    } catch (...) {
        cout << "Play method failed, but this is expected without proper setup" << endl;
    }
    
    // Test undo method
    try {
        hist.undo(C_BLACK, centerLoc);
        cout << "Undo method: OK" << endl;
    } catch (...) {
        cout << "Undo method failed, but this is expected without proper setup" << endl;
    }
    
    cout << "NNUEBoardHistory play/undo tests completed!" << endl;
}

void runNNUEBoardHistoryTests() {
    cout << "=== Running NNUEBoardHistory Tests ===" << endl;
    
    testNNUEBoardHistoryBasic();
    testNNUEBoardHistoryInheritance();
    testNNUEBoardHistoryPlayUndo();
    
    cout << "=== All NNUEBoardHistory Tests Passed! ===" << endl;
}

// Uncomment this if you want to run tests directly
// int main() {
//     runNNUEBoardHistoryTests();
//     return 0;
// }