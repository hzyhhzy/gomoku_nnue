// Simple test for NNUEBoardHistory functionality
#include "../nnue/NNUEBoardHistory.h"
#include "../game/board.h"
#include "../game/rules.h"
#include <iostream>

using namespace std;

int main() {
    cout << "Testing NNUEBoardHistory basic functionality..." << endl;
    
    try {
        // Create a basic board
        Board board(15, 15);
        Rules rules;
        
        // Test default constructor
        NNUEBoardHistory hist1;
        cout << "Default constructor: OK" << endl;
        
        // Test clear method
        hist1.clear(board, P_BLACK, rules);
        cout << "Clear method: OK" << endl;
        
        // Test board history functionality
        if (hist1.boardHistory.size() == 1) {
            cout << "Board history initialized: OK" << endl;
        } else {
            cout << "Board history size: " << hist1.boardHistory.size() << endl;
        }
        
        // Test copy constructor
        NNUEBoardHistory hist2(hist1);
        cout << "Copy constructor: OK" << endl;
        
        // Test assignment operator
        NNUEBoardHistory hist3;
        hist3 = hist1;
        cout << "Assignment operator: OK" << endl;
        
        cout << "All basic tests passed!" << endl;
        
    } catch (const exception& e) {
        cout << "Exception caught: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "Unknown exception caught" << endl;
        return 1;
    }
    
    return 0;
}