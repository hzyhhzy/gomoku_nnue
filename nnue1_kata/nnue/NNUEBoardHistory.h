#ifndef NNUE_NNUEBOARDHISTORY_H_
#define NNUE_NNUEBOARDHISTORY_H_

#include "../game/boardhistory.h"
#include "../game/board.h"
#include "Eva_nnuev2.h"
#include "../neuralnet/nninputs.h"

#include <vector>

using namespace NNUE;
using namespace NNUEV2;

//A BoardHistory that maintains NNUE evaluators and replaces NNUEBoard functionality
class NNUEBoardHistory : public BoardHistory {
public:
    // NNUE evaluators for both players
    Eva_nnuev2 blackEvaluator;
    Eva_nnuev2 whiteEvaluator;
    
    // Input buffers for neural network
    MiscNNInputParams nnInputParams;
    float gfInputBuf[NNUEV2::globalFeatureNum];
    bool illegalMapBuf[MaxBS * MaxBS];
    
    // Historical boards for undo operations
    std::vector<Board> historicalBoards;
    
    // Move cache for efficient undo operations
    struct MoveCache {
        bool isUndo;
        Color color;
        Loc loc;
        MoveCache() : isUndo(false), color(C_EMPTY), loc(Board::NULL_LOC) {}
        MoveCache(bool isUndo, Color color, Loc loc) : isUndo(isUndo), color(color), loc(loc) {}
    };
    
    MoveCache moveCacheB[MaxBS * MaxBS], moveCacheW[MaxBS * MaxBS];
    int moveCacheBlength, moveCacheWlength;

    // Constructors
    NNUEBoardHistory(const ModelWeight* weights, const MiscNNInputParams& nnInputParams);
    NNUEBoardHistory(const Board& board, Player pla, const Rules& rules, const ModelWeight* weights, const MiscNNInputParams& nnInputParams);
    
    // Copy and move constructors
    NNUEBoardHistory(const NNUEBoardHistory& other);
    NNUEBoardHistory& operator=(const NNUEBoardHistory& other);
    NNUEBoardHistory(NNUEBoardHistory&& other) noexcept;
    NNUEBoardHistory& operator=(NNUEBoardHistory&& other) noexcept;
    
    // Destructor
    ~NNUEBoardHistory();
    
    // Initialize with model weights
    void initializeEvaluators(const ModelWeight* weights);
    
    // Clear and reset the board history with NNUE state
    void clear(const Board& board, Player pla, const Rules& rules);
    
    // NNUEBoard replacement methods
    void play(Color color, Loc loc);
    void undo(Color color, Loc loc);
    

    
    // Update input buffers for neural network evaluation
    void updateInputBuf(Color nextPlayer);
    
    // Neural network evaluation functions
    NNUE::ValueType evaluateFull(Color color, NNUE::PolicyType* policy);
    void evaluatePolicy(Color color, NNUE::PolicyType* policy);
    NNUE::ValueType evaluateValue(Color color);
    
private:
    // Add move to cache for efficient undo
    void addCache(bool isUndo, Color color, Loc loc);
    
    // Clear cache for specific color
    void clearCache(Color color);
    
    // Check if two moves are contrary (undo each other)
    bool isContraryMove(MoveCache a, MoveCache b);
    
    // Internal helper to sync NNUE state with board state
    void syncNNUEWithBoard(const Board& board);
};

#endif // NNUE_NNUEBOARDHISTORY_H_