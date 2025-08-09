#pragma once
#include "../nnue/NNUEBoardHistory.h"
#include "../game/gamelogic.h"
#include "../search/mutexpool.h"
#include <unordered_map>
#include <memory>
#include <vector>

namespace NNUE_VCF_MCTSsearch {

const double policyQuant = 50000;
const double policyQuantInv = 1/policyQuant;

struct MCTSnode;
class MCTSsearch;

class MCTS_CacheTable {
public:
  struct Entry {
    Hash128 hash;
    Color maybeWinner;           // Result from getAllVCFAttackOrDefenseLocs
    int gameEndMovenum;          // Game end move number from VCF
    NNUE::ValueType nnueValue;   // NNUE evaluation result
    std::vector<std::pair<Loc, float>> legalMovesWithPolicy; // Legal moves with policy
    Loc bestMove;                // Best move for this position
    Entry();
    ~Entry();
  };
private:
  Entry* entries;
  MutexPool* mutexPool;
  uint64_t tableSize;
  uint64_t tableMask;
  uint32_t mutexPoolMask;

public:
  MCTS_CacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);
  ~MCTS_CacheTable();

  MCTS_CacheTable(const MCTS_CacheTable& other) = delete;
  MCTS_CacheTable& operator=(const MCTS_CacheTable& other) = delete;

  //These are thread-safe. For get, ret will be set to default upon a failure to find.
  bool get(Hash128 nnHash, Entry& ret);
  void set(const Entry& entry);
  void clear();
};












struct MCTSchild {
    MCTSnode* ptr;
    Loc loc;
    uint16_t policy; // Original policy multiplied by policyQuant
};

struct MCTSnode {
    // Node structure
    int16_t childrennum;
    int16_t legalChildrennum;
    MCTSchild* children;
    
    uint64_t visits;
    double WRtotal;  // Attacker win rate minus loss rate and draw rate
    Color nextColor;
    
    // Win/loss determination
    bool isWinDetermined;        // Whether this node's outcome is determined
    Color winner;                // Winner color (C_WALL if undetermined)
    int stepsToWin;              // Steps to win/loss (positive for win, negative for loss)
    
    MCTSnode(MCTSsearch* search, Color nextColor, double policyTemp);
    ~MCTSnode();
};

class MCTSsearch {
public:

  static const int IMMEDIATE_WIN_SEARCH_LAYERS = 0;//0(disable) or 1(search 1 layer) or 2(search 2 layers). 2 is too slow, 0 or 1 seems to be the best

    MCTS_CacheTable* cacheTable;
    
    
    MCTSnode* rootNode;
    NNUEBoardHistory* boardHistory;
    Player attackPlayer;  // For VCF search
    
    std::atomic_bool terminate;
    
    struct Option {
        int64_t maxNodes = 0;
    } option;
    
    struct Param {
        double expandFactor = 0.2;
        double puct = 2.0;
        double puctPow = 0.75;
        double puctBase = 10;
        double fpuReduction = 0.1;
        double policyTemp = 1.1;
        double localPolicyBonusStage1 = 0.0;
    } params;
    
    MCTSsearch(MCTS_CacheTable* cacheTable, NNUEBoardHistory* hist, Player attackPla);
    ~MCTSsearch();
    
    float fullsearch(Color color, int64_t  maxVisits, Loc& bestmove);
    void play(Color color, Loc loc);
    void undo();
    void clearBoard();
    
    Loc bestRootMove() const;
    float getRootValue() const;
    int64_t getRootVisit() const;
    std::vector<std::pair<Loc, uint64_t>> getPV() const;  // Get principal variation with visit counts
    void stop() { terminate.store(true, std::memory_order_relaxed); }
    
    void setOptions(size_t maxNodes) { option.maxNodes = maxNodes; }
    void loadParamFile(std::string filename);
    

    
    struct SearchResult {
        uint64_t newVisits;
        double WRchange;  // Value change from attacker's perspective
    };
    
    SearchResult search(MCTSnode* node, uint64_t remainVisits, bool isRoot);
    int selectChildIDToSearch(MCTSnode* node);
    std::vector<std::pair<Loc, float>> getLegalMovesAndVCFResultWithPolicy(Color color, Color& maybeWinner, int& gameEndMovenum);
    NNUE::ValueType evaluatePosition(Color color);
    std::pair<Color, int64_t> checkWinnerDetermined(const MCTSnode* node) const;
};

} // namespace NNUE_VCF_MCTSsearch