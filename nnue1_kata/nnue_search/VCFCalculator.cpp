#include "VCFCalculator.h"
#include "../neuralnet/nninputs.h"
#include "../game/rules.h"
#include "../nnue_search/VCFLogic.h"

using namespace NNUE_VCF_MCTSsearch;
using namespace NNUE;
using namespace NNUEV2;

VCFPrunedInfo::VCFPrunedInfo()
    : loc(Board::NULL_LOC), isPruned(false), moveNum(0), notPruned(false), calculateFactor(0), value(0) {
}

VCFPrunedInfo::VCFPrunedInfo(Loc loc, bool isPruned, int16_t moveNum, bool notPruned, float calculateFactor, float value)
    : loc(loc), isPruned(isPruned), moveNum(moveNum), notPruned(notPruned), calculateFactor(calculateFactor), value(value) {
}

VCFCalculator::VCFCalculator(MCTS_CacheTable* cacheTable, const ModelWeight* weights)
    : cacheTable(cacheTable), weights(weights), nnueHistory1(weights, MiscNNInputParams(), true) {
}

VCFCalculator::~VCFCalculator() {
    // No cleanup needed for pointer members (not owned by this class)
}

std::vector<VCFPrunedInfo> VCFCalculator::CalculateAllVCFDefendResults(
    const Board& board, 
    Color attackPlayer, 
    int maxMove, 
    double searchFactor) {
    
    // Route to appropriate stage-specific function based on board stage
    if (board.stage == 1) {
        return CalculateAllVCFDefendResults_stage1(board, attackPlayer, maxMove, searchFactor);
    } else {
        assert(board.stage == 0);
        return CalculateAllVCFDefendResults_stage0(board, attackPlayer, maxMove, searchFactor);
    }
}

std::vector<VCFPrunedInfo> VCFCalculator::CalculateAllVCFDefendResults_stage0(
    const Board& board,
    Color attackPlayer,
    int maxMove,
    double searchFactor) {

    Color defendPlayer = getOpp(attackPlayer);
    assert(board.stage == 0);
    assert(board.nextPla == defendPlayer);
    
    //check whether defend player can directly win
    {
        int defendMaxLenNow = VCFLogic::checkMaxConnectLen(board, defendPlayer);
        assert(defendMaxLenNow < 6);
        if (defendMaxLenNow - board.stage >= 4)//defend pla can win
        {
            return std::vector<VCFPrunedInfo>();
        }
    }


    Board boardWith2pass=board;
    boardWith2pass.playMoveAssumeLegal(Board::PASS_LOC,board.nextPla);
    boardWith2pass.playMoveAssumeLegal(Board::PASS_LOC,board.nextPla);

    // Calculate shortest VCF and dependency map for boardWith1pass
    std::vector<int8_t> dependMap1;
    int vcfSteps1 = calculateShortestVCFAndDependMap(boardWith2pass, attackPlayer, maxMove, 0, 0, searchFactor, dependMap1, false);
    
    std::vector<VCFPrunedInfo> results;
    
    if (vcfSteps1 <= 0) { //can't VCF even with 2 pass
        return results;
    }

    // If VCF found, analyze all legal positions for pruning
    for (int y1 = 0; y1 < board.y_size; y1++) {
        for (int x1 = 0; x1 < board.x_size; x1++) {
            Loc loc1 = Location::getLoc(x1, y1, board.x_size);

            // Check if position is legal and meets priority requirements
            if (!(board.isLegal(loc1, board.nextPla))) {
                continue;
            }

            Board board1 = board;
            board1.playMoveAssumeLegal(loc1, board1.nextPla);

            int8_t dependValue1 = dependMap1[loc1];

            std::vector<int8_t> dependMap2;
            int vcfSteps2=0;
            //if dependValue==1, the old vcf is not changed, but dependMap changed
            //if dependValue==2, the old vcf is changed, recalculate VCF for this position
            if (dependValue1 >= 1) {
                // Calculate VCF for the test board
                Board board1With1pass=board1;
                board1With1pass.playMoveAssumeLegal(Board::PASS_LOC,board1With1pass.nextPla);
                vcfSteps2 = calculateShortestVCFAndDependMap(board1With1pass, attackPlayer, maxMove, vcfSteps1, vcfSteps1, searchFactor, dependMap2, false);
                if(vcfSteps2<=0)
                {
                    //VCF not possible after playing this move - position is not pruned
                    continue;
                }
            }
            else{ //the old vcf and dependMap is completely not influenced
                dependMap2=dependMap1;
                vcfSteps2=vcfSteps1;
            }


            bool canDefend = false;
            int longestDefense = vcfSteps2; //initially the vcf win move of pass

            //search all children

            for (int y2 = 0; y2 < board.y_size; y2++) {
                if(canDefend)
                    break;

                for (int x2 = 0; x2 < board.x_size; x2++) {
                    Loc loc2 = Location::getLoc(x2, y2, board.x_size);
                    
                    // Check if position is legal and meets priority requirements
                    if (!(board1.isLegal(loc2, board1.nextPla) && 
                        board1.getLocationPriority(x2, y2) + Board::PRIOR_EPS >= board1.firstLocPriority)) {
                        continue;
                    }

                    int8_t dependValue2 = dependMap2[loc2];
                    if(dependValue2<2)
                    {
                        continue;
                    }

                    // Need to test this position by actually playing it
                    Board testBoard = board1;
                    testBoard.playMoveAssumeLegal(loc2, board1.nextPla);

                    // Calculate VCF for the test board
                    std::vector<int8_t> testDependMap;
                    int testVcfSteps = calculateShortestVCFAndDependMap(testBoard, attackPlayer, maxMove, vcfSteps2, vcfSteps2, searchFactor, testDependMap, false);

                    if (testVcfSteps != -1) {
                        longestDefense=std::max(longestDefense,testVcfSteps);
                    }
                    else {
                        canDefend=true;
                        break;
                    }
                   
                }
            }

            if(!canDefend) //all 2nd move are losing
            {
                VCFPrunedInfo info(loc1, true, longestDefense, false, 0.0f, 0.0f);
                results.push_back(info);
            }

        }
    }

    
    return results;



}
std::vector<VCFPrunedInfo> VCFCalculator::CalculateAllVCFDefendResults_stage1(
    const Board& board, 
    Color attackPlayer, 
    int maxMove, 
    double searchFactor) {
    
    Color defendPlayer = getOpp(attackPlayer);
    assert(board.stage == 1);
    assert(board.nextPla == defendPlayer);
    
    //check whether defend player can directly win
    {
        int defendMaxLenNow = VCFLogic::checkMaxConnectLen(board, defendPlayer);
        assert(defendMaxLenNow < 6);
        if (defendMaxLenNow - board.stage >= 4)//defend pla can win
        {
            return std::vector<VCFPrunedInfo>();
        }
    }


    Board boardWith1pass=board;
    boardWith1pass.playMoveAssumeLegal(Board::PASS_LOC,board.nextPla);

    // Calculate shortest VCF and dependency map for boardWith1pass
    std::vector<int8_t> dependMap;
    int vcfSteps = calculateShortestVCFAndDependMap(boardWith1pass, attackPlayer, maxMove, 0, 0, searchFactor, dependMap, false);
    
    std::vector<VCFPrunedInfo> results;
    
    if (vcfSteps <= 0) { //can't VCF even with 1 pass
        return results;
    }

    // If VCF found, analyze all legal positions for pruning
    for (int y = 0; y < board.y_size; y++) {
        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);
            
            // Check if position is legal and meets priority requirements
            if (!(board.isLegal(loc, board.nextPla) && 
                board.getLocationPriority(x, y) + Board::PRIOR_EPS >= board.firstLocPriority)) {
                continue;
            }
                
            int8_t dependValue = dependMap[loc];
            
            if (dependValue == 0 || dependValue == 1) {
                // Position is pruned - opponent will win by VCF
                VCFPrunedInfo info(loc, true, vcfSteps, false, 0.0f, 0.0f);
                results.push_back(info);
            }
            else if (dependValue == 2) {
                // Need to test this position by actually playing it
                Board testBoard = board;
                testBoard.playMoveAssumeLegal(loc, board.nextPla);
                
                // Calculate VCF for the test board
                std::vector<int8_t> testDependMap;
                int testVcfSteps = calculateShortestVCFAndDependMap(testBoard, attackPlayer, maxMove, vcfSteps, vcfSteps, searchFactor, testDependMap, false);
                
                if (testVcfSteps != -1) {
                    // VCF still possible after playing this move - position is pruned
                    VCFPrunedInfo info(loc, true, testVcfSteps, false, 0.0f, 0.0f);
                    results.push_back(info);
                } else {
                    // VCF not possible after playing this move - position is not pruned
                    //VCFPrunedInfo info(loc, false, 0, true, 0.0f, 0.0f);
                    //results.push_back(info);
                }
            }
            else 
                assert(false);
            
        }
    }

    
    return results;
}

int VCFCalculator::calculateShortestVCFAndDependMap(
    const Board& board,
    Color attackPlayer,
    int initialMaxMove,
    int minMaxMove,
    int recommendedMaxMove,
    double searchFactor,
    std::vector<int8_t>& dependMap,
    bool noOptimize) {
    
    dependMap.clear();

    assert(board.nextPla == attackPlayer);
    assert(board.stage == 0);
    
    //check whether attackPlayer has fours
    {
      std::vector<Loc> defenseFourLocs = VCFLogic::getAllDefenseFourLocs(board,attackPlayer);

      if(defenseFourLocs.size()>0){ //if attackPlayer has fours, then attackPlayer will win in 2 moves
        //mark all defenseFourLocs
        dependMap = std::vector<int8_t>(Board::MAX_ARR_SIZE, 0);
        for(Loc loc:defenseFourLocs){
          dependMap[loc] = 2;
        }
        return board.movenum + 2;
      }
    }


    // Initialize with the current board state
    Rules rules; // Use default rules, it will be overrided during the VCF calculating
    nnueHistory1.clear(board, board.nextPla, rules);


    
    int currentMaxMove = initialMaxMove;
    bool isTestingRecommendedMaxMove = false;
    if (recommendedMaxMove > 0 && recommendedMaxMove >= minMaxMove && recommendedMaxMove < initialMaxMove) //try recommendedMaxMove first
    {
        isTestingRecommendedMaxMove = true;
        currentMaxMove = recommendedMaxMove;
    }

    int lastSuccessfulVCFSteps = -1;
    std::vector<int8_t> lastSuccessfulDependMap;
    
    while (true) {
        if (currentMaxMove < board.movenum + 6 || currentMaxMove < minMaxMove)
            break;
        // Modify the rules to set new maxMoves
        Rules rules=Rules();
        rules.maxMoves = currentMaxMove;
        if(attackPlayer==C_BLACK)
          rules.VCNRule=Rules::VCNRULE_VC4_B;
        else
          rules.VCNRule=Rules::VCNRULE_VC4_W;

        nnueHistory1.rules = rules;
        
        // Create MCTS search instance
        MCTSsearch mcts(cacheTable, &nnueHistory1, attackPlayer);
        
        // Perform VCF search
        Loc bestMove;
        bool canWin = mcts.vcfSearchAutoStop(&nnueHistory1, attackPlayer, searchFactor);
        //std::cout << currentMaxMove <<" "<<canWin<< std::endl;
        
        if (canWin) {
            isTestingRecommendedMaxMove = false;
            // Calculate dependency map for successful VCF
            int winMoveNum = mcts.rootNode->stepsToWin; 
            assert(winMoveNum >= board.movenum + 6 && (currentMaxMove==0||winMoveNum<=currentMaxMove));
            assert((winMoveNum-board.movenum)%4 == 2);
            std::vector<int8_t> currentDependMap = mcts.calculateDefenseDependencyMap();
            
            // Store the successful result
            lastSuccessfulVCFSteps = winMoveNum;
            lastSuccessfulDependMap = currentDependMap;
            
            // Try with fewer moves (reduce by 4 as specified)
            currentMaxMove = winMoveNum - 4;
            if(noOptimize)
              break;
        } else {
            if (isTestingRecommendedMaxMove)
            {
                //recommendedMaxMove failed, at least recommendedMaxMove+4
                isTestingRecommendedMaxMove = false;
                currentMaxMove = initialMaxMove;
                minMaxMove = recommendedMaxMove + 4;
            }
            else
                break;// VCF failed, stop the search
        }
        
    }
    
    // Return the dependency map from the last successful VCF
    if (lastSuccessfulVCFSteps != -1) {
        dependMap = lastSuccessfulDependMap;
        return lastSuccessfulVCFSteps;
    } else {
        // No successful VCF found
        dependMap.clear();
        return -1;
    }
}