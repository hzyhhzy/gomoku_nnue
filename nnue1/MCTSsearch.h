#pragma once
#include "Evaluator.h"
#include "VCF/VCFsolver.h"
#include "ExtraStates.h"
const double policyQuant = 50000;
const double policyQuantInv = 1/policyQuant;
 
struct MCTSnode;
class MCTSsearch;

struct MCTSchild
{
  MCTSnode* ptr;
  NU_Loc loc;
  uint16_t policy;//����policyQuant��ԭʼpolicy
};

struct MCTSnode
{
  NNUE::MCTSsureResult sureResult;
  int16_t childrennum;
  int16_t legalChildrennum;//=min(�Ϸ��з���MAX_MCTS_CHILDREN)
  MCTSchild children[NNUE::MAX_MCTS_CHILDREN];
  

  uint64_t visits;//�����Լ�
  NNUE::ValueSum WRtotal;//����һ�����ӵ���һ�����ӽ�,1ʤ-1��
  //ƽ��ʤ��=WRtotal/visits

  Color nextColor;
  
  MCTSnode(MCTSsearch* search,  Color nextColor,double policyTemp);
  MCTSnode(NNUE::MCTSsureResult sureResult, Color nextColor);
  ~MCTSnode();
  
};

class MCTSsearch 
{
public:


  static NNUEHashTable hashTable;




  MCTSnode   *rootNode;
  Color       board[MaxBS * MaxBS];
  Hash128 posHash;// only board
  ExtraStates states;

  Evaluator *evaluator;  //��engine���������evaluator��������������
  VCFsolver vcfSolver[2];


  std::atomic_bool terminate;

  struct Option
  {
    size_t maxNodes=0;
  } option;
  struct Param
  {
    double expandFactor = 0.2;//��ͳ��mcts��Ӧ0.0, Ҳ����ÿ��playout�̶�����1��Ҷ�ӽڵ�
    double puct = 2;
    double puctPow = 0.75;//��ͳ��mcts��Ӧ0.5
    double puctBase = 10;
    double fpuReduction = 0.1;
    double policyTemp = 1.1;

  }params;



  MCTSsearch(Evaluator *e);
  float    fullsearch(Color color, double factor, NU_Loc &bestmove);
  void play(Color color, NU_Loc loc);
  void undo( NU_Loc loc);
  void clearBoard(); 

  NU_Loc bestRootMove() const;
  float getRootValue() const;
  int64_t getRootVisit() const;
  void stop() { terminate.store(true, std::memory_order_relaxed); }

  void             setOptions(size_t maxNodes) { option.maxNodes = maxNodes; }
  void loadParamFile(std::string filename);
  ~MCTSsearch() { if(rootNode!=NULL)delete rootNode; }

private:

  NU_Loc        locbuf[MaxBS * MaxBS];
  NNUE::PolicyType pbuf1[MaxBS * MaxBS], pbuf2[NNUE::MAX_MCTS_CHILDREN];
  float      pbuf3[NNUE::MAX_MCTS_CHILDREN];
  float      gfbuf[NNUEV2::globalFeatureNum];

  friend MCTSnode::MCTSnode(MCTSsearch *search,
                            Color       nextColor,
                            double      policyTemp);
  struct SearchResult
  {
    uint64_t newVisits;
    NNUE::ValueSum WRchange;
  };
  void playForSearch(Color color, NU_Loc loc);
  void undoForSearch(NU_Loc loc);

  SearchResult search(MCTSnode* node, uint64_t remainVisits,bool isRoot);
  NNUE::MCTSsureResult checkSureResult(NU_Loc nextMove, Color color);//check VCF
  int            selectChildIDToSearch(MCTSnode *node);
  void      getGlobalFeatureInput(Color nextPlayer);
  Hash128            getStateHash(Color nextPlayer);



};

