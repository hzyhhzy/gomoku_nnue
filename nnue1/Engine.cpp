
#include "Engine.h"
#include <chrono>
#include "TT.h"
using namespace std;

typedef int64_t Time;  // value in milliseconds
Time now()
{
  static_assert(sizeof(Time) == sizeof(std::chrono::milliseconds::rep),
                "Time should be 64 bits");

  auto dur = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
}


Engine::Engine(std::string evaluator_type, std::string weightfile, int TTsize) {
  evaluator = new Evaluator(evaluator_type, weightfile);
  search    = new PVSsearch(evaluator);
  TT.resize(TTsize);
  nextColor = C_BLACK;
}

void Engine::protocolLoop() { 
  while (true) {
    string command;
    cin >> command;
    if (command == "START") {
      int size;
      cin >> size;
      if (size == BS)
        cout << "OK";
      else
        cout << "ERROR This engine only support boardsize " << BS;
      evaluator->clear();
    }
    else if (command == "TURN") {
      Time tic = now();
      for (int depth = 0; depth < 100; depth++) {
        Loc    loc;
        //TODO : time limit
        double value = search->fullsearch(C_BLACK, depth, loc);
        Time   toc   = now();
        // search->evaluator->recalculate();
        cout << "INFO Depth = " << depth << " Value = " << valueText(value)
             << " Nodes = " << search->nodes << "(" << search->interiorNodes << ")"
             << " Time = " << toc - tic
             << " Nps = " << search->nodes * 1000.0 / (toc - tic)
             << " TT = " << 100.0 * search->ttHits / search->interiorNodes << "("
             << 100.0 * search->ttCuts / search->ttHits << ")"
             << " PV = " << search->rootPV() << endl;

    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else if (command == "") {
    }
    else
      cout << "ERROR unknown command";



    cout << endl;
  }
}
