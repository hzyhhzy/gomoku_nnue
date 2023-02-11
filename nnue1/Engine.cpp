#include "Engine.h"


#include <chrono>
#include <future>
using namespace NNUE;
using namespace std;

Engine::Engine(std::string evaluator_type, std::string weightfile, std::string configfile,bool writeLogEnable): writeLogEnable(writeLogEnable)
{
  evaluator = new Evaluator(evaluator_type, weightfile);
  search    = new MCTSsearch(evaluator);
  if (configfile != "")search->loadParamFile(configfile);

  nextColor = C_BLACK;

  if (writeLogEnable) {
    string logfilepath = weightfile;
    while (logfilepath[logfilepath.length() - 1] != '/'
      && logfilepath[logfilepath.length() - 1] != '\\' && logfilepath.length() > 0)
      logfilepath.pop_back();
    logfilepath = logfilepath + "log.txt";

    logfile = ofstream(logfilepath, ios::app | ios::out);
  }

  timeout_turn  = 1000;
  timeout_match = 10000000;
  time_left     = 10000000;
}

void Engine::writeLog(std::string str)
{
  if (writeLogEnable)
    logfile << str << endl;
}

std::string Engine::genmove()
{
  
  int64_t   tic         = now_ms();
  double bestvalue   = 0;
  NU_Loc    bestloc     = NU_LOC_NULL;
  int    maxTurnTime = min(timeout_turn - ReservedTime, time_left / 7);
  int    maxWaitTime = max(maxTurnTime - AsyncWaitReservedTime, 0);
  int    optimalTime = maxTurnTime * 2 / 5;

    auto result = std::async(std::launch::async, [&]() {
      NU_Loc    loc;
      double value = search->fullsearch(nextColor, 0, loc);
      return std::make_pair(value, loc);
    });

    if (result.wait_for(chrono::milliseconds(
            max(maxWaitTime + tic - now_ms(), 10LL)))
        == future_status::timeout) {
      search->stop();
    }

    std::tie(bestvalue, bestloc) = result.get();
    int64_t toc                     = now_ms();
    // search->evaluator->recalculate();
    std::cout << "MESSAGE Nodes = " << search->getRootVisit()
      << " Value = " << bestvalue
      << " Time = " << toc - tic << " Nps = "
      << search->getRootVisit() * 1000.0 / (toc - tic)
      //<< " TT = " << 100.0 * search->ttHits / search->interiorNodes << "("
      //<< 100.0 * search->ttCuts / search->ttHits << ")"
      << " BestMove = " << locstr(bestloc)
      //<< " PV = " << search->rootPV()
      << endl;

  

  if (bestloc != NU_LOC_NULL)
    search->play(nextColor, bestloc);
  else
    bestloc = NU_LOC_PASS;
  nextColor = getOpp(nextColor);

  int bestx = bestloc % MaxBS, besty = bestloc / MaxBS;
  return to_string(bestx) + "," + to_string(besty);



}

void Engine::protocolLoop()
{
  string line;
  writeLog("Start protocol loop");
  while (getline(cin, line)) {
    writeLog( line );
    bool           print_endl = true;
    string         response   = "";
    string         command;
    vector<string> pieces;
    {
      // Filter down to only "normal" ascii characters. Also excludes carrage returns.
      // Newlines are already handled by getline
      size_t newLen = 0;
      for (size_t i = 0; i < line.length(); i++)
        if (((int)line[i] >= 32 && (int)line[i] <= 126) || line[i] == '\t')
          line[newLen++] = line[i];

      line.erase(line.begin() + newLen, line.end());

      // Convert tabs to spaces
      for (size_t i = 0; i < line.length(); i++)
        if (line[i] == '\t' || line[i] == ',')
          line[i] = ' ';

      line = strOp::trim(line);

      if (line.length() == 0)
        continue;

      pieces = strOp::split(line, ' ');
      for (size_t i = 0; i < pieces.size(); i++)
        pieces[i] = strOp::trim(pieces[i]);

      command = pieces[0];
      pieces.erase(pieces.begin());
    }

    if (command == "START") {
      int size;
      if (pieces.size() != 1 || !strOp::tryStringToInt(pieces[0], size))
        response = "ERROR Bad command";
      else if (size != MaxBS)
        response = "ERROR This engine only support boardsize " + to_string(MaxBS);
      else
        response = "OK";
      search->clearBoard();
      nextColor = C_BLACK;
    }
    else if (command == "TURN") {
      int oppx, oppy;
      if (pieces.size() != 2 || !strOp::tryStringToInt(pieces[0], oppx)
          || !strOp::tryStringToInt(pieces[1], oppy))
        response = "ERROR Bad command";
      else {
        NU_Loc opploc = MakeLoc(oppx, oppy);
        search->play(nextColor, opploc);
        nextColor = getOpp(nextColor);
        response  = genmove();
      }
    }
    else if (command == "BOARD") {
      search->clearBoard();
      nextColor = C_BLACK;

      string line2;
      while (1) {
        getline(cin, line2);
        writeLog(line2);
        // Convert tabs to spaces
        for (size_t i = 0; i < line2.length(); i++)
          if (line2[i] == '\t' || line2[i] == ',')
            line2[i] = ' ';

        line2 = strOp::trim(line2);

        if (line2.length() == 0)
          continue;

        vector<string> pieces2 = strOp::split(line2, ' ');
        for (size_t i = 0; i < pieces2.size(); i++)
          pieces2[i] = strOp::trim(pieces2[i]);

        int x, y;

        if (pieces2.size() == 1 && pieces2[0] == "DONE") {  // finish
          response = genmove();
          break;
        }
        else if (pieces2.size() != 3 || !strOp::tryStringToInt(pieces2[0], x)
                 || !strOp::tryStringToInt(pieces2[1], y)) {
          cout << "ERROR Bad command";
        }
        else {  // normal move
          NU_Loc loc = MakeLoc(x, y);
          search->play(nextColor, loc);
          nextColor = getOpp(nextColor);
        }
      }
    }
    else if (command == "BEGIN") {
      search->clearBoard();
      nextColor = C_BLACK;
      response  = genmove();
    }
    else if (command == "INFO") {
      print_endl        = false;
      string subcommand = "";
      if (pieces.size() != 0) {
        subcommand = pieces[0];
        pieces.erase(pieces.begin());
      }
      if (subcommand == "timeout_turn") {
        int tmp;
        if (pieces.size() != 1 || !strOp::tryStringToInt(pieces[0], tmp))
          cout << "ERROR Bad command" << endl;
        else
          timeout_turn = tmp;
      }
      else if (subcommand == "timeout_match") {
        int tmp;
        if (pieces.size() != 1 || !strOp::tryStringToInt(pieces[0], tmp))
          cout << "ERROR Bad command" << endl;
        else
          timeout_match = tmp;
      }
      else if (subcommand == "time_left") {
        int tmp;
        if (pieces.size() != 1 || !strOp::tryStringToInt(pieces[0], tmp))
          cout << "ERROR Bad command" << endl;
        else
          time_left = tmp;
      }
      else {  // do nothing
      }
    }
    else if (command == "END") {
      break;
    }
    else if (command == "ABOUT") {
      response = " name = \"nnue_test\"";
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
      response = "ERROR unknown command";

    cout << response;
    if (print_endl)
      cout << endl;
    writeLog(response);
  }

  writeLog( "Finished protocol loop" );
}
