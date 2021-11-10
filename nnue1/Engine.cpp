#include "Engine.h"

#include "TT.h"

#include <chrono>
#include <future>
using namespace std;

namespace strOp {

string trim(const string &s)
{
  size_t p2 = s.find_last_not_of(" \t\r\n");
  if (p2 == string::npos)
    return string();
  size_t p1 = s.find_first_not_of(" \t\r\n");
  if (p1 == string::npos)
    p1 = 0;

  return s.substr(p1, (p2 - p1) + 1);
}

vector<string> split(const string &s)
{
  istringstream  in(s);
  string         token;
  vector<string> tokens;
  while (in >> token) {
    token = trim(token);
    tokens.push_back(token);
  }
  return tokens;
}
vector<string> split(const string &s, char delim)
{
  istringstream  in(s);
  string         token;
  vector<string> tokens;
  while (getline(in, token, delim))
    tokens.push_back(token);
  return tokens;
}

bool tryStringToInt(const string &str, int &x)
{
  int           val = 0;
  istringstream in(trim(str));
  in >> val;
  if (in.fail() || in.peek() != EOF)
    return false;
  x = val;
  return true;
}
}  // namespace strOp

Engine::Engine(std::string evaluator_type, std::string weightfile, int TTsize)
{
  evaluator = new Evaluator(evaluator_type, weightfile);
  search    = new PVSsearch(evaluator);
  TT.resize(TTsize);
  nextColor = C_BLACK;

  string logfilepath = weightfile;
  while (logfilepath[logfilepath.length() - 1] != '/'
         && logfilepath[logfilepath.length() - 1] != '\\' && logfilepath.length() > 0)
    logfilepath.pop_back();
  logfilepath = logfilepath + "log.txt";

  logfile = ofstream(logfilepath, ios::app | ios::out);

  timeout_turn  = 1000;
  timeout_match = 10000000;
  time_left     = 10000000;
}

std::string Engine::genmove()
{
  Time   tic         = now();
  double bestvalue   = VALUE_NONE;
  Loc    bestloc     = NULL_LOC;
  int    maxTurnTime = min(timeout_turn - ReservedTime, time_left / 5);
  int    maxWaitTime = max(maxTurnTime - AsyncWaitReservedTime, 0);

  search->clear();
  for (int depth = 1; depth < 100; depth++) {
    auto result = std::async(std::launch::async, [&]() {
      Loc    loc;
      double value = search->fullsearch(nextColor, depth, loc);
      return std::make_pair(value, loc);
    });

    if (result.wait_for(chrono::milliseconds(max(maxWaitTime + tic - now(), 0LL)))
        == future_status::timeout) {
      search->stop();
      break;
    }

    std::tie(bestvalue, bestloc) = result.get();
    Time toc                     = now();
    // search->evaluator->recalculate();
    cout << "MESSAGE Depth = " << depth << " Value = " << valueText(bestvalue)
         << " Nodes = " << search->nodes << "(" << search->interiorNodes << ")"
         << " Time = " << toc - tic << " Nps = " << search->nodes * 1000.0 / (toc - tic)
         << " TT = " << 100.0 * search->ttHits / search->interiorNodes << "("
         << 100.0 * search->ttCuts / search->ttHits << ")"
         << " PV = " << search->rootPV() << endl;

    // TODO : time limit
    if (toc - tic > maxTurnTime / 2)
      break;
  }

  evaluator->play(nextColor, bestloc);
  nextColor = ~nextColor;

  int bestx = bestloc % BS, besty = bestloc / BS;
  return to_string(bestx) + "," + to_string(besty);
}

void Engine::protocolLoop()
{
  string line;
  logfile << "Start protocol loop" << endl;
  while (getline(cin, line)) {
    logfile << line << endl;
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
      else if (size != BS)
        response = "ERROR This engine only support boardsize " + to_string(BS);
      else
        response = "OK";
      evaluator->clear();
      nextColor = C_BLACK;
    }
    else if (command == "TURN") {
      int oppx, oppy;
      if (pieces.size() != 2 || !strOp::tryStringToInt(pieces[0], oppx)
          || !strOp::tryStringToInt(pieces[1], oppy))
        response = "ERROR Bad command";
      else {
        Loc opploc = MakeLoc(oppx, oppy);
        evaluator->play(nextColor, opploc);
        nextColor = ~nextColor;
        response  = genmove();
      }
    }
    else if (command == "BOARD") {
      evaluator->clear();
      nextColor = C_BLACK;

      string line2;
      while (1) {
        getline(cin, line2);
        logfile << line2 << endl;
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
          Loc loc = MakeLoc(x, y);
          evaluator->play(nextColor, loc);
          nextColor = ~nextColor;
        }
      }
    }
    else if (command == "BEGIN") {
      evaluator->clear();
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
      else if (subcommand == "max_memory") {
        int tmp;
        if (!strOp::tryStringToInt(pieces[0], tmp))
          cout << "ERROR Bad command" << endl;
        else
          TT.resize(1 + tmp / (1024 * 1024) / 2);
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
    logfile << response << endl;
  }

  logfile << "Finished protocol loop" << endl;
}

Time now()
{
  static_assert(sizeof(Time) == sizeof(std::chrono::milliseconds::rep),
                "Time should be 64 bits");

  auto dur = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
}