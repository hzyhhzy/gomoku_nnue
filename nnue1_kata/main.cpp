#include "main.h"

#include "core/os.h"
#include "core/mainargs.h"


#include <sstream>

//------------------------
#include "core/using.h"
//------------------------

static void printHelp(const vector<string>& args) {
  cout << endl;
  if(args.size() >= 1)
    cout << "Usage: " << args[0] << " SUBCOMMAND ";
  else
    cout << "Usage: " << "./katago" << " SUBCOMMAND ";
  cout << endl;

  cout << R"%%(
---Common subcommands------------------

gtp : Runs GTP engine that can be plugged into any standard Go GUI for play/analysis.
benchmark : Test speed with different numbers of search threads.
genconfig : User-friendly interface to generate a config with rules and automatic performance tuning.

contribute : Connect to online distributed KataGo training and run perpetually contributing selfplay games.

match : Run self-play match games based on a config, more efficient than gtp due to batching.
version : Print version and exit.

analysis : Runs an engine designed to analyze entire games in parallel.
tuner : (OpenCL only) Run tuning to find and optimize parameters that work on your GPU.

---Selfplay training subcommands---------

selfplay : Play selfplay games and generate training data.
gatekeeper : Poll directory for new nets and match them against the latest net so far.

---Testing/debugging subcommands-------------
evalsgf : Utility/debug tool, analyze a single position of a game from an SGF file.

testgpuerror : Print the average error of the neural net between current config and fp32 config.


)%%" << endl;
}

static int handleSubcommand(const string& subcommand, const vector<string>& args) {
  vector<string> subArgs(args.begin()+1,args.end());
  if(subcommand == "testnnue")
    return MainCmds::testnnue();
  else if(subcommand == "version") {
    cout << Version::getKataGoVersionFullInfo() << std::flush;
    return 0;
  }
  else {
    cout << "Unknown subcommand: " << subcommand << endl;
    printHelp(args);
    return 1;
  }
  return 0;
}


int main(int argc, const char* const* argv) {
  vector<string> args = MainArgs::getCommandLineArgsUTF8(argc,argv);
  MainArgs::makeCoutAndCerrAcceptUTF8();

  if(args.size() < 2) {
    args.push_back("testnnue");
    //printHelp(args);
    //return 0;
  }
  string cmdArg = string(args[1]);
  if(cmdArg == "-h" || cmdArg == "--h" || cmdArg == "-help" || cmdArg == "--help" || cmdArg == "help") {
    printHelp(args);
    return 0;
  }

#if defined(OS_IS_WINDOWS)
  //On windows, uncaught exceptions reaching toplevel don't normally get printed out,
  //so explicitly catch everything and print
  int result;
  try {
    result = handleSubcommand(cmdArg, args);
  }
  catch(std::exception& e) {
    cerr << "Uncaught exception: " << e.what() << endl;
    return 1;
  }
  catch(...) {
    cerr << "Uncaught exception that is not a std::exception... exiting due to unknown error" << endl;
    return 1;
  }
  return result;
#else
  return handleSubcommand(cmdArg, args);
#endif
}


string Version::getKataGoVersion() {
  return string("1.12.4");
}

string Version::getKataGoVersionForHelp() {
  return string("KataGo v1.12.4");
}

string Version::getKataGoVersionFullInfo() {
  ostringstream out;
  out << Version::getKataGoVersionForHelp() << endl;
  out << "Compile Time: " << __DATE__ << " " << __TIME__ << endl;
#if defined(USE_CUDA_BACKEND)
  out << "Using CUDA backend" << endl;
#if defined(CUDA_TARGET_VERSION)
#define STRINGIFY(x) #x
#define STRINGIFY2(x) STRINGIFY(x)
  out << "Compiled with CUDA version " << STRINGIFY2(CUDA_TARGET_VERSION) << endl;
#endif
#elif defined(USE_TENSORRT_BACKEND)
  out << "Using TensorRT backend" << endl;
#elif defined(USE_OPENCL_BACKEND)
  out << "Using OpenCL backend" << endl;
#elif defined(USE_EIGEN_BACKEND)
  out << "Using Eigen(CPU) backend" << endl;
#else
  out << "Using dummy backend" << endl;
#endif

#if defined(USE_AVX2)
  out << "Compiled with AVX2 and FMA instructions" << endl;
#endif
#if defined(COMPILE_MAX_BOARD_LEN)
  out << "Compiled to allow boards of size up to " << COMPILE_MAX_BOARD_LEN << endl;
#endif
#if defined(BUILD_DISTRIBUTED)
  out << "Compiled to support contributing to online distributed selfplay" << endl;
#endif

  return out.str();
}
