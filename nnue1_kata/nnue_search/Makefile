# Makefile for MCTS New

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I.. -I../nnue -I../game -I../external
LDFLAGS = -pthread

# Source files
SRCS = MCTSsearch.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = libmcts.a

# Test files
TEST_SRCS = ../tests/test_mcts.cpp
TEST_TARGET = test_mcts

# Dependencies
NNUE_DIR = ../nnue
GAME_DIR = ../game
EXTERNAL_DIR = ../external

# Required object files from other modules
NNUE_OBJS = $(NNUE_DIR)/Eva_nnuev2.o $(NNUE_DIR)/NNUEBoardHistory.o $(NNUE_DIR)/NNUEglobal.o
GAME_OBJS = $(GAME_DIR)/board.o $(GAME_DIR)/boardhistory.o $(GAME_DIR)/gamelogic.o $(GAME_DIR)/location.o
EXTERNAL_OBJS = $(EXTERNAL_DIR)/sha2.o

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	ar rcs $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build test program
test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRCS) $(TARGET) $(NNUE_OBJS) $(GAME_OBJS) $(EXTERNAL_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build dependencies if they don't exist
$(NNUE_OBJS):
	$(MAKE) -C $(NNUE_DIR)

$(GAME_OBJS):
	$(MAKE) -C $(GAME_DIR)

$(EXTERNAL_OBJS):
	$(MAKE) -C $(EXTERNAL_DIR)

clean:
	rm -f $(OBJS) $(TARGET) $(TEST_TARGET)

# Install target (copy to parent directory)
install: $(TARGET)
	cp $(TARGET) ..

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Build the MCTS new library"
	@echo "  test     - Build and run test program"
	@echo "  clean    - Remove built files"
	@echo "  install  - Copy library to parent directory"
	@echo "  debug    - Build with debug symbols"
	@echo "  help     - Show this help message"