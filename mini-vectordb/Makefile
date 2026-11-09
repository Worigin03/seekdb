CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
LDFLAGS ?=

SRC := $(wildcard src/*.cpp)
OBJ := $(SRC:.cpp=.o)

all: minivectordb

minivectordb: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) minivectordb

.PHONY: all clean
