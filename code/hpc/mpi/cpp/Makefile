# Makefile for mpi_basic.cpp

# Specify the shell to use
SHELL = /bin/bash

# Compiler
CXX = mpic++

# Compiler flags
CXXFLAGS = -O2 -Wall

# Target executable
TARGET = mpi_basic.bin

# Source files
SRCS = mpi_basics.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean