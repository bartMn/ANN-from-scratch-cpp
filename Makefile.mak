# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++17 -fopenmp -pthread -I./src

# Directories
SRCDIR = src
TESTDIR = tests
OBJDIR = build
BINDIR = .

# Find all source files
SOURCES = $(wildcard $(SRCDIR)/**/*.cpp) $(wildcard $(TESTDIR)/**/*.cpp) $(SRCDIR)/main.cpp

# Convert .cpp files to object files
OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))
OBJECTS := $(patsubst $(TESTDIR)/%.cpp, $(OBJDIR)/%.o, $(OBJECTS))

# Output executable
TARGET = $(BINDIR)/my_project

# Rule to build the project
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to compile source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(TESTDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
