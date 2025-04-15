# Compiler
CXX = nvcc

# Compiler flags
CXXFLAGS =  -std=c++17 -g

# Linker flags (if needed)
LDFLAGS = -lcublas -lcusparse

# Source files located in src/ directory
SRCS = $(wildcard src/*.cpp)

# Object files (replace .cpp with .o, and place in obj/ directory)
OBJS = $(SRCS:src/%.cpp=obj/%.o)

# Output executable name
TARGET = matrix_solver

# Default target
all: $(TARGET)

# Create obj/ directory if it doesn't exist
obj/%.o: src/%.cpp | obj
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for linking object files into the executable
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS)

# Create obj/ directory if it doesn't exist
obj:
	mkdir -p obj

# Clean up object files and executable
clean:
	rm -rf obj $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

