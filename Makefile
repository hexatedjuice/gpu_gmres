# Compiler settings for GPU
CXX_GPU = nvcc
CXXFLAGS_GPU = -std=c++17 -g -DUSE_GPU
LDFLAGS_GPU = -lcublas -lcusparse

# Compiler settings for CPU with OpenMP
CXX_CPU = g++
CXXFLAGS_CPU = -std=c++17 -g -fopenmp

# Source files located in src/ directory
SRCS = $(wildcard src/*.cpp)
GPU_SRCS = $(SRCS)
CPU_SRCS = $(filter-out src/gpu_gmres.cpp, $(SRCS))

# Object files (replace .cpp with .o, and place in obj/ directory)
OBJS_GPU = $(GPU_SRCS:src/%.cpp=obj/gpu_%.o)
OBJS_CPU = $(CPU_SRCS:src/%.cpp=obj/cpu_%.o)

# Output executable names
TARGET_GPU = matrix_solver_gpu
TARGET_CPU = matrix_solver_cpu

# Default target (build both versions)
all: $(TARGET_GPU) $(TARGET_CPU)

# GPU object compilation rule
obj/gpu_%.o: src/%.cpp | obj
	$(CXX_GPU) $(CXXFLAGS_GPU) -c $< -o $@

# CPU object compilation rule
obj/cpu_%.o: src/%.cpp | obj
	$(CXX_CPU) $(CXXFLAGS_CPU) -c $< -o $@

# GPU version linking rule
$(TARGET_GPU): $(OBJS_GPU)
	$(CXX_GPU) $(LDFLAGS_GPU) -o $(TARGET_GPU) $(OBJS_GPU)

# CPU version linking rule (with OpenMP)
$(TARGET_CPU): $(OBJS_CPU)
	$(CXX_CPU) $(CXXFLAGS_CPU) -o $(TARGET_CPU) $(OBJS_CPU)

# Create obj/ directory if it doesn't exist
obj:
	mkdir -p obj

# Clean up object files and executables
clean:
	rm -rf obj $(TARGET_GPU) $(TARGET_CPU)

# Run the GPU version of the program
run_gpu: $(TARGET_GPU)
	./$(TARGET_GPU)

# Run the CPU version of the program (with OpenMP)
run_cpu: $(TARGET_CPU)
	./$(TARGET_CPU)
