# Makefile for Metal DEM rod simulation
# Builds rods_kernels.metallib and metal_sim executable

CXX       = clang++
CXXFLAGS  = -std=c++17 -O3 -ffast-math -march=native -flto -Wall

# Paths — adjust if Eigen/nlohmann are installed elsewhere
EIGEN_INC ?= /opt/homebrew/include/eigen3
JSON_INC  ?= /opt/homebrew/include

INCLUDES  = -I$(EIGEN_INC) -I$(JSON_INC)

FRAMEWORKS = -framework Metal -framework Foundation -framework CoreGraphics

TARGET    = metal_sim
MM_SRC    = main_metal.mm
METAL_SRC = rods_kernels.metal
AIR_OUT   = rods_kernels.air
METALLIB  = rods_kernels.metallib

.PHONY: all clean

all: $(METALLIB) $(TARGET)

# Compile Metal shader to AIR
$(AIR_OUT): $(METAL_SRC)
	xcrun -sdk macosx metal -O2 -ffast-math -c $(METAL_SRC) -o $(AIR_OUT)

# Link AIR to metallib
$(METALLIB): $(AIR_OUT)
	xcrun -sdk macosx metallib $(AIR_OUT) -o $(METALLIB)

# Compile Objective-C++ host code
$(TARGET): $(MM_SRC) $(METALLIB)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(MM_SRC) -o $(TARGET) $(FRAMEWORKS)

clean:
	rm -f $(AIR_OUT) $(METALLIB) $(TARGET)
