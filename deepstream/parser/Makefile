# Builds libnvdsparsebbox_yolo11.so for DeepStream nvinfer.
#   make                      # uses the default DeepStream / CUDA / TensorRT locations
#   make DS_ROOT=/path/to/ds  # custom DeepStream install

DS_ROOT ?= /opt/nvidia/deepstream/deepstream
CUDA_ROOT ?= /usr/local/cuda
TRT_INCLUDE ?= /usr/include/x86_64-linux-gnu

CXX ?= g++
CXXFLAGS += -O3 -std=c++17 -Wall -Wextra -fPIC
INCLUDES := -I$(DS_ROOT)/sources/includes -I$(CUDA_ROOT)/include -I$(TRT_INCLUDE)

TARGET := libnvdsparsebbox_yolo11.so
SRCS := nvdsparsebbox_yolo11.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: all clean
