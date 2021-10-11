CPP=g++CC=gcc
CPP=g++
AR=ar
NVCC=nvcc

CSRCS := $(shell find . -name '*.c' -not -name '._*')
COBJS := $(subst .c,.o,$(CSRCS))

CUSRCS := $(shell find . -name '*.cu' -not -name '._*')
CUOBJS := $(subst .cu,.o,$(CUSRCS))

LIBDIR := -L/usr/local/cuda/lib64

CUFLAGS= \
-I. \
-Xcompiler \
-fPIC

LDFLAGS=-L. -lm -lpthread -lrt

all: cublas MM_v1 MM_v2 clean

%.o: %.c
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

cublas: $(CUOBJS) $(COBJS)
	$(NVCC) -o ../exec/cublas cublas.cu $(COBJS) -lcublas

MM_v1: $(CUOBJS) $(COBJS)
	$(NVCC) -o ../exec/MM_v1 MM_v1.cu $(COBJS) -lcublas


MM_v2: $(CUOBJS) $(COBJS)
	$(NVCC) -o ../exec/MM_v2 MM_v2.cu $(COBJS) -lcublas

clean:
	find . -name "*.o" -exec rm -f '{}' ';'

