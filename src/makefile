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

all: cublas MM_v1 MM_v2 MM_v3 MM_v4 MM_v5 clean

%.o: %.c
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

# cublas: $(CUOBJS) 
# 	$(NVCC) -o ../exec/cublas cublas.cu -lcublas

# MM_v1: $(CUOBJS) 
# 	$(NVCC) -o ../exec/MM_v1 MM_v1.cu -lcublas

# MM_v2: $(CUOBJS)
# 	$(NVCC) -o ../exec/MM_v2 MM_v2.cu -lcublas

# MM_v3: $(CUOBJS) 
# 	$(NVCC) -o ../exec/MM_v3 MM_v3.cu -lcublas

# MM_v4: $(CUOBJS) 
# 	$(NVCC) -o ../exec/MM_v4 MM_v4.cu -lcublas

# MM_v5: $(CUOBJS) $(COBJS)
# 	$(NVCC) -o ../exec/MM_v5 MM_v5.cu $(COBJS) -lcublas

cublas: 
	$(NVCC) -o ../exec/cublas cublas.cu -lcublas

MM_v1: 
	$(NVCC) -o ../exec/MM_v1 MM_v1.cu -lcublas

MM_v2: 
	$(NVCC) -o ../exec/MM_v2 MM_v2.cu -lcublas

MM_v3:
	$(NVCC) -o ../exec/MM_v3 MM_v3.cu -lcublas

MM_v4: 
	$(NVCC) -o ../exec/MM_v4 MM_v4.cu -lcublas

MM_v5:
	$(NVCC) -o ../exec/MM_v5 MM_v5.cu util.c -lcublas

clean:
	find . -name "*.o" -exec rm -f '{}' ';'

