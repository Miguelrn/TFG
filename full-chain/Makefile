# Target rules
ViennaCl     	?= /opt/ViennaCL-1.7.1


CL_CFLAGS ?= -I/opt/intel/opencl-1.2-4.5.0.8/include
CL_LDFLAGS ?= -L/opt/intel/opencl-1.2-4.5.0.8/lib64

CL_VFLAGS = -I$(ViennaCl) -lOpenCL -DVIENNACL_WITH_OPENCL

MAGMADIR     ?= /opt/clmagma-1.3.0
CUDADIR      ?= /usr/local/cuda-7.5
OPENBLASDIR  ?= /opt/clBLAS-2.4.0-Linux-x64
OPENCL_     ?= /opt/intel/opencl-1.2-4.5.0.8
#OPENCL_	     ?= /opt/intel/opencl-1.2-3.2.1.16712

MAGMA_CFLAGS   :=  -DADD_ -I$(MAGMADIR)/include -I$(CUDADIR)/include -I$(OPENBLASDIR)/include

MAGMA_LIBS   := -L$(OPENCL_)/lib64 -L$(MAGMADIR)/lib -L$(CUDADIR)/lib64 -L$(OPENBLASDIR)/lib64 \
                -llapack -lOpenCL -lclmagma -lclBLAS -lblas

LIBRARIES := -lOpenCL -lm

OPT := -O3 

CC = g++
#-ftree-vectorize -msse2 
#-fopt-info-vec
#-ftree-vectorize -msse2 -fassociative-math 

################################################################################

# Target rules

all: build
build: full-chain

init_platform.o: init_platform.c
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) 

gene.o: gene.cpp
	$(CC) $(OPT) -o $@ -c $< $(CL_VFLAGS) $(CFLAGS) $(MAGMA_CFLAGS) -lgsl

sga.o : sga.c
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w

lsu.o: lsu.cpp
	$(CC) $(OPT) -o $@ -c $< $(CL_VFLAGS) $(CFLAGS) $(MAGMA_CFLAGS)

ReadWrite.o: ReadWrite.c	
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w

main.o: main.cpp	
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w -I$(ViennaCl) $(MAGMA_CFLAGS)

full-chain: main.o ReadWrite.o sga.o lsu.o gene.o init_platform.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) $(MAGMA_LIBS) 

clean:
	rm -f full-chain *.o 
