# Target rules


CL_CFLAGS ?= -I/opt/intel/opencl-1.2-4.5.0.8/include

CL_LDFLAGS ?= -L/opt/intel/opencl-1.2-4.5.0.8/lib64


LIBRARIES := -lOpenCL -lm

OPT := -O3 
#-vec-report

CC = gcc
#-ftree-vectorize -msse2 
#-fopt-info-vec
#-ftree-vectorize -msse2 -fassociative-math 

################################################################################

# Target rules

all: build
build: full-chain

#gene.o: gene.c

sga.o : sga.c
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w

#lsu.o: lsu.cpp

ReadWrite.o: ReadWrite.c	
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w

main.o: main.c	
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w

full-chain: main.o ReadWrite.o sga.o 
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) 

clean:
	rm -f full-chain *.o 
