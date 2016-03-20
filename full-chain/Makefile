# Target rules
ViennaCl     	?= /opt/ViennaCL-1.7.1
Boost		?= /home/mrnavarro/boost_1_60_0



CL_CFLAGS ?= -I/opt/intel/opencl-1.2-4.5.0.8/include

CL_LDFLAGS ?= -L/opt/intel/opencl-1.2-4.5.0.8/lib64

CL_VFLAGS = -I$(ViennaCl) -I$(Boost) -lOpenCL -DVIENNACL_WITH_OPENCL


LIBRARIES := -lOpenCL -lm

OPT := -O3 
#-vec-report

CC = g++
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

lsu.o: lsu.cpp
	$(CC) $(OPT) -o $@ -c $< $(CL_VFLAGS) -w

ReadWrite.o: ReadWrite.c	
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w

main.o: main.cpp	
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w 

full-chain: main.o ReadWrite.o sga.o lsu.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) 

clean:
	rm -f full-chain *.o 
