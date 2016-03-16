# Target rules


CL_CFLAGS ?= -I/opt/intel/opencl-1.2-4.5.0.8/include

CL_LDFLAGS ?= -L/opt/intel/opencl-1.2-4.5.0.8/lib64


LIBRARIES := -lOpenCL -lm

OPT := -O3 -restrict -axSSE4.1 -xSSE4.2 -opt-report
#-fno-tree-vectorize -ffast-math
#-vec-report

CC = icc
#-ftree-vectorize -msse2 
#-fopt-info-vec
#-ftree-vectorize -msse2 -fassociative-math 


################################################################################

# Target rules

all: build
build: hyper

ReadWrite.o: ReadWrite.c	
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w

main_cl.o: main_cl.c	
	$(CC) $(OPT) -o $@ -c $< $(LIBRARIES) -w

hyper: main_cl.o ReadWrite.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) 

clean:
	rm -f hyper *.o 
