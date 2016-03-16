# Target rules


CL_CFLAGS ?= -I/opt/intel/opencl-1.2-4.5.0.8/include

CL_LDFLAGS ?= -L/opt/intel/opencl-1.2-4.5.0.8/lib64

CL_OPENCV ?= -L/usr/include/opencv/build/lib


LIBRARIES := -lOpenCL -lm
LIB_CV := -lOpenCL -lm -lopencv_core -lopencv_imgproc -lopencv_highgui

OPT := -O3 
#-vec-report

CC = g++
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
	$(CC) $(OPT) -o $@ -c $< $(CL_OPENCV) $(LIB_CV) -w

hyper: main_cl.o ReadWrite.o
	$(CC) $(OPT) -o $@ $+ $(LIB_CV) 

clean:
	rm -f hyper *.o 
