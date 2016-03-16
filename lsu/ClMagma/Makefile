# This is a simple standalone example. See README.txt
# Initially it is setup to use OpenBLAS.
# See magma/make.inc for alternate BLAS and LAPACK libraries,
# or use pkg-config as described below.

# Paths where MAGMA, CUDA, and OpenBLAS are installed
#MAGMADIR     ?= /home/mrnavarro/magma-1.7.0
MAGMADIR     ?= /opt/clmagma-1.3.0
CUDADIR      ?= /usr/local/cuda-7.5
OPENBLASDIR  ?= /opt/clBLAS-2.4.0-Linux-x64
OPENCL_	     ?= /opt/intel/opencl-1.2-4.5.0.8

CC            = g++
LD            = g++
CFLAGS        = -Wall
LDFLAGS       = -Wall

MAGMA_CFLAGS   :=  -DADD_ -I$(MAGMADIR)/include -I$(CUDADIR)/include -I$(OPENBLASDIR)/include

#La libreria CUDADIR es necesaria aun siendo OpenCl???? ( -lcublas -lcudart )
MAGMA_LIBS   := -L$(OPENCL_)/lib64 -L$(MAGMADIR)/lib -L$(CUDADIR)/lib64 -L$(OPENBLASDIR)/lib64 \
                -llapack -lOpenCL -lclmagma -lclBLAS
		#-L/usr/lib64 -lstdc++ -lm

# Alternatively, using pkg-config (see README.txt):
# MAGMA_CFLAGS := $(shell pkg-config --cflags magma)
# MAGMA_LIBS   := $(shell pkg-config --libs   magma)


all: c

c: ClMagma

clean:
	-rm -f ClMagma *.o *.mod

.SUFFIXES:


# ----------------------------------------
# C example
%.o: %.cpp
	$(CC) -O3 $(CFLAGS) $(MAGMA_CFLAGS) -c -o $@ $<

ClMagma: ClMagma.o
	 $(LD) $(LDFLAGS) -O3 -o $@ $^ $(MAGMA_LIBS)

