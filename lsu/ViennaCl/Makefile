
ViennaCl     	?= /opt/ViennaCL-1.7.1
Boost		?= /home/mrnavarro/boost_1_60_0


CC = g++


CFLAGS   := -I$(ViennaCl) -I$(Boost)



all: c

c: vienna

clean:
	-rm -f vienna *.o

.SUFFIXES:


# ----------------------------------------
# C example
%.o: %.cpp
	$(CC) -O3 $(CFLAGS) -c -o $@ $< -lOpenCL -DVIENNACL_WITH_OPENCL

vienna: SCLSU_ViennaCL.o
	 $(CC) -O3 -o $@ $^ $(CFLAGS) -lOpenCL




