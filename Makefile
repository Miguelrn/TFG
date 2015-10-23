CC = gcc

EXE   = hyper 

SOURCES    = main.c funciones.c


OBJS    = $(SOURCES:.c=.o)

LIBS = -lm 

SOURCEDIR = .

$(EXE) :$(OBJS) 
	$(CC) $(CFLAGS)  -o $@ $? $(LIBS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<


clean:
	rm -f $(OBJS) $(EXE)
