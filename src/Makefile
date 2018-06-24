#
# 'make depend' uses makedepend to automatically generate dependencies
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'FeedforwardNT'
# 'make clean'  removes all .o and executable files
#

CC = clang

# Compile flags
CFLAGS = -Wall -O3

INCLUDES = -I../include

# BLAS/LAPACK (required) and OpenCL (optional) on macOS or use any other implementation
LFLAGS = -framework Accelerate -framework OpenCL -framework Metal -framework Foundation

LIBS = -lm

SRCS = main.c NeuralNetwork.c LoadIrisDataSet.c LoadMNISTDataSet.c Parsing.c Memory.c Utils.c TimeProfile.c OpenCLUtils.c MetalCompute.m

# define the C object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.c=.o)

# define the executable file
MAIN = FeedforwardNT

.PHONY: depend clean

all:    $(MAIN)
	@echo Program compiled

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
