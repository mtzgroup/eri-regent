# Make sure LG_RT_DIR is set to legion/runtime
ifndef LG_RT_DIR
$(error LG_RT_DIR is not set)
endif

CC = c++
CFLAGS = -O2 -Wall -Werror
INCLUDES = -I$(LG_RT_DIR)
LFLAGS = -L$(LG_RT_DIR)/../bindings/regent -L .
LIBS = -lregent -lcoulomb_tasks

SRCS_RG = ../mcmurchie/boys.rg ../coulomb.rg ../fields.rg ../mcmurchie/generate_integral.rg ../generate_spin_pattern.rg
SRCS_CPP = top.cpp
TARGET = coulomb_cpp

.PHONY: run clean

all: $(TARGET)

libcoulomb_tasks.so coulomb_tasks.h: $(SRCS_RG)
	regent ../generate_lib.rg ./

$(TARGET): $(SRCS_CPP) libcoulomb_tasks.so coulomb_tasks.h
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS_CPP) -o $@ $(LFLAGS) $(LIBS)

run: $(TARGET)
	LD_LIBRARY_PATH=$(LG_RT_DIR)/../bindings/regent:. ./$(TARGET)

clean:
	$(RM) $(TARGET) libcoulomb_tasks.so coulomb_tasks.h
