# Make sure LG_RT_DIR is set to legion/runtime
ifndef LG_RT_DIR
$(error LG_RT_DIR is not set)
endif

CC := c++
CFLAGS := -O2 -Wall -Werror
INCLUDES := -I$(LG_RT_DIR)
LFLAGS := -L$(LG_RT_DIR)/../bindings/regent -L .
LIBS := -lregent -lcoulomb_tasks

SRCS_RG := ../coulomb.rg ../fields.rg ../generate_spin_pattern.rg ../generate_lib.rg
SRCS_RG += ../mcmurchie/generate_integral.rg ../mcmurchie/generate_R_table.rg ../mcmurchie/gamma_table.h ../mcmurchie/populate_gamma_table.rg
SRCS_RG += ../rys/generate_integral.rg

SRCS_CPP := top.cpp
TARGET := coulomb_cpp

.PHONY: run clean

all: $(TARGET)

libcoulomb_tasks.so coulomb_tasks.h: $(SRCS_RG)
	regent ../generate_lib.rg ./ -fflow 0

$(TARGET): $(SRCS_CPP) libcoulomb_tasks.so coulomb_tasks.h
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS_CPP) -o $@ $(LFLAGS) $(LIBS)

run: $(TARGET)
	LD_LIBRARY_PATH=$(LG_RT_DIR)/../bindings/regent:. ./$(TARGET)

clean:
	$(RM) $(TARGET) libcoulomb_tasks.so coulomb_tasks.h
