ifndef LEGION
	$(error Please set LEGION to source path)
endif

REGENT := regent
REGENT_FLAGS := -fflow 0

CC := c++
CFLAGS := -O2 -Wall -Werror
INCLUDES := -I$(LEGION)/runtime
LFLAGS := -L$(LEGION)/bindings/regent -L .
LIBS := -lregent -ljfock_tasks

SRCS_RG := ../jfock.rg ../fields.rg ../helper.rg ../generate_lib.rg
SRCS_RG += ../mcmurchie/jfock/generate_jfock_integral.rg
SRCS_RG += ../mcmurchie/jfock/generate_kernel_statements.rg
SRCS_RG += ../mcmurchie/jfock/generate_R_table.rg
# SRCS_RG += ../mcmurchie/gamma_table.h ../mcmurchie/populate_gamma_table.rg

SRCS_CPP := top.cpp
TARGET := jfock_cpp

.PHONY: test clean

all: $(TARGET)

libjfock_tasks.so jfock_tasks.h: $(SRCS_RG)
ifndef MAX_MOMENTUM
	$(error Please set MAX_MOMENTUM to one of `[S|P|D|F|G]`)
endif
	$(REGENT) ../generate_lib.rg -o ./ -L $(MAX_MOMENTUM) $(REGENT_FLAGS)

$(TARGET): $(SRCS_CPP) libjfock_tasks.so jfock_tasks.h
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS_CPP) -o $@ $(LFLAGS) $(LIBS)

test: $(TARGET)
	LD_LIBRARY_PATH=$(LEGION)/bindings/regent:. ./$(TARGET)

clean:
	$(RM) $(TARGET) libjfock_tasks.so jfock_tasks.h
