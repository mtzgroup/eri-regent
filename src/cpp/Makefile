ifndef LG_RT_DIR
$(error Please set LG_RT_DIR to source path)
endif

REGENT := regent
REGENT_FLAGS := -fflow 0

CC := c++
CFLAGS := -O2 -Wall -Werror
INCLUDES := -I$(LG_RT_DIR) -I$(LG_RT_DIR)/../bindings/regent
LFLAGS := -L$(LG_RT_DIR)/../bindings/regent -L .
LIBS := -lregent -ljfock_tasks
SRCS := eri_regent.cpp main.cpp
HEADERS := eri_regent.h helper.h

SRCS_RG := ../jfock.rg ../fields.rg ../helper.rg ../generate_lib.rg
SRCS_RG += ../mcmurchie/jfock/generate_jfock_integral.rg
SRCS_RG += ../mcmurchie/jfock/generate_kernel_statements.rg
SRCS_RG += ../mcmurchie/jfock/generate_R_table.rg
# SRCS_RG += ../mcmurchie/gamma_table.h ../mcmurchie/populate_gamma_table.rg

TARGET := jfock_cpp

.PHONY: test lint clean

all: $(TARGET)

libjfock_tasks.so jfock_tasks.h: $(SRCS_RG)
ifndef MAX_MOMENTUM
	$(error Please set MAX_MOMENTUM to one of `[S|P|D|F|G]`)
endif
	$(REGENT) ../generate_lib.rg -o ./ -L $(MAX_MOMENTUM) $(REGENT_FLAGS)

$(TARGET): $(SRCS) $(HEADERS) libjfock_tasks.so jfock_tasks.h
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS) -o $@ $(LFLAGS) $(LIBS)

test: $(TARGET)
	LD_LIBRARY_PATH=$(LG_RT_DIR)/../bindings/regent:. ./$(TARGET)

lint: $(SRCS) $(HEADERS)
	clang-format -i $(SRCS) $(HEADERS)

clean:
	$(RM) $(TARGET) libjfock_tasks.so jfock_tasks.h
