ifndef LG_RT_DIR
$(error Please set LG_RT_DIR to `/path/to/legion/runtime`)
endif

TARGET := jfock_cpp
LIB_TARGET_PATH := $(PWD)

REGENT := regent
REGENT_FLAGS := -fflow 0
RG_RT_DIR := $(LG_RT_DIR)/../bindings/regent

CXX := c++
CFLAGS := -O2 -Wall -Werror -std=c++11
INCLUDES := -I$(LG_RT_DIR) -I$(RG_RT_DIR)
LFLAGS := -L$(RG_RT_DIR) -L$(LIB_TARGET_PATH)
LIBS := -lregent -ljfock_tasks
LDDFLAGS := -Wl,-rpath,$(RG_RT_DIR),-rpath,$(LIB_TARGET_PATH)
SRCS := eri_regent.cpp main.cpp
HEADERS := eri_regent.h helper.h

SRCS_RG := ../jfock.rg ../fields.rg ../helper.rg ../generate_lib.rg
SRCS_RG += ../mcmurchie/jfock/generate_jfock_integral.rg
SRCS_RG += ../mcmurchie/jfock/generate_kernel_statements.rg
SRCS_RG += ../mcmurchie/jfock/generate_R_table.rg

.PHONY: lib lint clean

all: $(TARGET)

lib: libjfock_tasks.so jfock_tasks.h

libjfock_tasks.so jfock_tasks.h: $(SRCS_RG)
ifndef MAX_MOMENTUM
	$(error Please set MAX_MOMENTUM to one of `[S|P|D|F|G]`)
endif
	$(REGENT) ../generate_lib.rg -o $(LIB_TARGET_PATH) -L $(MAX_MOMENTUM) $(REGENT_FLAGS)

$(TARGET): $(SRCS) $(HEADERS) lib
	$(CXX) $(CFLAGS) $(INCLUDES) $(SRCS) -o $@ $(LFLAGS) $(LIBS) $(LDDFLAGS)

lint: $(SRCS) $(HEADERS)
	clang-format -i $(SRCS) $(HEADERS)

clean:
	$(RM) $(TARGET) libjfock_tasks.so jfock_tasks.h
