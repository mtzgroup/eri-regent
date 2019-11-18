ifndef LG_RT_DIR
$(error Please set LG_RT_DIR to `/path/to/legion/runtime`)
endif

PREFIX := $(PWD)/../../../build
TARGET := $(PREFIX)/jfock_test
ERI_REGENT_LIB := $(PREFIX)/lib
ERI_REGENT_HEADERS := $(PREFIX)/include/eri_regent_tasks.h
ERI_REGENT_SRC := ../..

RG_RT_DIR := $(LG_RT_DIR)/../bindings/regent

CXX := c++
CFLAGS := -O2 -Wall -Werror -std=c++11
INCLUDES := -I$(LG_RT_DIR) -I$(RG_RT_DIR) -I$(PREFIX)/include -I$(ERI_REGENT_SRC)/cpp
LFLAGS := -L$(RG_RT_DIR) -L$(ERI_REGENT_LIB)
LIBS := -lregent -lERIRegent
LDDFLAGS := -Wl,-rpath,$(RG_RT_DIR),-rpath,$(ERI_REGENT_LIB)
SRCS := main.cpp $(ERI_REGENT_SRC)/cpp/eri_regent.cpp
HEADERS := $(ERI_REGENT_SRC)/cpp/eri_regent.h $(ERI_REGENT_SRC)/cpp/helper.h

.PHONY: clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS) $(ERI_REGENT_LIB)/libERIRegent.so $(ERI_REGENT_HEADERS)
	$(CXX) $(CFLAGS) $(INCLUDES) $(SRCS) -o $@ $(LFLAGS) $(LIBS) $(LDDFLAGS)

lint: $(SRCS) $(HEADERS)
	clang-format -i $(SRCS) $(HEADERS)

clean:
	@rm -f $(TARGET)
