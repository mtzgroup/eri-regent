PREFIX := $(PWD)/../../../build
TARGET := $(PREFIX)/jfock_test
ERI_REGENT_LIB := $(PREFIX)/lib
ERI_REGENT_HEADERS := $(PREFIX)/include/eri_regent_tasks.h
ERI_REGENT_SRC := $(PWD)/../..

CPP := c++
CPPFLAGS := -O2 -Wall -Werror -std=c++11
CPPFLAGS += -I$(PREFIX)/include -I$(ERI_REGENT_SRC)/cpp
LDFLAGS := -lregent -llegion -lrealm -L$(ERI_REGENT_LIB) -lERIRegent
LDFLAGS += -Wl,-rpath,$(ERI_REGENT_LIB)

SRCS := main.cpp $(ERI_REGENT_SRC)/cpp/eri_regent.cpp
HEADERS := $(ERI_REGENT_SRC)/cpp/eri_regent.h $(ERI_REGENT_SRC)/cpp/helper.h

.PHONY: clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS) $(ERI_REGENT_LIB)/libERIRegent.so $(ERI_REGENT_HEADERS)
	$(CXX) $(CPPFLAGS) $(SRCS) -o $@ $(LDFLAGS)

lint: $(SRCS) $(HEADERS)
	clang-format -i $(SRCS) $(HEADERS)

clean:
	@rm -f $(TARGET)
