ifndef LEGIONHOME
LEGIONHOME := /usr/local
endif

ifndef LEGIONLIB
LEGIONLIB := $(LEGIONHOME)/lib
endif

PREFIX := $(PWD)/../../../build
TARGET := $(PREFIX)/eri_regent_test
ERI_REGENT_LIB := $(PREFIX)/lib
ERI_REGENT_HEADERS := $(PREFIX)/include/eri_regent_tasks.h
ERI_REGENT_SRC := $(PWD)/../..

CPP := c++
CPPFLAGS := -O2 -Wall -Werror -std=c++11
CPPFLAGS += -I$(LEGIONHOME)/include -I$(PREFIX)/include -I$(ERI_REGENT_SRC)/cpp
LDFLAGS := -L$(LEGIONLIB) -lrealm -llegion -lregent -L$(ERI_REGENT_LIB) -lERIRegent
LDFLAGS += -Wl,-rpath,$(LEGIONLIB),-rpath,$(ERI_REGENT_LIB)

SRCS := main.cpp
SRCS += $(ERI_REGENT_SRC)/cpp/eri_regent.cpp
SRCS += $(ERI_REGENT_SRC)/cpp/terachem_jdata.cpp
SRCS += $(ERI_REGENT_SRC)/cpp/terachem_kdata.cpp
HEADERS := $(ERI_REGENT_SRC)/cpp/eri_regent.h $(ERI_REGENT_SRC)/cpp/helper.h

.PHONY: clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS) $(ERI_REGENT_HEADERS)
	$(CXX) $(CPPFLAGS) $(SRCS) -o $@ $(LDFLAGS)

lint: $(SRCS) $(HEADERS)
	clang-format -i $(SRCS) $(HEADERS)

clean:
	@rm -f $(TARGET)
