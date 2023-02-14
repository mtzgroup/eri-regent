REGENT ?= regent
REGENT_FLAGS ?= -fflow 0 -fseparate 1 -fcuda-generate-cubin 1
TOP_KFOCK_FLAGS ?= -L P
MAX_KFOCK ?= 1

TOP_KFOCK_SO := libtopkfock.so
TOP_KFOCK_H := topkfock.h
TOP_KFOCK_RG := top_kfock.rg

.PHONY: all
all: main

-include kfock.make

main: main.t $(TOP_KFOCK_SO) $(KFOCK_SO)
	MAX_KFOCK=$(MAX_KFOCK) INCLUDE_PATH=".;./mcmurchie/kfock" $(REGENT) $<

$(TOP_KFOCK_SO): $(TOP_KFOCK_RG)
	SAVELIB=1 $(REGENT) $< $(REGENT_FLAGS) $(TOP_KFOCK_FLAGS)

kfock.make: generate_kfock_make.t
	MAX_KFOCK=$(MAX_KFOCK) $(REGENT) $< > $@

.PHONY: clean
clean:
	rm -f kfock.make main $(TOP_KFOCK_SO) $(TOP_KFOCK_H) $(KFOCK_SO) $(KFOCK_H)
