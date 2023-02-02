REGENT ?= regent
REGENT_FLAGS ?= -fflow 0 -fseparate 1
TOP_KFOCK_FLAGS ?= -L S

.PHONY: all
all: main libtopkfock.so mcmurchie/kfock/libkfock00000.so

main: main.t libtopkfock.so topkfock.h mcmurchie/kfock/libkfock00000.so mcmurchie/kfock/kfock00000.h
	INCLUDE_PATH=".;./mcmurchie/kfock" $(REGENT) $<

libtopkfock.so topkfock.h: top_kfock.rg
	SAVELIB=1 $(REGENT) $< $(REGENT_FLAGS) $(TOP_KFOCK_FLAGS)

mcmurchie/kfock/libkfock00000.so mcmurchie/kfock/kfock00000.h: mcmurchie/kfock/generate_kfock_integral_def.rg
	$(REGENT) $< $(REGENT_FLAGS) 0 0 0 0 0

.PHONY: clean
clean:
	rm -f main libtopkfock.so topkfock.h mcmurchie/kfock/libkfock00000.so mcmurchie/kfock/kfock00000.h
