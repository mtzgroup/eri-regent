TARGETS := libERIRegent.so eri_regent_tasks.h

REGENT := regent
RGFLAGS := -fflow 0

RGSRCS := jfock.rg fields.rg helper.rg generate_lib.rg
RGSRCS += mcmurchie/jfock/generate_jfock_integral.rg
RGSRCS += mcmurchie/jfock/generate_kernel_statements.rg
RGSRCS += mcmurchie/jfock/generate_R_table.rg

.PHONY: clean

all: $(TARGETS)

$(TARGETS): $(RGSRCS)
ifndef MAX_MOMENTUM
	$(error Please set MAX_MOMENTUM to one of `[S|P|D|F|G]`)
endif
	$(REGENT) generate_lib.rg -o $(PWD) -L $(MAX_MOMENTUM) $(RGFLAGS)

clean:
	$(RM) $(TARGETS)
