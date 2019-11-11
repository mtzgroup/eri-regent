PREFIX := $(PWD)/build

TARGETS := $(PREFIX)/lib/libERIRegent.so $(PREFIX)/include/eri_regent_tasks.h

REGENT := regent
RGFLAGS := -fflow 0

RGSRCS := src/jfock.rg src/fields.rg src/helper.rg src/generate_lib.rg
RGSRCS += src/mcmurchie/jfock/generate_jfock_integral.rg
RGSRCS += src/mcmurchie/jfock/generate_kernel_statements.rg
RGSRCS += src/mcmurchie/jfock/generate_R_table.rg

# TODO: Compile api and add to lib
# TODO: Make module.mk

.PHONY: clean

all: $(TARGETS)

$(TARGETS): $(RGSRCS)
ifndef MAX_MOMENTUM
	$(error Please set MAX_MOMENTUM to one of `[S|P|D|F|G]`)
endif
	@mkdir -p $(PREFIX)/lib $(PREFIX)/include
	$(REGENT) src/generate_lib.rg --lib $(PREFIX)/lib/libERIRegent.so \
	                              --header $(PREFIX)/include/eri_regent_tasks.h \
															  -L $(MAX_MOMENTUM) $(RGFLAGS)

clean:
	$(RM) $(TARGETS)
