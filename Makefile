ifndef RGLIB
RGLIB := $(PWD)/build/lib
endif

ifndef RGINCLUDE
RGINCLUDE := $(PWD)/build/include
endif

RGTARGETS := $(RGLIB)/libERIRegent.so $(RGINCLUDE)/eri_regent_tasks.h

REGENT := regent
RGFLAGS := -fflow 0

RGSRCS := src/jfock.rg src/fields.rg src/helper.rg src/generate_lib.rg
RGSRCS += src/mcmurchie/jfock/generate_jfock_integral.rg
RGSRCS += src/mcmurchie/jfock/generate_kernel_statements.rg
RGSRCS += src/mcmurchie/jfock/generate_R_table.rg

.PHONY: rg.clean

all: $(RGTARGETS)

$(RGTARGETS): $(RGSRCS)
ifndef RG_MAX_MOMENTUM
	$(error Please set RG_MAX_MOMENTUM to one of `[S|P|D|F|G]`)
endif
	@mkdir -p $(RGLIB) $(RGINCLUDE)
	$(REGENT) src/generate_lib.rg --lib $(RGLIB)/libERIRegent.so \
	                              --header $(RGINCLUDE)/eri_regent_tasks.h \
															  -L $(RG_MAX_MOMENTUM) $(RGFLAGS)

rg.clean:
	$(RM) $(RGTARGETS)
