ifndef RGLIB
RGLIB := $(PWD)/build/lib
endif

ifndef RGINCLUDE
RGINCLUDE := $(PWD)/build/include
endif

ifndef RGSRC
RGSRC := src
endif

ifndef REGENT
REGENT := regent
endif

RGTARGETS := $(RGLIB)/libERIRegent.so $(RGINCLUDE)/eri_regent_tasks.h

RGFLAGS := -fflow 0

RGSRCS := $(RGSRC)/jfock.rg $(RGSRC)/fields.rg $(RGSRC)/helper.rg $(RGSRC)/generate_lib.rg
RGSRCS += $(RGSRC)/mcmurchie/jfock/generate_jfock_integral.rg
RGSRCS += $(RGSRC)/mcmurchie/jfock/generate_kernel_statements.rg
RGSRCS += $(RGSRC)/mcmurchie/jfock/generate_R_table.rg

.PHONY: rg.clean

all: $(RGTARGETS)

$(RGTARGETS): $(RGSRCS)
ifndef RG_MAX_MOMENTUM
	$(error Please set RG_MAX_MOMENTUM to one of `[S|P|D|F|G]`)
endif
	@mkdir -p $(RGLIB) $(RGINCLUDE)
	$(REGENT) $(RGSRC)/generate_lib.rg --lib $(RGLIB)/libERIRegent.so --header $(RGINCLUDE)/eri_regent_tasks.h -L $(RG_MAX_MOMENTUM) $(RGFLAGS)

rg.clean:
	@rm -f $(RGTARGETS)