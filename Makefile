
-include ../../../make.vars

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

ifndef RGFLAGS
RGFLAGS :=
endif

RGFLAGS += -fflow 0

# to view the metaprogrammed result use -fpretty 1
#RGFLAGS += -fpretty 1

RGTARGETS := $(RGLIB)/libERIRegent.so $(RGINCLUDE)/eri_regent_tasks.h

RGSRCS := $(RGSRC)/jfock.rg $(RGSRC)/kfock.rg
RGSRCS += $(RGSRC)/fields.rg $(RGSRC)/helper.rg $(RGSRC)/generate_lib.rg
RGSRCS += $(RGSRC)/mcmurchie/jfock/generate_jfock_integral.rg
RGSRCS += $(RGSRC)/mcmurchie/jfock/generate_kernel_statements.rg
RGSRCS += $(RGSRC)/mcmurchie/kfock/generate_kfock_integral.rg
RGSRCS += $(RGSRC)/mcmurchie/kfock/generate_kernel.rg
RGSRCS += $(RGSRC)/mcmurchie/generate_R_table.rg

.PHONY: rg.clean

INCLUDE_PATH += ";$(CUDAINC)"

all: $(RGTARGETS)

$(RGLIB)/libERIRegent.so: $(RGSRCS)
ifndef RG_MAX_MOMENTUM
	$(error Please set RG_MAX_MOMENTUM to one of `[S|P|D|F|G]`)
endif
	@mkdir -p $(RGLIB) $(RGINCLUDE)
	INCLUDE_PATH=$(INCLUDE_PATH) \
	$(REGENT) $(RGSRC)/generate_lib.rg --lib $(RGLIB)/libERIRegent.so --header $(RGINCLUDE)/eri_regent_tasks.h -L $(RG_MAX_MOMENTUM) $(RGFLAGS) 

$(RGINCLUDE)/eri_regent_tasks.h: $(RGLIB)/libERIRegent.so

rg.clean:
	@rm -f $(RGTARGETS)
