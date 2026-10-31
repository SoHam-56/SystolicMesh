PRJ_DIR = $(shell pwd)
SRC_DIR = $(PRJ_DIR)/src
TB_DIR = $(PRJ_DIR)/testbenches

# FMT ?= fp32 (fp32, bf16 or int8; anything else stops make) picks the build's define, FMT_DEF = +define+FMT_<FMT>.
include $(PRJ_DIR)/ArithmeticLibrary/scripts/fmt.mk
# SIM ?= verilator, or vcs (ArithmeticLibrary's sim.mk: plain VCS builds; TRACE=1, XPROP=1, RANDINIT=1); make vcs runs TB_SystolicMesh on VCS.
include $(PRJ_DIR)/ArithmeticLibrary/scripts/sim.mk

# Toolchain
VERILATOR = verilator
VCS = vcs
WAVE = surfer

REGRESSION_SCRIPT = $(PRJ_DIR)/regression.py
MATRIX_SIZE       ?= 16
# TB_SystolicMesh's sizes, passed with MATRIX_SIZE as Verilator -G overrides; regression.py sets each test's.
TILE_SIZE         ?= 4
NUM_TEST_SETS     ?= 5
COLLAPSE_K        ?= 1
# HELD_ACCUM=1: TB_SystolicMesh's held accumulate stage too (the mesh-held points).
HELD_ACCUM        ?= 0
REGRESSION_GROUP  ?= all
FAST              ?= 0
REGRESSION_OPTS   ?=
PYTHON            ?= python3

_FAST_FLAG = $(if $(filter 1,$(FAST)),--fast)

DESIGN_FILES = \
  ../ArithmeticLibrary/src/fmt/sienna_fmt_pkg.sv \
  ../ArithmeticLibrary/src/credit/credit_link_if.sv \
  ../ArithmeticLibrary/src/credit/credit_counter.sv \
  ../ArithmeticLibrary/src/credit/credit_link_checker.sv \
  top/SystolicMesh.sv \
  top/SystolicArray.sv \
  mem/MeshOutputSram.sv \
  engine/AccumulationUnit.sv \
  engine/ProcessingElement.sv \
  ../ArithmeticLibrary/src/fmt/fmt_mul.sv \
  ../ArithmeticLibrary/src/fmt/fmt_add.sv \
  ../ArithmeticLibrary/src/multipliers/R4Booth.sv \
  ../ArithmeticLibrary/src/multipliers/karatsubaUnsigned.sv \
  ../ArithmeticLibrary/src/multipliers/fp32Multiplier.sv \
  ../ArithmeticLibrary/src/multipliers/bf16Multiplier.sv \
  ../ArithmeticLibrary/src/multipliers/bf16WideMultiplier.sv \
  ../ArithmeticLibrary/src/multipliers/int8Multiplier.sv \
  ../ArithmeticLibrary/src/converters/fp32ToBf16.sv \
  ../ArithmeticLibrary/src/adders/LZC.sv \
  ../ArithmeticLibrary/src/adders/fp32Adder.sv \
  ../ArithmeticLibrary/src/adders/bf16Adder.sv \
  ../ArithmeticLibrary/src/adders/int32Adder.sv

TOP_MODULE = TB_SystolicMesh
# TB_SystolicMesh is the top bench (testbenches/top); TB_PE, TB_PE_pack and TB_SystolicArray are unit benches (testbenches/units)
TESTBENCH = $(if $(filter TB_SystolicMesh,$(TOP_MODULE)),top,units)/$(TOP_MODULE).sv

VERILATOR_DIR = $(PRJ_DIR)/Verilator
VCS_DIR = $(PRJ_DIR)/VCS

# Waveforms cost ~100 extra translation units and dominate build time: TRACE=1 writes TB_SystolicMesh.vcd, the default 0 none; `make wave` builds with TRACE=1.
TRACE ?= 0

# ccache 3.7 here served corrupted objects that segfaulted at start-up; USE_CCACHE=1 opts back in.
USE_CCACHE ?= 0

# Very large meshes put thousands of PEs into a few generated functions that -Os takes hours on; OPT_FAST=-O0 builds in minutes.
OPT_FAST ?= -Os
# Simulation threads; several simulations at once each need fewer (Verilator's threads spin, so oversubscribed runs crawl).
SIM_THREADS ?= $(shell nproc)
ifeq ($(USE_CCACHE),0)
export CCACHE_DISABLE := 1
endif

# One verilation thread: 5.035 at -j 8 with --threads 8 hit V3TSP.cpp:353 "No unmarked edges" in about a third of int8 N8 T2 builds; at -j 1, none.
VERILATOR_FLAGS = \
	--timing \
	--assert \
	--top-module $(TOP_MODULE) \
	--threads $(SIM_THREADS) \
	-j 1 \
	--build-jobs $(shell nproc) \
	--output-split 20000 \
	--output-split-cfuncs 20000 \
	--output-groups 64 \
	-Wno-UNOPTTHREADS \
	-MAKEFLAGS OPT_FAST=$(OPT_FAST) \
	--sv \
	-I$(SRC_DIR) \
	-I$(TB_DIR) \
	--Mdir $(VERILATOR_DIR) \
	--Wno-WIDTHTRUNC \
	--Wno-WIDTHEXPAND \
	--Wno-WIDTHCONCAT

ifeq ($(TRACE),1)
VERILATOR_FLAGS += --trace -DENABLE_TRACE
endif

# LARGE_CODE=1: the large code model, for a binary past 2 GB (collapse-k 0 at N = 64: regression.py sets it there); --no-relax as ld cannot relax GOTPCREL past 2 GB.
LARGE_CODE ?= 0
ifeq ($(LARGE_CODE),1)
VERILATOR_FLAGS += -CFLAGS -mcmodel=large -CFLAGS -fno-pie -LDFLAGS -no-pie -LDFLAGS -Wl,--no-relax
endif

VERILATOR_FLAGS += $(FMT_DEF)
ifeq ($(TOP_MODULE),TB_SystolicMesh)
VERILATOR_FLAGS += -GMATRIX_SIZE=$(MATRIX_SIZE) -GTILE_SIZE=$(TILE_SIZE) -GNUM_TEST_SETS=$(NUM_TEST_SETS) -GCOLLAPSE_K=$(COLLAPSE_K)
ifneq ($(HELD_ACCUM),0)
VERILATOR_FLAGS += -GHELD_ACCUM=$(HELD_ACCUM)
endif
endif

# One-off defines, e.g. EXTRA_FLAGS=-DASSERT_SELFTEST
VERILATOR_FLAGS += $(EXTRA_FLAGS)

# VCS: the same define, sizes (-pvalue+ for Verilator's -G) and EXTRA_FLAGS' -G and -D; sim.mk keeps it plain (TRACE=1 adds the waveform).
VCS_FLAGS = +incdir+$(SRC_DIR) +incdir+$(TB_DIR) $(FMT_DEF) \
	$(call vcs_g,$(TOP_MODULE),$(filter -G%,$(VERILATOR_FLAGS))) $(patsubst -D%,+define+%,$(filter -D%,$(EXTRA_FLAGS)))

default: verilator

verilator:
	@echo "=== Verilator simulation for Systolic Array ==="
	@mkdir -p $(VERILATOR_DIR)
	$(VERILATOR) --binary \
		$(VERILATOR_FLAGS) \
		$(addprefix $(SRC_DIR)/,$(DESIGN_FILES)) \
		$(TB_DIR)/$(TESTBENCH) \
		-o $(TOP_MODULE)_sim
	@echo "-- Compiling Verilator simulation"
	make -C $(VERILATOR_DIR) -f V$(TOP_MODULE).mk
	@echo "-- Copying test files"
	@rm -f $(VERILATOR_DIR)/packShift*.mem $(VERILATOR_DIR)/matrixBias*.mem $(VERILATOR_DIR)/acc*.mem  # the copy below only adds; an earlier test's shift, bias or passes would run in this one
		@if ls $(TB_DIR)/stimulus/*.mem 1> /dev/null 2>&1; then \
		cp $(TB_DIR)/stimulus/*.mem $(VERILATOR_DIR); \
		fi
	@echo "-- Running Verilator simulation"
	cd $(VERILATOR_DIR) && $(shell command -v stdbuf >/dev/null && echo stdbuf -oL) ./$(TOP_MODULE)_sim $(SIM_ARGS)  # line-buffered, so a crash log shows how far it got; SIM_ARGS go to the simulator
	@echo "-- Verilator simulation complete"
	@$(if $(filter 1,$(TRACE)),echo "-- Trace file: $(VERILATOR_DIR)/$(TOP_MODULE).vcd",true)

vcs:
	@echo "=== VCS simulation for Systolic Array ==="
	$(call vcs_build,$(TOP_MODULE),$(VCS_DIR),$(addprefix $(SRC_DIR)/,$(DESIGN_FILES)) $(TB_DIR)/$(TESTBENCH),$(VCS_FLAGS))
	@echo "-- Copying test files"
	@rm -f $(VCS_DIR)/packShift*.mem $(VCS_DIR)/matrixBias*.mem $(VCS_DIR)/acc*.mem  # as make verilator: no earlier test's shift, bias or passes
	@if ls $(TB_DIR)/stimulus/*.mem 1> /dev/null 2>&1; then cp $(TB_DIR)/stimulus/*.mem $(VCS_DIR); fi
	@echo "-- Running VCS simulation"
	cd $(VCS_DIR) && ./sim $(VCS_RUN) $(SIM_ARGS)
	@echo "-- VCS simulation complete"

wave:
	@[ -f $(VERILATOR_DIR)/$(TOP_MODULE).vcd ] || $(MAKE) verilator TRACE=1  # no dump yet: build with waveforms and run
	@if [ -f $(VERILATOR_DIR)/$(TOP_MODULE).vcd ]; then \
		echo "-- Opening WAVE"; \
		$(WAVE) $(VERILATOR_DIR)/$(TOP_MODULE).vcd; \
	else \
		echo "-- No waveform dumps found"; \
	fi

lint:
	@echo "=== Linting Systolic Array with Verilator ==="
	@mkdir -p $(VERILATOR_DIR)
	$(VERILATOR) --lint-only \
		$(VERILATOR_FLAGS) \
		$(addprefix $(SRC_DIR)/,$(DESIGN_FILES)) \
		$(TB_DIR)/$(TESTBENCH)
	@echo "-- Lint check complete"

debug: VERILATOR_FLAGS += --debug --gdbbt
debug: verilator

perf: VERILATOR_FLAGS += --stats --profile-cfuncs
perf: verilator

# check-fmt: no RTL outside the allowlist (the package, the Fmt wrappers, the AriL units) names a precision; C4: a planted EXP_W must fail it.
FMT_ALLOW = ^ArithmeticLibrary/src/(fmt|adders|multipliers|converters|dividers|quantized)/.*$$
check-fmt:
	@bash $(PRJ_DIR)/ArithmeticLibrary/scripts/check_fmt.sh $(PRJ_DIR) '$(FMT_ALLOW)' && echo "check-fmt OK: the tree passes" || { echo "check-fmt FAIL: the tree"; exit 1; }
	@t=$$(mktemp -d); mkdir -p $$t/src/top; cp $(SRC_DIR)/top/SystolicMesh.sv $$t/src/top/; \
	sed -i 's/^endmodule/  localparam int EXP_W = 8;\nendmodule/' $$t/src/top/SystolicMesh.sv; \
	if bash $(PRJ_DIR)/ArithmeticLibrary/scripts/check_fmt.sh $$t '$(FMT_ALLOW)' > $$t/c4.log; then echo "check-fmt FAIL: C4 planted EXP_W passed"; exit 1; \
	elif ! /usr/bin/grep -q '^check-fmt: 1 RTL files outside the allowlist, 1 naming a precision$$' $$t/c4.log; then echo "check-fmt FAIL: C4 failed for another reason"; cat $$t/c4.log; exit 1; \
	else echo "check-fmt OK: C4 planted EXP_W in SystolicMesh.sv fails"; fi; rm -rf $$t

# check-hier: at fp32, bf16, int8 and no define, both mesh modes, Verilator's elaborated hierarchy holds only that precision's units (check_hier.py).
check-hier:
	@$(PYTHON) $(PRJ_DIR)/scripts/check_hier.py --verilator $(VERILATOR) --out $(PRJ_DIR)/Verilator_hier --files $(DESIGN_FILES)

clean:
	@echo "-- Cleaning simulation artifacts"
	-rm -rf $(VERILATOR_DIR) $(VCS_DIR)
	-rm -f *.vpd *.vcd *.wlf *.log
	-rm -f csrc simv simv.daidir
	-rm -f *.key DVEfiles
	@echo "-- Clean complete"

# One N: the model checks unless --no-checks, then every tile size (or REGRESSION_OPTS="--tiles 4 --format bf16 ...").
mesh-regression:
	@echo "=== SystolicMesh Regression Suite ==="
	@echo "-- Matrix size : $(MATRIX_SIZE)"
	@echo "-- Group       : $(REGRESSION_GROUP)"
	@echo "-- Fast        : $(if $(filter 1,$(FAST)),yes,no)"
	$(PYTHON) $(REGRESSION_SCRIPT) \
		--matrix-size $(MATRIX_SIZE) \
		--group $(REGRESSION_GROUP) \
		$(_FAST_FLAG) \
		$(REGRESSION_OPTS)$(if $(filter vcs,$(SIM)), --sim vcs)

# Checks against their own references (hand corners, an integer RNE, float64), which a rounding bug shared by an RTL unit and its model fails.
aril-fpu:
	$(MAKE) -C $(PRJ_DIR)/ArithmeticLibrary common-fpu

aril-narrow:
	$(MAKE) -C $(PRJ_DIR)/ArithmeticLibrary fp32ToBf16-check fp32ToBf16-lint

model-tests:
	cd $(PRJ_DIR) && $(PYTHON) scripts/test_mesh_model_packed.py && $(PYTHON) scripts/test_stim_format.py && $(PYTHON) scripts/test_regression_parse.py && $(PYTHON) scripts/test_regression_sweep.py

# make regression: the model checks once, the mesh at N 8-64 (SMALL=1: 8, 16) x every tile size x format; one target per point, then one table.
FORMATS = fp32 bf16 int8
# Only SMALL= and FMT= on make's command line narrow it, not exported ones.
REG_SMALL = $(if $(filter command line,$(origin SMALL)),$(SMALL),0)
REG_FMT   = $(if $(filter command line,$(origin FMT)),$(FMT))
ifneq ($(filter-out 0 1,$(REG_SMALL))$(words $(REG_SMALL)),1)
$(error Invalid SMALL=$(SMALL): must be 0 or 1)
endif
ifneq ($(REG_FMT),)
ifneq ($(filter-out $(FORMATS),$(REG_FMT))$(words $(REG_FMT)),1)
$(error Invalid FMT=$(FMT): must be one of fp32, bf16, int8)
endif
endif
comma := ,
empty :=
space := $(empty) $(empty)
REG_FORMATS = $(subst $(space),$(comma),$(strip $(or $(REG_FMT),$(FORMATS))))
REG_SELECT  = --formats $(REG_FORMATS)$(if $(filter 1,$(REG_SMALL)), --small)
REG_DIR    ?= $(TB_DIR)/results/regression
REG_SUMMARY = $(PYTHON) $(REGRESSION_SCRIPT) --action summary --results $(REG_DIR) $(REG_SELECT) --sim $(SIM)
ifneq ($(filter regression,$(MAKECMDGOALS)),)
ifeq ($(origin REG_RUN),undefined)
REG_RUN := $(shell date +%Y%m%d-%H%M%S)-$(shell echo $$PPID)
endif
REG_POINTS := $(shell $(PYTHON) $(REGRESSION_SCRIPT) --action points $(REG_SELECT))
endif

# One point, e.g. make regression/bf16/N8_T2/mesh: runs in a private copy of the tree, writes $(REG_DIR)/<point>.log and .json.
regression/%:
	@$(PYTHON) $(REGRESSION_SCRIPT) --action point --name $* --results $(REG_DIR) --sim $(SIM)$(if $(REG_RUN), --run $(REG_RUN))$(if $(filter regression,$(MAKECMDGOALS)), --keep-going)

regression: $(addprefix regression/,$(REG_POINTS))
	@$(REG_SUMMARY) --run $(REG_RUN)

# The table of the results already in REG_DIR (SMALL and FMT pick the points it expects); fails on any FAIL or NOT RUN.
regression-summary:
	@$(REG_SUMMARY)$(if $(REG_RUN), --run $(REG_RUN))

.PHONY: default verilator vcs wave lint debug perf clean mesh-regression regression regression-summary aril-fpu aril-narrow model-tests check-fmt \
        check-hier
