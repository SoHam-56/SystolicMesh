PRJ_DIR = $(shell pwd)
SRC_DIR = $(PRJ_DIR)/src
TB_DIR = $(PRJ_DIR)/testbenches

# Toolchain
VERILATOR = verilator
VCS = vcs
VVP = vvp
WAVE = surfer

REGRESSION_SCRIPT = $(PRJ_DIR)/regression.py
MATRIX_SIZE       ?= 16
REGRESSION_GROUP  ?= all
FAST              ?= 0
REGRESSION_OPTS   ?=

_FAST_FLAG = $(if $(filter 1,$(FAST)),--fast)

DESIGN_FILES = \
  top/SystolicMesh.sv \
  top/SystolicArray.sv \
  mem/RowInputQueue.sv \
  mem/ColumnInputQueue.sv \
  mem/OutputSram.sv \
  mem/MeshOutputSram.sv \
  engine/PEMesh.sv \
  engine/ProcessingElement.sv \
  engine/AccumulationUnit.sv \
  engine/MAC.sv \
  ../ArithmeticLibrary/Multipliers/Radix4Booth/src/R4Booth.sv \
  ../ArithmeticLibrary/Multipliers/Karatsuba/src/karatsubaUnsigned.sv \
	../ArithmeticLibrary/Multipliers/FP32/src/fp32Multiplier.sv \
  ../ArithmeticLibrary/Adders/FP32/src/LZC.sv \
  ../ArithmeticLibrary/Adders/FP32/src/fp32Adder.sv

TOP_MODULE = TB_SystolicMesh
TESTBENCH = $(TOP_MODULE).sv

VERILATOR_DIR = $(PRJ_DIR)/Verilator
VCS_DIR = $(PRJ_DIR)/VCS

# Waveforms cost ~100 extra translation units and dominate build time. TRACE=0 skips
# them; `make wave` needs TRACE=1, which is the default.
TRACE ?= 1

VERILATOR_FLAGS = \
	--timing \
	--assert \
	--top-module $(TOP_MODULE) \
	--threads 8 \
	--build-jobs $(shell nproc) \
	--sv \
	-I$(SRC_DIR) \
	-I$(TB_DIR) \
	--Mdir $(VERILATOR_DIR) \
	--Wno-WIDTHTRUNC \
	--Wno-WIDTHEXPAND \
	--Wno-WIDTHCONCAT

ifeq ($(TRACE),1)
VERILATOR_FLAGS += --trace
endif

# One-off defines, e.g. EXTRA_FLAGS=-DASSERT_SELFTEST
VERILATOR_FLAGS += $(EXTRA_FLAGS)

VCS_FLAGS = \
	-full64 \
	-sverilog \
	-debug_all \
	-timescale=1ns/1ps \
	-Mdir=$(VCS_DIR) \
	+v2k \
	+incdir+$(SRC_DIR) \
	+incdir+$(TB_DIR) \
	+define+VCS

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
		@if ls $(TB_DIR)/stimulus/*.mem 1> /dev/null 2>&1; then \
		cp $(TB_DIR)/stimulus/*.mem $(VERILATOR_DIR); \
		fi
	@echo "-- Running Verilator simulation"
	cd $(VERILATOR_DIR) && ./$(TOP_MODULE)_sim
	@echo "-- Verilator simulation complete"
	@echo "-- Trace file: $(VERILATOR_DIR)/dump.vcd"

vcs:
	@echo "=== VCS simulation for Systolic Array ==="
	@mkdir -p $(VCS_DIR)
	$(VCS) $(VCS_FLAGS) \
		-o $(VCS_DIR)/$(TOP_MODULE)_sim \
		$(addprefix $(SRC_DIR)/,$(DESIGN_FILES)) \
		$(TB_DIR)/$(TESTBENCH)
	@echo "-- Running VCS simulation"
	cd $(VCS_DIR) && ./$(TOP_MODULE)_sim
	@echo "-- VCS simulation complete"

wave:
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
debug: IVERILOG_FLAGS += -g
debug: verilator

perf: VERILATOR_FLAGS += --stats --profile-cfuncs
perf: verilator

clean:
	@echo "-- Cleaning simulation artifacts"
	-rm -rf $(VERILATOR_DIR) $(IVERILOG_DIR) $(VCS_DIR)
	-rm -f *.vpd *.vcd *.wlf *.log
	-rm -f csrc simv simv.daidir
	-rm -f *.key DVEfiles
	@echo "-- Clean complete"

regression:
	@echo "=== SystolicMesh Regression Suite ==="
	@echo "-- Matrix size : $(MATRIX_SIZE)"
	@echo "-- Group       : $(REGRESSION_GROUP)"
	@echo "-- Fast        : $(if $(filter 1,$(FAST)),yes,no)"
	python3 $(REGRESSION_SCRIPT) \
		--matrix-size $(MATRIX_SIZE) \
		--group $(REGRESSION_GROUP) \
		$(_FAST_FLAG) \
		$(REGRESSION_OPTS)

.PHONY: default verilator vcs wave lint debug perf clean regression
