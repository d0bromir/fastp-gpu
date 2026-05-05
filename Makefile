DIR_INC := ./inc
DIR_SRC := ./src
DIR_OBJ := ./obj

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
INCLUDE_DIRS ?=
LIBRARY_DIRS ?=

NVCC := $(shell ls /usr/local/cuda-13.1/bin/nvcc 2>/dev/null || command -v nvcc 2>/dev/null)
# Derive CUDA root from nvcc location; fall back to /usr/local/cuda
CUDA_ROOT := $(or $(patsubst %/bin/nvcc,%,$(NVCC)),/usr/local/cuda)

LIBS := -lisal -ldeflate -lpthread


SRC := $(wildcard ${DIR_SRC}/*.cpp)

# Default: do not attempt CUDA build unless explicitly requested.
# To enable CUDA build set WITH_CUDA=1 when invoking make.
ifeq ($(WITH_CUDA),1)
	ifneq ($(NVCC),)
		CUDA_SRC := $(wildcard $(DIR_SRC)/*.cu)
		CUDA_OBJ := $(patsubst $(DIR_SRC)/%.cu,$(DIR_OBJ)/%.o,$(CUDA_SRC))
		LIBS += -L$(CUDA_ROOT)/lib64 -lcudart -lcuda
		# Filter out stub when building with CUDA
		SRC := $(filter-out $(DIR_SRC)/cuda_stats_stub.cpp,$(SRC))
		CUDA_ENABLED := 1
	else
$(warning WITH_CUDA=1 but nvcc not found; falling back to CPU stub)
		# Filter out wrapper and use stub instead
		SRC := $(filter-out $(DIR_SRC)/cuda_stats_wrapper.cpp,$(SRC))
		SRC += $(DIR_SRC)/cuda_stats_stub.cpp
		CUDA_OBJ :=
		CUDA_ENABLED := 0
	endif
else
	# No CUDA requested: filter out wrapper and compile CPU stub
	SRC := $(filter-out $(DIR_SRC)/cuda_stats_wrapper.cpp,$(SRC))
	SRC += $(DIR_SRC)/cuda_stats_stub.cpp
	CUDA_OBJ :=
	CUDA_ENABLED := 0
endif

# Deduplicate source list (avoid adding stub twice)
SRC := $(sort $(SRC))

OBJ := $(patsubst %.cpp,${DIR_OBJ}/%.o,$(notdir ${SRC}))

# Filter out test_cuda_stats from main build
OBJ := $(filter-out obj/test_cuda_stats.o,$(OBJ))

# Only include cuda_stats.o, not test_cuda_stats.o
ifeq ($(WITH_CUDA),1)
	ifneq ($(NVCC),)
		CUDA_OBJ := $(filter-out %test_cuda_stats.o,$(CUDA_OBJ))
	endif
endif

OBJ += $(CUDA_OBJ)

TARGET := fastp

BIN_TARGET := ${TARGET}

CXX ?= g++-13
CXXFLAGS := -std=c++11 -pthread -g -O3 -march=native -flto -funroll-loops -MD -MP -I${DIR_INC} $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) ${CXXFLAGS}
# Optional: enable detailed profiling for paper benchmarks
# Build with: make PROFILING=1 WITH_CUDA=1 WITH_NVCOMP=1
ifeq ($(PROFILING),1)
	CXXFLAGS += -DFASTP_PROFILING
endif
ifeq ($(CUDA_ENABLED),1)
	CXXFLAGS += -DHAVE_CUDA -I$(CUDA_ROOT)/include
	# Optional: link nvCOMP for GPU-accelerated BGZF decompression
	# Enable with: make WITH_CUDA=1 WITH_NVCOMP=1
	ifeq ($(WITH_NVCOMP),1)
		# Auto-detect nvCOMP location: prefer pip wheel (x86_64), fall back to aarch64 system pkg
		NVCOMP_PIP_DIR := $(shell python3 -c "import nvidia.libnvcomp,os;print(os.path.dirname(nvidia.libnvcomp.__file__))" 2>/dev/null)
		ifneq ($(NVCOMP_PIP_DIR),)
			NVCOMP_INC := $(NVCOMP_PIP_DIR)/include
			NVCOMP_LIB := $(NVCOMP_PIP_DIR)/lib64
		else
			NVCOMP_INC := /usr/include/nvcomp_13
			NVCOMP_LIB := /usr/lib/aarch64-linux-gnu/nvcomp/13
		endif
		CXXFLAGS    += -DHAVE_NVCOMP -I$(NVCOMP_INC)
		LIBS        += -L$(NVCOMP_LIB) -lnvcomp
		LD_FLAGS    += -Wl,-rpath,$(NVCOMP_LIB)
	endif
	# Optional: GPU-Direct Storage (NVMe -> GPU DMA via cuFile)
	# Enable with: make WITH_CUDA=1 WITH_GDS=1
	# Requires: libcufile.so, GDS-capable driver, NVMe/Lustre/GPFS filesystem
	ifeq ($(WITH_GDS),1)
		CXXFLAGS    += -DHAVE_GDS -I$(CUDA_ROOT)/include
		LIBS        += -lcufile
	endif
endif
STATIC_FLAGS := -static -Wl,--no-as-needed -pthread
LD_FLAGS := $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(LIBS) $(LD_FLAGS)
ifeq ($(CUDA_ENABLED),1)
	LD_FLAGS += -Wl,-rpath,$(CUDA_ROOT)/lib64
endif
STATIC_LD_FLAGS := $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(STATIC_FLAGS) $(LIBS) $(STATIC_LD_FLAGS)


${BIN_TARGET}:${OBJ}
	$(CXX) $(OBJ) -o $@ -flto $(LD_FLAGS)

static:${OBJ}
	$(CXX) $(OBJ) -o ${BIN_TARGET} $(STATIC_LD_FLAGS)

${DIR_OBJ}/%.o:${DIR_SRC}/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Rule to compile CUDA sources when nvcc is available
$(DIR_OBJ)/%.o: $(DIR_SRC)/%.cu
	@mkdir -p $(@D)
	$(NVCC) -c $< -o $@ \
		-ccbin g++-13 \
		-Xcompiler "-fPIC -std=c++14 -Wno-error" \
		-O3 -I${DIR_INC} --std=c++14 \
		-Wno-deprecated-declarations \
		-D__NV_GLIBC_PROVIDES_IEC_60559_FUNCS=1 \
		-DHAVE_CUDA \
		$(if $(filter 1,$(PROFILING)),-DFASTP_PROFILING,) \
		$(if $(filter 1,$(WITH_NVCOMP)),-DHAVE_NVCOMP -I$(NVCOMP_INC),) \
		$(if $(filter 1,$(WITH_GDS)),-DHAVE_GDS,) \
		$(NVCC_ARCH_FLAGS) \
		2>&1 | grep -v "redefinition" || true
.PHONY:version-check
.PHONY:version-increment
.PHONY:version-get
.PHONY:test
.PHONY:check

# Version management targets
version-check:
	@bash scripts/version_manager.sh check

version-increment:
	@bash scripts/version_manager.sh increment

version-get:
	@bash scripts/version_manager.sh get

# Built-in unit tests for the just-built binary.  Hard-fails on any failure.
# Use SKIP_TESTS=1 to bypass (NOT recommended).
test: ${BIN_TARGET}
	@if [ "$(SKIP_TESTS)" = "1" ]; then \
		echo "[WARN] SKIP_TESTS=1 - skipping unit tests"; \
	else \
		echo ">>> Running built-in unit tests: ./${BIN_TARGET} test"; \
		out=$$(./${BIN_TARGET} test 2>&1); rc=$$?; \
		echo "$$out"; \
		if [ $$rc -ne 0 ] || echo "$$out" | grep -qiE '^FAILED|: failed'; then \
			echo "[FAIL] built-in unit tests FAILED" >&2; exit 1; \
		fi; \
		echo "[OK] built-in unit tests passed"; \
	fi

# Full local test suite (built-in unit tests + CPU<->GPU regression).
check:
	@bash scripts/run_tests.sh

clean:
	@rm -rf $(DIR_OBJ)
	@rm -f $(TARGET) fastp-gpu

install:
	install $(TARGET) $(BINDIR)/$(TARGET)
	@echo "Installed."

-include $(OBJ:.o=.d)
