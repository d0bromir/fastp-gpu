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

# Auto-detect a host C++ compiler accepted by this nvcc (CUDA toolkits cap the
# supported GCC version, e.g. CUDA 12.0 rejects gcc>12, CUDA 13 accepts gcc 13).
# User can override by setting NVCC_CCBIN explicitly.
ifeq ($(WITH_CUDA),1)
ifneq ($(NVCC),)
ifeq ($(origin NVCC_CCBIN),undefined)
NVCC_CCBIN := $(shell for c in g++-14 g++-13 g++-12 g++-11 g++-10 g++; do \
    command -v $$c >/dev/null 2>&1 || continue; \
    echo '' | $(NVCC) -ccbin $$c -x cu - -E >/dev/null 2>&1 && { echo $$c; break; }; \
done)
endif
ifeq ($(strip $(NVCC_CCBIN)),)
$(error No CUDA-compatible host C++ compiler found for $(NVCC). Install a supported g++ (e.g. g++-12 for CUDA 12.x) or set NVCC_CCBIN=/path/to/g++)
endif
# Use the same compatible compiler for .cpp files that include CUDA headers.
CXX ?= $(NVCC_CCBIN)
endif
endif

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
	# Override auto-detection with: NVCOMP_INC=/path NVCOMP_LIB=/path
	ifeq ($(WITH_NVCOMP),1)
		# Honor user-supplied NVCOMP_INC/NVCOMP_LIB; otherwise probe in order:
		#   1) pip wheel (x86_64 / generic)
		#   2) aarch64 system package layout (/usr/include/nvcomp_13)
		#   3) generic system layout (<sysroot>/include/nvcomp/deflate.h)
		ifeq ($(origin NVCOMP_INC),undefined)
			# Try several python interpreters (env python may differ from system python)
			NVCOMP_PIP_DIR := $(shell for py in python3 python python3.13 python3.12 python3.11 python3.10 /home/$$USER/miniconda3/bin/python3; do \
				command -v $$py >/dev/null 2>&1 || continue; \
				d=$$($$py -c "import nvidia.libnvcomp,os;print(os.path.dirname(nvidia.libnvcomp.__file__))" 2>/dev/null); \
				[ -n "$$d" ] && [ -f "$$d/include/nvcomp/deflate.h" ] && { echo $$d; break; }; \
			done)
			# Fallback: glob common pip install locations for the header directly
			ifeq ($(strip $(NVCOMP_PIP_DIR)),)
				NVCOMP_PIP_DIR := $(shell for h in \
					$$HOME/.local/lib/python*/site-packages/nvidia/libnvcomp/include/nvcomp/deflate.h \
					$$HOME/.local/lib/python*/dist-packages/nvidia/libnvcomp/include/nvcomp/deflate.h \
					$$CONDA_PREFIX/lib/python*/site-packages/nvidia/libnvcomp/include/nvcomp/deflate.h \
					/usr/lib/python*/dist-packages/nvidia/libnvcomp/include/nvcomp/deflate.h \
					/usr/local/lib/python*/dist-packages/nvidia/libnvcomp/include/nvcomp/deflate.h; do \
					[ -f $$h ] && { echo $${h%/include/nvcomp/deflate.h}; break; }; \
				done)
			endif
			ifneq ($(wildcard $(NVCOMP_PIP_DIR)/include/nvcomp/deflate.h),)
				NVCOMP_INC := $(NVCOMP_PIP_DIR)/include
				NVCOMP_LIB ?= $(NVCOMP_PIP_DIR)/lib64
			else ifneq ($(wildcard /usr/include/nvcomp_13/nvcomp/deflate.h),)
				NVCOMP_INC := /usr/include/nvcomp_13
				NVCOMP_LIB ?= /usr/lib/aarch64-linux-gnu/nvcomp/13
			else ifneq ($(wildcard /usr/include/nvcomp/deflate.h),)
				NVCOMP_INC := /usr/include
				NVCOMP_LIB ?= /usr/lib/x86_64-linux-gnu
			else ifneq ($(wildcard $(CUDA_ROOT)/include/nvcomp/deflate.h),)
				NVCOMP_INC := $(CUDA_ROOT)/include
				NVCOMP_LIB ?= $(CUDA_ROOT)/lib64
			endif
		endif
		ifeq ($(strip $(NVCOMP_INC)),)
			# Pick the correct cuXX wheel suffix from the installed nvcc major version
			NVCOMP_CUDA_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\).*/\1/p' | head -n1)
$(error WITH_NVCOMP=1 but nvCOMP headers (nvcomp/deflate.h) were not found. Install with: pip install nvidia-nvcomp-cu$(NVCOMP_CUDA_MAJOR) -- or set NVCOMP_INC=/path/to/include NVCOMP_LIB=/path/to/lib explicitly)
		endif
		ifeq ($(wildcard $(NVCOMP_INC)/nvcomp/deflate.h),)
$(error nvCOMP header not found: $(NVCOMP_INC)/nvcomp/deflate.h does not exist. Set NVCOMP_INC to the directory containing nvcomp/deflate.h)
		endif
		CXXFLAGS    += -DHAVE_NVCOMP -I$(NVCOMP_INC)
		LIBS        += -L$(NVCOMP_LIB) -lnvcomp
		LD_FLAGS    += -Wl,-rpath,$(NVCOMP_LIB)
	endif
	# Optional: GPU-Direct Storage (NVMe -> GPU DMA via cuFile)
	# Enable with: make WITH_CUDA=1 WITH_GDS=1
	# Requires: libcufile.so, GDS-capable driver, NVMe/Lustre/GPFS filesystem
	# Override auto-detection with: GDS_INC=/path GDS_LIB=/path
	#
	# Install cuFile / GPU-Direct Storage:
	#   * Ubuntu 24.04+ / Debian 12+: package is in the default repo
	#       sudo apt install libcufile-dev libcufile0
	#       # header: /usr/include/cufile.h
	#   * Older Ubuntu (22.04, 20.04) or to match a specific CUDA: use NVIDIA's
	#     CUDA apt repo (package is named `nvidia-gds` there). Adjust the
	#     distro tag (ubuntu2204, ubuntu2004, debian12, ...) and arch:
	#       wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
	#       sudo dpkg -i cuda-keyring_1.1-1_all.deb
	#       sudo apt update
	#       sudo apt install nvidia-gds          # or pin: nvidia-gds-12-9
	#       # header: $(CUDA_ROOT)/include/cufile.h
	#
	# NOTE: GDS is NOT supported on WSL2 (no direct NVMe DMA path). For
	# header-only compile testing on unsupported platforms:
	#   pip install nvidia-cufile-cu12
	#   make WITH_CUDA=1 WITH_NVCOMP=1 WITH_GDS=1 \
	#     GDS_INC=$$(python3 -c "import nvidia.cufile,os;print(os.path.dirname(nvidia.cufile.__file__))")/include \
	#     GDS_LIB=$$(python3 -c "import nvidia.cufile,os;print(os.path.dirname(nvidia.cufile.__file__))")/lib
	ifeq ($(WITH_GDS),1)
		ifeq ($(origin GDS_INC),undefined)
			# Probe common cuFile header locations
			ifneq ($(wildcard $(CUDA_ROOT)/include/cufile.h),)
				GDS_INC := $(CUDA_ROOT)/include
				GDS_LIB ?= $(CUDA_ROOT)/lib64
			else ifneq ($(wildcard $(CUDA_ROOT)/targets/x86_64-linux/include/cufile.h),)
				GDS_INC := $(CUDA_ROOT)/targets/x86_64-linux/include
				GDS_LIB ?= $(CUDA_ROOT)/targets/x86_64-linux/lib
			else ifneq ($(wildcard /usr/local/gds/include/cufile.h),)
				GDS_INC := /usr/local/gds/include
				GDS_LIB ?= /usr/local/gds/lib
			else ifneq ($(wildcard /usr/include/cufile.h),)
				GDS_INC := /usr/include
				GDS_LIB ?= /usr/lib/x86_64-linux-gnu
			endif
		endif
		ifeq ($(strip $(GDS_INC)),)
			# Detect distro to print the right install command
			GDS_DISTRO_ID    := $(shell . /etc/os-release 2>/dev/null && echo $$ID)
			GDS_DISTRO_VER   := $(shell . /etc/os-release 2>/dev/null && echo $$VERSION_ID)
			GDS_IS_WSL       := $(shell grep -qi microsoft /proc/version 2>/dev/null && echo 1)
			ifeq ($(GDS_DISTRO_ID),ubuntu)
				ifeq ($(shell printf '%s\n' "$(GDS_DISTRO_VER)" "24.04" | sort -V | head -n1),24.04)
					GDS_INSTALL_HINT := sudo apt install libcufile-dev libcufile0
				else
					GDS_INSTALL_HINT := use NVIDIA's CUDA apt repo, then 'sudo apt install nvidia-gds' (see Makefile comment above for repo setup)
				endif
			else ifeq ($(GDS_DISTRO_ID),debian)
				GDS_INSTALL_HINT := sudo apt install libcufile-dev libcufile0    # Debian 12+; older Debian: use NVIDIA's CUDA apt repo
			else ifneq ($(filter rhel centos rocky almalinux fedora,$(GDS_DISTRO_ID)),)
				GDS_INSTALL_HINT := sudo dnf install nvidia-gds                  # via NVIDIA's CUDA dnf repo
			else
				GDS_INSTALL_HINT := install the cuFile/GDS package for your distro (see Makefile comment above)
			endif
			ifeq ($(GDS_IS_WSL),1)
				GDS_WSL_NOTE := WARNING: GDS runtime I/O is not supported on WSL2; the binary will compile and link but cannot use GPU-Direct Storage at runtime.
			else
				GDS_WSL_NOTE := Note: GDS also needs a GDS-capable driver and NVMe/Lustre/GPFS filesystem at runtime.
			endif
$(error WITH_GDS=1 but cuFile headers (cufile.h) were not found. To install on this system ($(GDS_DISTRO_ID) $(GDS_DISTRO_VER)): $(GDS_INSTALL_HINT) -- or set GDS_INC=/path/to/include GDS_LIB=/path/to/lib explicitly. $(GDS_WSL_NOTE))
		endif
		ifeq ($(wildcard $(GDS_INC)/cufile.h),)
$(error cuFile header not found: $(GDS_INC)/cufile.h does not exist. Set GDS_INC to the directory containing cufile.h)
		endif
		CXXFLAGS    += -DHAVE_GDS -I$(GDS_INC)
		LIBS        += -L$(GDS_LIB) -lcufile
		LD_FLAGS    += -Wl,-rpath,$(GDS_LIB)
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

# Rule to compile CUDA sources when nvcc is available.
# Run via bash with pipefail so the grep filter for noisy "redefinition"
# messages does NOT mask a real nvcc failure (previously `|| true` swallowed
# all errors, causing silent skipped object files and confusing link errors).
$(DIR_OBJ)/%.o: $(DIR_SRC)/%.cu
	@mkdir -p $(@D)
	@/bin/bash -c 'set -o pipefail; $(NVCC) -c $< -o $@ \
		-ccbin $(NVCC_CCBIN) \
		-Xcompiler "-fPIC -std=c++14 -Wno-error" \
		-O3 -I${DIR_INC} --std=c++14 \
		-Wno-deprecated-declarations \
		-D__NV_GLIBC_PROVIDES_IEC_60559_FUNCS=1 \
		-DHAVE_CUDA \
		$(if $(filter 1,$(PROFILING)),-DFASTP_PROFILING,) \
		$(if $(filter 1,$(WITH_NVCOMP)),-DHAVE_NVCOMP -I$(NVCOMP_INC),) \
		$(if $(filter 1,$(WITH_GDS)),-DHAVE_GDS -I$(GDS_INC),) \
		$(NVCC_ARCH_FLAGS) \
		2>&1 | { grep -v "redefinition" || true; }'
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
