.PHONY: all clean data train full cpu cuda

# --- Configuration Commune ---
INCLUDE_DIR := -I include
PYTHON ?= .venv/bin/python

# --- Configuration CPU (C standard) ---
CC := gcc
CFLAGS := -Wall -Wextra $(INCLUDE_DIR) -pthread
LDFLAGS := -lm -pthread
SRC_CPU := src/cpu/neural_evol.c
TARGET_CPU := hermes_cpu

# --- Configuration GPU (CUDA) ---
NVCC := nvcc
CUFLAGS := $(INCLUDE_DIR)
SRC_GPU := src/gpu/neural_evol.cu
TARGET_GPU := hermes_cuda

# ==========================================
# COMMANDES
# ==========================================

all: cpu

cpu: $(SRC_CPU)
	@echo "[MAKE] Compiling CPU Engine..."
	$(CC) $(CFLAGS) $(SRC_CPU) -o $(TARGET_CPU) $(LDFLAGS)

cuda: $(SRC_GPU)
	@echo "[MAKE] Compiling CUDA Engine..."
	$(NVCC) $(CUFLAGS) $(SRC_GPU) -o $(TARGET_GPU)

clean:
	@echo "[MAKE] Delete executables..."
	rm -f $(TARGET_CPU) $(TARGET_GPU)

data:
	@echo "[MAKE] Downloading Data..."
	$(PYTHON) src/python/get_data.py

full: data cpu
	@echo "[MAKE] Launching Hermes (CPU)..."
	./$(TARGET_CPU)